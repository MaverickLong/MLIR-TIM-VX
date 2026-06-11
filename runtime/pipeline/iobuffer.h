// pipeline/iobuffer.h — buffer-triple pool for the zerocopy backends.
//
// What this is for
// ----------------
// The zerocopy paths need preprocess/inference/postprocess stages to
// share host memory directly so we skip the per-iter `CopyDataToTensor`
// / `CopyDataFromTensor` memcpys. There are two zerocopy backends, each
// with slightly different buffer needs:
//
//   cpu_zero_copy: the CPU preprocessor writes the model-ready bytes
//     DIRECTLY into the NPU input tensor's bound host buffer (here:
//     `io_buf->in_data`). After `graph->Run()` the NPU output lives in
//     `io_buf->out_data` (the bound buffer for the NN output tensor)
//     and the post stage reads from there.
//
//   ppu_zero_copy: the CPU decodes the JPEG into `io_buf->pre_in_data`,
//     a per-slot host buffer bound (via SwapHandle on every iter) to
//     the PPU preprocessor graph's INPUT tensor. The PPU graph runs
//     Resize → DataConvert (u8 → fp32) on the shader; the fp32 output
//     is copied back to a small CPU scratch buffer for per-channel
//     normalize + quantize + layout-convert, which writes its final
//     bytes DIRECTLY into `io_buf->in_data` (the NN input tensor's
//     bound host buffer). So the JPEG → PPU and NN → CPU edges are
//     zerocopy via SwapHandle / CopyDataFromTensor-as-invalidate;
//     the trailing PPU → NN normalize step stays on the CPU
//     (writing once into the NN-bound buffer with no extra memcpy
//     past what the math intrinsically needs). An attempted
//     "fully-PPU" variant that folded the normalize+quant into the
//     same TIM-VX graph (Sub+Multiply with broadcast, and later
//     BatchNorm) gave wrong classifications on this driver — the
//     per-channel ops disagreed with the resize op on which physical
//     dim is C — so we keep the proven CPU normalize+quant path.
//     Each slot needs THREE aligned buffers (pre_in, in, out).
//
// On TIM-VX the binding is done via `Graph::CreateIOTensor(spec, buf)`
// once at graph build, then `Tensor::SwapHandle(buf)` per-iter to point
// at whichever Job's slot is active. Cache-coherency steps:
//
//   * Before any device Run that consumes a CPU-written buffer, the
//     producing stage calls `Tensor::FlushCacheForHandle()` to push
//     CPU writes out to DDR. The PPU stage does this for its input
//     tensor (which the CPU just wrote); the NN's input was filled by
//     the PPU graph (via DMA), so CPU cache is untouched and the NN
//     does NOT call FlushCacheForHandle on its own input — see
//     `PreProcessor::sync_for_infer()` for the per-backend hook.
//
//   * After Run, the NN output handle is brought back to CPU
//     coherence via `Tensor::CopyDataFromTensor(handle_ptr)` (a
//     self-memcpy that exists for its cache-invalidate side effect —
//     `InvalidateCacheForHandle()` is documented broken on
//     VIP9000; see `timvx_zerocopy.cpp`).
//
// For the orchestrators that have multiple in-flight Jobs (pipeline,
// pool, hybrid) we need MULTIPLE buffer triples so that pre worker A
// can be filling buf_A's pre_in while the infer thread is running with
// buf_B and post worker C is reading buf_B's out_data — otherwise
// everything would serialize on a single shared buffer. The
// IoBufferPool owns N preallocated buffer triples, hands them out via
// `acquire()`, takes them back via `release()`, and blocks `acquire()`
// when the pool is empty.
//
// Per-iteration binding
// ---------------------
// The TIM-VX graph has ONE input tensor (and ONE output tensor) per
// model. To make per-Job buffers visible to the NPU, the infer step
// calls `Tensor::SwapHandle(buf_ptr, ...)` to rebind. SwapHandle is
// not thread-safe; the orchestrators serialize it via the existing
// "single thread calls graph->Run()" invariant.
//
// Sizing
// ------
// Slot count = bounded queue depth across all stages (pre → infer →
// post). For the pipeline orchestrator with queue_cap=4 that's up to
// ~8 in-flight buffers; for pool/hybrid the typical config is N=4-8.
// Each slot allocates `in_bytes` + `out_bytes` of aligned host memory
// up-front. For resnet50 (1×3×224×224 u8 in, 1×1000 u8 out) that's
// 151 KiB per slot — trivial total.

#ifndef TIMVX_PIPELINE_IOBUFFER_H
#define TIMVX_PIPELINE_IOBUFFER_H

#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>

namespace timvx_pipeline {

struct IoBuffer {
  // PPU pre input (raw decoded RGB). Allocated only when the pool was
  // constructed with pre_in_bytes > 0 (PPU zerocopy modes); null
  // otherwise so CPU preprocessors don't pay the allocation cost.
  uint8_t* pre_in_data = nullptr;
  size_t   pre_in_size = 0;
  uint8_t* in_data  = nullptr;
  size_t   in_size  = 0;
  uint8_t* out_data = nullptr;
  size_t   out_size = 0;
  // Caller-meaningful "slot index" so we can release back to the right
  // pool entry without searching. Set by `acquire()`.
  size_t   slot     = 0;
};

class IoBufferPool {
 public:
  // Allocate `n_slots` triples of (pre_in_size, in_size, out_size)
  // buffers. `pre_in_bytes == 0` (the default) skips the pre_in
  // allocation — CPU zerocopy modes don't need a separate "raw RGB"
  // slot, only PPU pipeline modes do.
  // 64-byte aligned (a typical NPU/DMA alignment requirement; the
  // exact value isn't documented for VIP9000 but matches what
  // TIM-VX expects elsewhere).
  IoBufferPool(size_t n_slots, size_t in_bytes, size_t out_bytes,
               size_t pre_in_bytes = 0)
      : slots_(n_slots), free_(n_slots, true), free_count_(n_slots),
        in_bytes_(in_bytes), out_bytes_(out_bytes),
        pre_in_bytes_(pre_in_bytes) {
    constexpr size_t kAlign = 64;
    for (size_t i = 0; i < n_slots; ++i) {
      void* p_in = nullptr;
      void* p_out = nullptr;
      void* p_pre = nullptr;
      if (posix_memalign(&p_in,  kAlign, in_bytes)  != 0) p_in  = nullptr;
      if (posix_memalign(&p_out, kAlign, out_bytes) != 0) p_out = nullptr;
      if (pre_in_bytes > 0 &&
          posix_memalign(&p_pre, kAlign, pre_in_bytes) != 0) p_pre = nullptr;
      storage_in_.emplace_back(p_in,  &std::free);
      storage_out_.emplace_back(p_out, &std::free);
      storage_pre_.emplace_back(p_pre, &std::free);
      slots_[i].in_data    = static_cast<uint8_t*>(p_in);
      slots_[i].in_size    = in_bytes;
      slots_[i].out_data   = static_cast<uint8_t*>(p_out);
      slots_[i].out_size   = out_bytes;
      slots_[i].pre_in_data = static_cast<uint8_t*>(p_pre);
      slots_[i].pre_in_size = pre_in_bytes;
      slots_[i].slot       = i;
    }
  }

  // Blocks until a slot is free, then returns a pointer to its
  // IoBuffer. The returned pointer is owned by the pool — release via
  // `release(buf)` when the Job is done.
  IoBuffer* acquire() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&]{ return free_count_ > 0; });
    for (size_t i = 0; i < free_.size(); ++i) {
      if (free_[i]) {
        free_[i] = false;
        --free_count_;
        return &slots_[i];
      }
    }
    // Unreachable (free_count_ > 0 guarantees a free slot).
    return nullptr;
  }

  // Mark a slot free; wakes one acquire().
  void release(IoBuffer* buf) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!buf) return;
    size_t i = buf->slot;
    if (i < free_.size() && !free_[i]) {
      free_[i] = true;
      ++free_count_;
      cv_.notify_one();
    }
  }

  size_t in_bytes()     const { return in_bytes_;  }
  size_t out_bytes()    const { return out_bytes_; }
  size_t pre_in_bytes() const { return pre_in_bytes_; }
  size_t n_slots()      const { return slots_.size(); }

  // For setup: hand the FIRST slot's buffers to CreateIOTensor as the
  // initial binding. The orchestrator's infer step will SwapHandle
  // away from this on each Run() to whichever Job's slot is active.
  IoBuffer* peek_slot0() { return slots_.empty() ? nullptr : &slots_[0]; }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::vector<IoBuffer> slots_;
  std::vector<bool> free_;
  size_t free_count_;
  std::vector<std::unique_ptr<void, void(*)(void*)>> storage_in_;
  std::vector<std::unique_ptr<void, void(*)(void*)>> storage_out_;
  std::vector<std::unique_ptr<void, void(*)(void*)>> storage_pre_;
  size_t in_bytes_, out_bytes_;
  size_t pre_in_bytes_ = 0;
};

} // namespace timvx_pipeline

#endif // TIMVX_PIPELINE_IOBUFFER_H
