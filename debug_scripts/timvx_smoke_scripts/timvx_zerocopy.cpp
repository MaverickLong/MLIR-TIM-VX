// timvx_zerocopy.cpp — does TIM-VX honor a CPU-allocated backing buffer?
//
// The runner template currently round-trips every input/output through
// CopyDataToTensor / CopyDataFromTensor, which means a memcpy each
// inference even though the SoC is unified-memory. This smoke verifies
// the alternative: hand TIM-VX a pointer the host owns, let the NPU
// write into the same physical memory, and read the result straight
// from the host pointer with no separate device→host transfer.
//
// What the bisect found (see ../*.cpp /tmp/timvx_iter*.cpp from the
// investigation; numbers also in this file's commit message):
//
//   1. Graph::CreateIOTensor(spec, void* data) is the zero-copy entry
//      point. For INPUT/OUTPUT tensors it forwards `data` as
//      `external_cache` to vsi_nn_AddTensorFromHandle — the tensor's
//      vx handle is backed by the caller's buffer. CreateTensor(spec,
//      const void*) — what the current runner uses — allocates
//      TIM-VX-owned storage and memcpys the user buffer into it once,
//      then drops the pointer. That's the path this test displaces.
//
//   2. Buffer must be page/cacheline-aligned. The lenet sample uses
//      aligned_alloc(64, ...); we follow that.
//
//   3. INPUT: after the CPU writes new input bytes,
//          input_t->FlushCacheForHandle();
//      pushes the CPU writes through to where the NPU reads. Works on
//      every iteration — proven by feeding fresh input each iter and
//      seeing the NPU's outputs track them.
//
//   4. OUTPUT: the API is misleading.
//        * Tensor::InvalidateCacheForHandle() returns true but does
//          NOT actually invalidate the CPU cache — its body only calls
//          vsi_nn_GetTensorHandle (a getter) and never reaches the
//          underlying vsi_nn_InvalidateHandle / vxInvalidateHandleVSI.
//          Direct reads from the host buffer after Run() can return
//          stale (zero) bytes from before the kernel wrote.
//        * Tensor::map(/*invalidate=*/true) has the same bug — also
//          only calls vsi_nn_GetTensorHandle.
//        * The one call that DOES trigger vsi_nn_InvalidateHandle is
//          Tensor::CopyDataFromTensor(buf). It runs the invalidate
//          first and then a memcpy(buf, handle_ptr, nbytes). If you
//          pass the same pointer you handed to CreateIOTensor, the
//          memcpy is self-to-self — effectively just the cache
//          invalidate, which is what we wanted.
//
//      So the canonical zero-copy OUTPUT-read pattern is:
//          output_t->CopyDataFromTensor(out_buf.ptr);  // == handle ptr
//          // read out_buf.ptr directly
//      The self-memcpy adds a single line of memory traffic over the
//      buffer once — much cheaper than the full device→host copy the
//      current runner does after each Run.
//
//   5. The buffer must outlive the tensor.
//
// The op is Pow(x, 2.0) — logically unary but expressed as a binary
// elementwise where the exponent is a broadcast-scalar CONSTANT. FP32
// Pow is well-supported on this SoC per the op-probe matrix, so any
// failure here points at the zero-copy plumbing rather than the kernel.
//
// We run four iterations on the same compiled graph. Each iteration
// overwrites the input in place AFTER compile — if the NPU's outputs
// track the new values, the input buffer is genuinely shared, not
// copied at create time.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"
#include "tim/vx/types.h"
#include "tim/vx/ops/elementwise.h"  // Pow

namespace {

#define MEM_ALIGN(x, a) (((x) + ((a) - 1)) & ~((a) - 1))
constexpr size_t kAlign = 64;
constexpr size_t kN     = 8;

struct AlignedFloatBuf {
  float* ptr{nullptr};
  size_t nbytes{0};
  explicit AlignedFloatBuf(size_t numel) {
    nbytes = MEM_ALIGN(numel * sizeof(float), kAlign);
    ptr    = static_cast<float*>(aligned_alloc(kAlign, nbytes));
    if (!ptr) {
      std::fprintf(stderr, "aligned_alloc(%zu, %zu) failed\n", kAlign, nbytes);
      std::abort();
    }
    std::memset(ptr, 0, nbytes);
  }
  ~AlignedFloatBuf() { std::free(ptr); }
  AlignedFloatBuf(const AlignedFloatBuf&)            = delete;
  AlignedFloatBuf& operator=(const AlignedFloatBuf&) = delete;
};

bool verify_squares(const float* observed, const float* input, size_t n,
                    int iter) {
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    const float expect = input[i] * input[i];
    const float diff   = std::fabs(observed[i] - expect);
    // Pow on the NPU is exp/log under the hood — small ulp drift expected.
    if (diff > 1e-4f * std::fabs(expect) + 1e-5f) {
      if (bad < 4) {
        std::printf("  iter%d mismatch[%zu]: got=%.6f want=%.6f (x=%.6f)\n",
                    iter, i, observed[i], expect, input[i]);
      }
      ++bad;
    }
  }
  std::printf("  iter%d %s  in=[", iter, bad == 0 ? "PASS" : "FAIL");
  for (size_t i = 0; i < n; ++i)
    std::printf("%.2f%s", input[i], i + 1 < n ? " " : "] out=[");
  for (size_t i = 0; i < n; ++i)
    std::printf("%.2f%s", observed[i], i + 1 < n ? " " : "]\n");
  return bad == 0;
}

}  // namespace

int main() {
  using DT = tim::vx::DataType;
  using TA = tim::vx::TensorAttribute;

  AlignedFloatBuf in_buf(kN);
  AlignedFloatBuf out_buf(kN);

  // Prefill the input. Subsequent iters will overwrite in place to
  // prove the NPU is reading our buffer each Run, not a snapshot.
  for (size_t i = 0; i < kN; ++i) in_buf.ptr[i] = static_cast<float>(i + 1);

  auto ctx = tim::vx::Context::Create();
  if (!ctx) { std::printf("Context::Create returned null\n"); return 1; }
  auto g = ctx->CreateGraph();
  if (!g)   { std::printf("CreateGraph returned null\n");    return 1; }

  // Hand the host buffers straight to TIM-VX.
  tim::vx::TensorSpec in_spec (DT::FLOAT32, {kN}, TA::INPUT);
  tim::vx::TensorSpec out_spec(DT::FLOAT32, {kN}, TA::OUTPUT);
  auto in_t  = g->CreateIOTensor(in_spec,  in_buf.ptr);
  auto out_t = g->CreateIOTensor(out_spec, out_buf.ptr);
  if (!in_t || !out_t) {
    std::printf("CreateIOTensor returned null\n");
    return 1;
  }

  // Sanity: TIM-VX really is using in_buf.ptr / out_buf.ptr as the handle
  // backing store. map() with invalidate=false just returns data_, which
  // equals the pointer we passed in if the IO path is wired right.
  void* in_mapped  = in_t->map(false);
  void* out_mapped = out_t->map(false);
  in_t->unmap();
  out_t->unmap();
  std::printf("handle check: in_t=%p in_buf=%p same=%s | out_t=%p out_buf=%p same=%s\n",
              in_mapped,  static_cast<void*>(in_buf.ptr),
              in_mapped == in_buf.ptr ? "yes" : "NO",
              out_mapped, static_cast<void*>(out_buf.ptr),
              out_mapped == out_buf.ptr ? "yes" : "NO");

  // Exponent: broadcast scalar 2.0f. CONSTANT data captured at create time.
  tim::vx::TensorSpec exp_spec(DT::FLOAT32, {1}, TA::CONSTANT);
  float exp_val = 2.0f;
  auto exp_t = g->CreateTensor(exp_spec, &exp_val);
  if (!exp_t) { std::printf("Failed to create exponent tensor\n"); return 1; }

  auto op = g->CreateOperation<tim::vx::ops::Pow>();
  (*op).BindInputs({in_t, exp_t}).BindOutput(out_t);

  if (!g->Compile()) { std::printf("graph->Compile failed\n"); return 1; }

  bool ok = true;
  for (int iter = 0; iter < 4; ++iter) {
    // Rewrite the input buffer in place. iter 0 keeps the pre-create
    // values; iters 1..3 mutate, to prove TIM-VX is re-reading the
    // buffer each Run.
    if (iter > 0) {
      for (size_t i = 0; i < kN; ++i)
        in_buf.ptr[i] = static_cast<float>(i + 1) + 0.25f * iter;
    }

    in_t->FlushCacheForHandle();

    if (!g->Run()) {
      std::printf("graph->Run failed (iter %d)\n", iter);
      return 1;
    }

    // OUTPUT cache invalidate: see file header for why this and not
    // InvalidateCacheForHandle() / map(true). The self-memcpy is the
    // bargain we strike for the cache invalidate.
    out_t->CopyDataFromTensor(out_buf.ptr);

    // Read directly from out_buf.ptr — the actual zero-copy claim.
    ok &= verify_squares(out_buf.ptr, in_buf.ptr, kN, iter);
  }

  std::printf("---\n%s\n", ok ? "zero-copy OK" : "zero-copy BROKEN");
  return ok ? 0 : 1;
}
