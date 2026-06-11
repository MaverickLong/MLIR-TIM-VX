// Declaration: AI involved in the bug fixing (PPU pre/post) of this file.
// 
// pipeline/postproc.h — `PostProcessor` interface + implementation.
//
// A PostProcessor turns the model's output tensor (still on the device
// in TIM-VX innermost-first byte order) into a typed `InferenceResult`
// containing top-K class IDs and inference time.
//
// Provided implementation
// -----------------------
//   CpuTopKPostProcessor : dequantize → top-K on CPU. Used by every
//                          mode (CPU and PPU). The vendor's reference
//                          pipeline does no softmax at this stage —
//                          raw dequantized logits go straight into
//                          top-K — so we match that.
//
// (A previous `PpuSoftMaxTopKPostProcessor` ran softmax on the shader
// unit before top-K. It was removed: (a) the softmax step doesn't
// change argmax-ordering anyway, (b) running another shader graph
// concurrently with `PpuJpegPreProcessor`'s shader graph wedged the
// driver on rn18 — see CLAUDE.md's "graph->Run failed after 23222.64
// ms" note — because the chip's single shader unit can't accept
// concurrent dispatch from independently-locked graphs.)
//
// `process()` must be reentrant — orchestrators may call it from
// multiple threads.

#ifndef TIMVX_PIPELINE_POSTPROC_H
#define TIMVX_PIPELINE_POSTPROC_H

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"
#include "tim/vx/types.h"

#include "pipeline/byte_layout.h"
#include "pipeline/tensor_io.h"

namespace timvx_pipeline {

struct InferenceResult {
  // Filled by orchestrator (the NPU stage time), not by the postprocessor.
  double infer_ms = 0.0;
  // Top-K. Default K=5; the caller sets that on the postprocessor.
  std::vector<TopKEntry> topk;
  // Set to non-empty on failure paths; orchestrators propagate this
  // to their futures.
  std::string error;
  // Wall-clock timestamp captured just before the orchestrator sets
  // the promise (i.e. immediately after post completes). The bench
  // harness pairs this with its own pre-submit timestamp to derive
  // the per-request end-to-end latency. Using a time_point (not a
  // millisecond offset) keeps the orchestrator from needing to know
  // the harness's t0.
  std::chrono::steady_clock::time_point complete_tp;
  // Per-stage wall-clock stamps, captured by the orchestrator so the
  // bench harness can compute pure pre/post processing times AND the
  // wait time spent between stages (mutex contention, queue wait,
  // tensor copy/SwapHandle setup). Defined edges:
  //   pre_ms       = pre_end_tp     - pre_start_tp
  //   pre_wait_ms  = infer_start_tp - pre_end_tp      (incl. copy/swap setup)
  //   infer_ms     = infer_end_tp   - infer_start_tp  (also reported in infer_ms)
  //   post_wait_ms = post_start_tp  - infer_end_tp    (incl. cache invalidate
  //                                                     + queue-pop)
  //   post_ms      = post_end_tp    - post_start_tp
  // For Sequential the *_wait quantities are ~0; for Pipeline / Pool /
  // Hybrid they pick up queue / mutex time.
  std::chrono::steady_clock::time_point pre_start_tp;
  std::chrono::steady_clock::time_point pre_end_tp;
  std::chrono::steady_clock::time_point infer_start_tp;
  std::chrono::steady_clock::time_point infer_end_tp;
  std::chrono::steady_clock::time_point post_start_tp;
  std::chrono::steady_clock::time_point post_end_tp;
};

class PostProcessor {
 public:
  virtual ~PostProcessor() = default;
  virtual void process(const std::shared_ptr<tim::vx::Tensor>& out,
                       InferenceResult& result) = 0;
  // Zero-copy variant: the caller has already pulled the model's
  // output bytes into a host buffer (via the post-Run
  // `CopyDataFromTensor(out_buf)` which doubles as the cache-invalidate
  // trigger on this chip). The post-processor reads those bytes
  // directly. The `meta` tensor is only used to read shape / dtype /
  // quant context — its CopyDataFromTensor is NOT called here. Default
  // impl falls back to the regular `process()` path (which DOES touch
  // the tensor); overrides that genuinely consume `out_buf` skip the
  // second CopyDataFromTensor.
  virtual void process_inplace(const uint8_t* out_buf, size_t out_bytes,
                                const std::shared_ptr<tim::vx::Tensor>& meta,
                                InferenceResult& result) {
    (void)out_buf; (void)out_bytes;
    process(meta, result);
  }
  virtual const char* name() const = 0;
};

// ── 1. CPU dequant → top-K ──────────────────────────────────────────────
class CpuTopKPostProcessor : public PostProcessor {
 public:
  explicit CpuTopKPostProcessor(size_t K = 5) : K_(K) {}
  void process(const std::shared_ptr<tim::vx::Tensor>& out,
               InferenceResult& result) override {
    std::vector<float> buf;
    if (!dequantize_output(out, buf)) {
      result.error = "dequantize_output failed";
      return;
    }
    result.topk = compute_topk(buf, K_);
  }
  // Zero-copy path: dequant from `out_buf` directly (no second
  // CopyDataFromTensor on the tensor). `meta` provides shape / dtype /
  // (S, Z); the BYTES in `out_buf` are already the device's output
  // post-Run + post-CopyDataFromTensor-as-invalidate.
  void process_inplace(const uint8_t* out_buf, size_t out_bytes,
                        const std::shared_ptr<tim::vx::Tensor>& meta,
                        InferenceResult& result) override {
    const auto& shape = meta->GetShape();
    size_t numel = 1;
    for (auto d : shape) numel *= d;
    if (numel == 0) return;
    auto dtype = meta->GetDataType();
    auto quant = meta->GetQuantization();
    float scale = 1.0f; int32_t zp = 0;
    if (quant.Type() != tim::vx::QuantType::NONE && !quant.Scales().empty()) {
      scale = quant.Scales()[0];
      if (!quant.ZeroPoints().empty()) zp = quant.ZeroPoints()[0];
    }
    size_t elem = bytesPerElem(dtype);
    if (out_bytes < numel * elem) {
      result.error = "out_buf too small";
      return;
    }
    // The bytes in `out_buf` are in TIM-VX innermost-first order.
    // Layout-flip back to MLIR row-major before the argmax — top-K
    // doesn't actually care about layout (it's a flat reduction), but
    // we keep the conversion in one place so downstream consumers
    // that DO use shape indices stay correct.
    std::vector<uint8_t> mlir_bytes(numel * elem);
    layoutConvert(out_buf, mlir_bytes.data(), shape, elem,
                  /*from_mlir_to_tvx=*/false);
    std::vector<float> buf(numel);
    if (dtype == tim::vx::DataType::FLOAT32) {
      std::memcpy(buf.data(), mlir_bytes.data(), numel * sizeof(float));
    } else if (dtype == tim::vx::DataType::INT8) {
      auto* raw = reinterpret_cast<const int8_t*>(mlir_bytes.data());
      for (size_t i = 0; i < numel; ++i)
        buf[i] = (static_cast<float>(raw[i]) - static_cast<float>(zp)) * scale;
    } else if (dtype == tim::vx::DataType::UINT8) {
      auto* raw = mlir_bytes.data();
      for (size_t i = 0; i < numel; ++i)
        buf[i] = (static_cast<float>(raw[i]) - static_cast<float>(zp)) * scale;
    } else {
      result.error = "unsupported output dtype for zerocopy postproc";
      return;
    }
    result.topk = compute_topk(buf, K_);
  }
  const char* name() const override { return "cpu-topk"; }
 private:
  size_t K_;
};

} // namespace timvx_pipeline

#endif // TIMVX_PIPELINE_POSTPROC_H
