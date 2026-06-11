// Declaration: AI involved in the bug fixing (PPU pre/post) of this file.
// 
// pipeline/factory.h — orchestrator+pre/post factory keyed on mode names.
//
// Sixteen mode strings, mapping the user's 4-orchestrator × 4-backend
// matrix:
//
//   "cpu-sequential"            "cpu-pipeline"            "cpu-pool"            "cpu-hybrid"
//   "cpu_zero_copy-sequential"  "cpu_zero_copy-pipeline"  "cpu_zero_copy-pool"  "cpu_zero_copy-hybrid"
//   "ppu-sequential"            "ppu-pipeline"            "ppu-pool"            "ppu-hybrid"
//   "ppu_zero_copy-sequential"  "ppu_zero_copy-pipeline"  "ppu_zero_copy-pool"  "ppu_zero_copy-hybrid"
//
// The cpu_zero_copy backend has the same CPU preprocessing logic as
// cpu, but threads its output bytes directly into the NPU input
// tensor's bound host buffer (no `CopyDataToTensor` memcpy). See
// `pipeline/iobuffer.h` and the `infer_one_zerocopy` helper in
// `pipeline/orchestrator.h` for the cache-coherency dance this
// requires on VIP9000.
//
// The ppu_zero_copy backend is the same idea applied to the PPU
// pre + NN path: the libjpeg-decoded RGB bytes land in
// `io_buf->pre_in_data` (bound to the PPU preprocessor graph's INPUT
// tensor); the PPU graph's OUTPUT is bound to `io_buf->in_data`
// which is in turn the NN graph's INPUT binding; the NN's OUTPUT is
// bound to `io_buf->out_data` which the CPU post-processor reads.
// Three "edges" — CPU→PPU, PPU→NN, NN→CPU — are each zerocopy. See
// `PpuJpegPreProcessor::process_inplace`.
//
// The factory wires up the orchestrator with the right pre/post
// processor pair. Modes are mostly there for benchmarking — see
// `benchmark.h` for the harness that submits N inputs and prints
// throughput stats.
//
// Choice of pre/post backend:
//   * cpu-* / cpu_zero_copy-* : CpuJpegPreProcessor + CpuTopKPostProcessor
//   * ppu-* / ppu_zero_copy-* : PpuJpegPreProcessor + CpuTopKPostProcessor
//
// Both backends take raw JPEG file bytes as input (the contents of an
// e.g. cat105.jpg). The CPU backend decodes / resizes / normalises /
// quantises on the host; the PPU backend decodes on the host and runs
// resize + dequant + normalize + requantize as a SINGLE TIM-VX/PPU
// graph that writes bytes in the model's expected layout — no CPU
// normalize/requantize step. See `pipeline/preproc.h`.
//
// The post stage is CPU-only in every mode — vendor's reference
// pipeline matches this (no softmax in post; raw dequantized logits
// go straight into top-K, since softmax preserves argmax). A previous
// `PpuSoftMaxTopKPostProcessor` ran softmax on the shader unit, but
// that introduced a hard-to-debug shader-vs-shader concurrency wedge
// when running rn18 with PPU pre + PPU post simultaneously — see the
// removal note in `pipeline/postproc.h`.
//
// PPU source shape
// ----------------
// The PPU graph is built for a fixed (src_w, src_h, 3) input. The
// runner probes the JPEG header to get those dimensions and passes
// them to `make_backend()`; the IoBufferPool's pre_in_bytes is
// `src_w * src_h * 3` so each slot has room for one decoded RGB image.
//
// Env-var `TIMVX_PASSTHROUGH_INPUT=1` opts back into the legacy
// already-quantized-bytes-from-Python path; intended for compatibility
// with `run_timvx.py`'s old per-iter blob format. PPU + passthrough
// is silently downgraded to CPU passthrough (the PPU graph has no
// useful work to do on already-quantized bytes).

#ifndef TIMVX_PIPELINE_FACTORY_H
#define TIMVX_PIPELINE_FACTORY_H

#include <array>
#include <cstring>
#include <memory>
#include <string>

#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"

#include "pipeline/input_spec.h"
#include "pipeline/iobuffer.h"
#include "pipeline/orchestrator.h"
#include "pipeline/postproc.h"
#include "pipeline/preproc.h"

namespace timvx_pipeline {
struct Backend {
  std::unique_ptr<PreProcessor> pre;
  std::unique_ptr<PostProcessor> post;
};

struct ModeConfig {
  std::string mode;      // "cpu-pipeline", "ppu-hybrid", "cpu_zero_copy-pool"...
  size_t num_pool_workers = 4;     // worker count for pool / hybrid
  size_t queue_cap        = 8;     // for pipeline/hybrid/pool
  // Zerocopy IoBuffer pool size. Has to be large enough that all
  // in-flight Jobs across stages can hold a slot at once. Set
  // automatically by `make_orchestrator()` based on strategy + worker
  // counts; can also be overridden via TIMVX_IOPOOL_SLOTS env.
  size_t iopool_slots     = 0;
};

inline bool mode_is_cpu_zerocopy(const std::string& m) {
  return m.rfind("cpu_zero_copy-", 0) == 0;
}
inline bool mode_is_ppu_zerocopy(const std::string& m) {
  return m.rfind("ppu_zero_copy-", 0) == 0;
}
inline bool mode_is_zerocopy(const std::string& m) {
  return mode_is_cpu_zerocopy(m) || mode_is_ppu_zerocopy(m);
}
inline bool mode_is_ppu(const std::string& m) {
  return m.rfind("ppu-", 0) == 0 || mode_is_ppu_zerocopy(m);
}

// ImageNet defaults — match `inputmeta.yml` from the vendor's resnet50_tflite.
inline std::array<float, 3> kImagenetMean{0.485f, 0.456f, 0.406f};
inline std::array<float, 3> kImagenetStd {0.229f, 0.224f, 0.225f};
// Conservative source-shape fallback used only when the caller can't
// probe the input. Real bench runs pass actual (src_w, src_h) into
// `make_backend()` so the PPU graph and the IoBufferPool's pre_in
// slot are sized correctly.
inline std::array<uint32_t, 2> kPpuSourceShapeFallback{224, 224};  // W, H

// `ppu_src_shape` = {src_w, src_h} of the (CPU-decoded) raw RGB
// image the PPU graph will resize. Caller probes the JPEG header
// (libjpeg `read_header`) to get the actual dims and passes them in.
inline Backend make_backend(const std::string& mode,
                             const InputSpec& spec,
                             uint32_t num_classes,
                             double output_scale,
                             int32_t output_zp,
                             std::array<uint32_t, 2> ppu_src_shape
                                 = kPpuSourceShapeFallback) {
  Backend b;
  bool passthrough = std::getenv("TIMVX_PASSTHROUGH_INPUT") != nullptr;
  (void)output_scale; (void)output_zp; (void)num_classes;
  if (mode_is_ppu(mode)) {
    if (passthrough) {
      // PPU + passthrough doesn't make sense — the input is already in
      // the model's quantized layout, so there's no resize / normalize
      // / requantize work for the PPU. Silently fall back to CPU
      // passthrough (which still works under the ppu-* mode in terms
      // of orchestrator + postproc selection).
      b.pre = std::make_unique<CpuPassthroughPreProcessor>(spec);
    } else {
      // The shared shader-unit mutex. Currently only the PPU
      // preprocessor dispatches on the shader; allocated here as a
      // shared_ptr so any future shader-using post-processor can
      // take the same lock and avoid the wedge documented in
      // CLAUDE.md.
      auto shader_mu = std::make_shared<std::mutex>();
      b.pre = std::make_unique<PpuJpegPreProcessor>(
          spec, kImagenetMean, kImagenetStd,
          ppu_src_shape[0], ppu_src_shape[1],
          /*zerocopy=*/mode_is_ppu_zerocopy(mode),
          shader_mu);
    }
    // Match the vendor reference pipeline: dequantize + top-K on
    // CPU, no softmax (it doesn't change argmax-ordering anyway).
    b.post = std::make_unique<CpuTopKPostProcessor>(/*K=*/5);
  } else {
    // CPU and CPU_ZERO_COPY share the SAME preprocessors and
    // postprocessors — the difference is only how the bytes flow to
    // and from the NPU's input/output tensors. The zerocopy path uses
    // PreProcessor::process_inplace + PostProcessor::process_inplace
    // (both have sensible default implementations on the base class,
    // and the CpuJpeg + CpuTopK paths override or extend them with
    // direct-buffer-write/read implementations).
    if (passthrough) {
      b.pre = std::make_unique<CpuPassthroughPreProcessor>(spec);
    } else {
      b.pre = std::make_unique<CpuJpegPreProcessor>(
          spec, kImagenetMean, kImagenetStd);
    }
    b.post = std::make_unique<CpuTopKPostProcessor>(/*K=*/5);
  }
  return b;
}

// Returns an Orchestrator wired up for `cfg.mode`.
//
// `graphs` carries one WorkerGraph per worker thread for the pool
// strategy (each worker owns its own model graph, so multiple Run()s
// fire in parallel); for sequential / pipeline / hybrid only
// `graphs[0]` is consumed — those strategies all share a single graph
// instance and serialize infer structurally or via a mutex.
//
// When the mode is a zerocopy variant, the caller MUST have supplied
// a non-null `pool` whose slot count is ≥ the strategy's worst-case
// in-flight Job count (see ModeConfig::iopool_slots; the factory
// computes reasonable defaults below). For non-zerocopy modes, `pool`
// is ignored.
inline std::unique_ptr<Orchestrator>
make_orchestrator(const ModeConfig& cfg,
                  std::vector<WorkerGraph> graphs,
                  PreProcessor* pre, PostProcessor* post,
                  IoBufferPool* pool = nullptr) {
  if (graphs.empty()) {
    std::fprintf(stderr, "make_orchestrator: empty graphs vector\n");
    return nullptr;
  }
  // Strip the backend prefix to get the orchestration strategy.
  auto pos = cfg.mode.find('-');
  std::string strat = (pos == std::string::npos) ? cfg.mode
                                                   : cfg.mode.substr(pos + 1);
  // Single-graph strategies just consume graphs[0].
  const WorkerGraph& g0 = graphs.front();
  if (strat == "sequential")
    return std::make_unique<SequentialOrchestrator>(
        g0.graph, g0.in, g0.out, pre, post, pool);
  if (strat == "pipeline")
    return std::make_unique<PipelineOrchestrator>(
        g0.graph, g0.in, g0.out, pre, post, cfg.queue_cap, pool);
  if (strat == "hybrid")
    return std::make_unique<HybridPoolOrchestrator>(
        g0.graph, g0.in, g0.out, pre, post,
        cfg.num_pool_workers, cfg.queue_cap, pool);
  if (strat == "pool") {
    if (graphs.size() < cfg.num_pool_workers) {
      std::fprintf(stderr,
                   "make_orchestrator: pool strategy needs %zu WorkerGraphs, "
                   "got %zu\n",
                   cfg.num_pool_workers, graphs.size());
      return nullptr;
    }
    graphs.resize(cfg.num_pool_workers);
    return std::make_unique<ThreadPoolOrchestrator>(
        std::move(graphs), pre, post, cfg.queue_cap, pool);
  }
  std::fprintf(stderr,
               "unknown orchestration strategy in mode '%s' "
               "(expected one of: sequential, pipeline, pool, hybrid)\n",
               cfg.mode.c_str());
  return nullptr;
}

// How many WorkerGraph instances the runner needs to build for `cfg`.
// Pool uses one per worker; the other strategies all share one graph.
inline size_t graphs_needed(const ModeConfig& cfg) {
  auto pos = cfg.mode.find('-');
  std::string strat = (pos == std::string::npos) ? cfg.mode
                                                   : cfg.mode.substr(pos + 1);
  return strat == "pool" ? cfg.num_pool_workers : 1;
}

// Compute a reasonable IoBuffer pool size for `cfg`. The pool needs at
// least one slot per in-flight Job across all stages:
//   * sequential : 1
//   * pipeline   : 1 (pre) + 1 (infer) + 1 (post) = 3, but bounded
//                  queues add their depth → 3 + 2*queue_cap is safe.
//   * pool       : 1 per worker, with one slot per queue head:
//                  num_pool_workers + queue_cap.
//   * hybrid     : num_pool_workers + queue_cap (same shape as pool —
//                  each worker holds a slot from pre to post, queue
//                  buffers a few more).
// We pick conservative defaults so the pool never starves.
inline size_t default_iopool_slots(const ModeConfig& cfg) {
  auto pos = cfg.mode.find('-');
  std::string strat = (pos == std::string::npos) ? cfg.mode
                                                   : cfg.mode.substr(pos + 1);
  if (strat == "sequential") return 2;
  if (strat == "pipeline")   return 3 + 2 * cfg.queue_cap;
  if (strat == "pool")       return cfg.num_pool_workers + cfg.queue_cap;
  if (strat == "hybrid")     return cfg.num_pool_workers + cfg.queue_cap;
  return 8;
}

inline bool is_valid_mode(const std::string& m) {
  static const char* kModes[] = {
    "cpu-sequential",            "cpu-pipeline",            "cpu-pool",            "cpu-hybrid",
    "cpu_zero_copy-sequential",  "cpu_zero_copy-pipeline",  "cpu_zero_copy-pool",  "cpu_zero_copy-hybrid",
    "ppu-sequential",            "ppu-pipeline",            "ppu-pool",            "ppu-hybrid",
    "ppu_zero_copy-sequential",  "ppu_zero_copy-pipeline",  "ppu_zero_copy-pool",  "ppu_zero_copy-hybrid",
  };
  for (auto* k : kModes) if (m == k) return true;
  return false;
}

} // namespace timvx_pipeline

#endif // TIMVX_PIPELINE_FACTORY_H
