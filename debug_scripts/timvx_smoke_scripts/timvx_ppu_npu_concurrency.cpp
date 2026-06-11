// timvx_ppu_npu_concurrency.cpp — measure whether the A733's shader/PPU
// path and the NN-engine NPU can run concurrently across two independent
// tim::vx graphs, two contexts, two threads.
//
// If they truly run in parallel, we can fold preprocessing into NPU time
// for free — the question this binary answers.
//
// Setup
// -----
//   Preproc graph (shader/PPU path):
//       u8 480x360x3x1            (asym quant, scale=1/127.5, zp=0)
//         → Resize bilinear
//       u8 224x224x3x1
//         → DataConvert (dequantize)
//       fp32 224x224x3x1
//
//   ResNet50 graph (NN-engine NPU path):
//       u8 {1,3,224,224}          (asym quant, scale=0.1546, zp=114)
//         → timvx_main(...)       (the full lowered ResNet50 v1 body
//                                  from example/lower_out/resnet50_v1/)
//       u8 {1,1000}
//
// Three modes timed back-to-back
// ------------------------------
//   A) preproc only          — 100 Run() iters on the main thread
//   B) ResNet50 only         — 100 Run() iters on the main thread
//   C) both concurrent       — thread 1 spins preproc 100x; thread 2 spins
//                              ResNet50 100x; both threads start their hot
//                              loop at the same instant (barrier sync).
//
// Mode C deliberately does NOT take a global mutex around Run().
// The point is to see whether the underlying libGAL / OVXLIB stack can
// dispatch two independent graphs in parallel. Vivante's own
// TIM-VX/samples/multi_thread_test wraps Run() in a global mutex; this
// test is what tells us whether that mutex is necessary or just defensive.
//
// Verdict heuristic
// -----------------
//   serial_baseline = total(A) + total(B)
//   concurrent_total = max(thread1 time, thread2 time) in mode C
//   speedup = serial_baseline / concurrent_total
//     >= 1.6x  → PARALLEL          (PPU and NPU are independent)
//     1.2–1.6x → PARTIAL OVERLAP   (some queue sharing)
//     <  1.2x  → SERIALIZED        (one device or one dispatch queue)
//
// Build/run:  bash timvx_ppu_npu_concurrency.bash

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"
#include "tim/vx/types.h"
#include "tim/vx/ops/resize.h"
#include "tim/vx/ops/simple_operations.h"  // DataConvert

// timvx_runtime.h pulls in the shared helpers used by every generated
// runner (mmap_const, conv2d, fully_connected, ...) and is required by
// the included resnet50_v1.func.cpp below.
#include "timvx_runtime.h"
#include "resnet50_v1.func.cpp"  // defines timvx_main(graph, input)

namespace {

using DT = tim::vx::DataType;
using TA = tim::vx::TensorAttribute;
using Clock = std::chrono::steady_clock;

constexpr int kWarmup = 5;
constexpr int kIters  = 100;

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct Stats { double mean, median, min, max, p99; };
Stats summarize(std::vector<double> ms) {
  std::sort(ms.begin(), ms.end());
  Stats s{};
  s.min    = ms.front();
  s.max    = ms.back();
  s.median = ms[ms.size() / 2];
  s.p99    = ms[std::min<size_t>(ms.size() - 1, ms.size() * 99 / 100)];
  s.mean   = std::accumulate(ms.begin(), ms.end(), 0.0) / ms.size();
  return s;
}
void print_stats(const char* tag, const Stats& s, double total_ms) {
  std::printf("  [%-7s] total=%8.2f ms   per-iter: mean=%7.3f  median=%7.3f  "
              "min=%7.3f  p99=%7.3f  max=%7.3f ms\n",
              tag, total_ms, s.mean, s.median, s.min, s.p99, s.max);
}

// ── Preprocessing graph (shader / PPU) ───────────────────────────────
struct PreprocGraph {
  std::shared_ptr<tim::vx::Context> ctx;
  std::shared_ptr<tim::vx::Graph>   graph;
  std::shared_ptr<tim::vx::Tensor>  in_t, fp32_t;
};
PreprocGraph build_preproc() {
  PreprocGraph G;
  G.ctx   = tim::vx::Context::Create();
  G.graph = G.ctx->CreateGraph();

  // u8 480×360×3×1 input — typical decoder output. quant=(1/127.5, 0)
  // so the DataConvert at the tail produces fp32 in [-128/127.5, +127/127.5].
  tim::vx::Quantization u8q(tim::vx::QuantType::ASYMMETRIC, 1.0f / 127.5f, 0);
  G.in_t = G.graph->CreateTensor(tim::vx::TensorSpec(
      DT::UINT8, {480, 360, 3, 1}, TA::INPUT, u8q));

  // Stage 1 — Resize bilinear → 224×224×3×1 (still u8).
  auto resized = G.graph->CreateTensor(tim::vx::TensorSpec(
      DT::UINT8, {224, 224, 3, 1}, TA::TRANSIENT, u8q));
  auto resize = G.graph->CreateOperation<tim::vx::ops::Resize>(
      tim::vx::ResizeType::BILINEAR, /*factor=*/0.0f,
      /*align_corners=*/false, /*half_pixel_centers=*/true,
      /*target_height=*/224, /*target_width=*/224);
  (*resize).BindInput(G.in_t).BindOutput(resized);

  // Stage 2 — DataConvert u8|asym → fp32. This dequantizes via the input
  // quant params. PASS in the op_probe pair table for shader path.
  G.fp32_t = G.graph->CreateTensor(tim::vx::TensorSpec(
      DT::FLOAT32, {224, 224, 3, 1}, TA::OUTPUT));
  auto dc = G.graph->CreateOperation<tim::vx::ops::DataConvert>();
  (*dc).BindInput(resized).BindOutput(G.fp32_t);

  if (!G.graph->Compile()) {
    std::fprintf(stderr, "preproc graph Compile failed\n");
    std::abort();
  }
  return G;
}

// ── ResNet50 graph (NN engine / NPU) ─────────────────────────────────
struct ResnetGraph {
  std::shared_ptr<tim::vx::Context> ctx;
  std::shared_ptr<tim::vx::Graph>   graph;
  std::shared_ptr<tim::vx::Tensor>  in_t, out_t;
};
ResnetGraph build_resnet() {
  ResnetGraph G;
  G.ctx   = tim::vx::Context::Create();
  G.graph = G.ctx->CreateGraph();
  // Same spec as kInputs[0] in resnet50_v1/runner_main.cpp. shape is
  // passed in MLIR text order — timvx_main's first op is a transpose
  // {3,2,1,0} that maps to the model's native NHWC-like layout.
  tim::vx::Quantization in_q(tim::vx::QuantType::ASYMMETRIC,
                             0.1546357512f, 114);
  G.in_t = G.graph->CreateTensor(tim::vx::TensorSpec(
      DT::UINT8, {1, 3, 224, 224}, TA::INPUT, in_q));
  G.out_t = timvx_main(G.graph, G.in_t);
  if (!G.graph->Compile()) {
    std::fprintf(stderr, "ResNet50 graph Compile failed\n");
    std::abort();
  }
  return G;
}

// Minimal 2-thread barrier: both arrivers block until the second reaches
// arrive_and_wait(), at which point both proceed simultaneously.
struct Barrier {
  std::mutex m;
  std::condition_variable cv;
  int waiting = 0;
  int target;
  bool released = false;
  explicit Barrier(int t) : target(t) {}
  void arrive_and_wait() {
    std::unique_lock<std::mutex> lk(m);
    if (++waiting >= target) { released = true; cv.notify_all(); }
    else cv.wait(lk, [&]{ return released; });
  }
};

struct RunResult {
  std::vector<double> per_iter_ms;
  double total_ms;
  bool ok;
};

RunResult run_n(std::shared_ptr<tim::vx::Graph> g, int n, Barrier* barrier,
                const char* tag) {
  RunResult r;
  r.per_iter_ms.reserve(n);
  r.ok = true;
  if (barrier) barrier->arrive_and_wait();
  auto t0 = Clock::now();
  for (int i = 0; i < n; ++i) {
    auto ti = Clock::now();
    if (!g->Run()) {
      std::fprintf(stderr, "[%s] Run() returned false at iter %d\n", tag, i);
      r.ok = false;
      break;
    }
    r.per_iter_ms.push_back(ms_since(ti));
  }
  r.total_ms = ms_since(t0);
  return r;
}

void warmup(std::shared_ptr<tim::vx::Graph> g, const char* tag) {
  for (int i = 0; i < kWarmup; ++i) {
    if (!g->Run()) {
      std::fprintf(stderr, "[%s] warmup Run() failed at iter %d\n", tag, i);
      std::abort();
    }
  }
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("timvx_ppu_npu_concurrency: %d warmup + %d timed iters / mode\n",
              kWarmup, kIters);

  std::printf("[stage] building preproc graph (shader/PPU path)\n");
  auto pp = build_preproc();
  std::printf("[stage] building ResNet50 graph (NN engine / NPU)\n");
  auto rn = build_resnet();

  // Pre-load inputs ONCE; we exclude CopyDataToTensor from timing. The
  // question being measured is purely about NPU/PPU dispatch contention,
  // not host-side I/O cost.
  std::vector<uint8_t> pp_in(480 * 360 * 3, 128);
  pp.in_t->CopyDataToTensor(pp_in.data(), pp_in.size());
  std::vector<uint8_t> rn_in(1 * 3 * 224 * 224, 128);
  rn.in_t->CopyDataToTensor(rn_in.data(), rn_in.size());

  // Warm both graphs (NPU clock ramp, lazy kernel JIT).
  std::printf("[stage] warmup\n");
  warmup(pp.graph, "PPU");
  warmup(rn.graph, "NPU");

  // Mode A — preproc alone.
  std::printf("[stage] Mode A — preproc alone\n");
  auto rA = run_n(pp.graph, kIters, nullptr, "A/PPU");

  // Mode B — ResNet50 alone.
  std::printf("[stage] Mode B — ResNet50 alone\n");
  auto rB = run_n(rn.graph, kIters, nullptr, "B/NPU");

  // Mode C — concurrent. Barrier-synchronized start so both hot loops
  // begin at the same instant. No mutex around Run().
  std::printf("[stage] Mode C — both concurrent (no Run() mutex)\n");
  Barrier barrier(2);
  RunResult rC_pp{}, rC_rn{};
  std::thread tp([&]{ rC_pp = run_n(pp.graph, kIters, &barrier, "C/PPU"); });
  std::thread tr([&]{ rC_rn = run_n(rn.graph, kIters, &barrier, "C/NPU"); });
  tp.join();
  tr.join();

  std::printf("\n== results ==\n");
  print_stats("A PPU",  summarize(rA.per_iter_ms),    rA.total_ms);
  print_stats("B NPU",  summarize(rB.per_iter_ms),    rB.total_ms);
  if (rC_pp.ok)
    print_stats("C PPU", summarize(rC_pp.per_iter_ms), rC_pp.total_ms);
  else
    std::printf("  [C PPU  ] FAILED at iter %zu (Run returned false)\n",
                rC_pp.per_iter_ms.size());
  if (rC_rn.ok)
    print_stats("C NPU", summarize(rC_rn.per_iter_ms), rC_rn.total_ms);
  else
    std::printf("  [C NPU  ] FAILED at iter %zu (Run returned false)\n",
                rC_rn.per_iter_ms.size());

  if (rC_pp.ok && rC_rn.ok) {
    double serial_baseline  = rA.total_ms + rB.total_ms;
    double concurrent_total = std::max(rC_pp.total_ms, rC_rn.total_ms);
    double speedup = serial_baseline / concurrent_total;
    const char* verdict =
        speedup >= 1.6  ? "PARALLEL          (PPU and NPU run independently)"
        : speedup >= 1.2 ? "PARTIAL OVERLAP   (some queue sharing)"
                         : "SERIALIZED        (one dispatch queue / device)";
    std::printf("\nserial baseline (A+B): %8.2f ms\n", serial_baseline);
    std::printf("concurrent (max C):    %8.2f ms\n", concurrent_total);
    std::printf("speedup:               %8.2fx  →  %s\n", speedup, verdict);
  } else {
    std::printf("\nMode C failed — concurrent Run() is not supported on this "
                "configuration (treat as SERIALIZED or worse).\n");
  }
  return 0;
}
