// Declaration: This is a benchmark-only file that is AI-generated
// and reviewed by human.
//
// pipeline/benchmark.h — submit N inputs to an Orchestrator and print
// throughput + per-stage stats. Used by the `--bench N` runner mode and
// the implicit one-shot path when `--mode` is selected.
//
// The harness submits all N inputs without waiting on intermediate
// futures, then drains. This lets the orchestrator fill its pipeline
// to depth and the steady-state max-stage bottleneck become visible.
//
// Stats reported
// --------------
//   wall      : total elapsed time, submit-of-first → drain-complete
//   per-img   : wall / N   (the streaming throughput inverse)
//   infer     : per-iter NPU Run() time, mean / median / p99 / max +
//               SE / 95% MoE. Measured inside infer_one(_zerocopy)
//               around `graph->Run()` (the NPU's actual work only —
//               excludes queueing, pre, and post).
//   e2e       : per-iter end-to-end latency: clock taken just before
//               `orch.submit(raw)` for iter i, to the orchestrator's
//               post-completion timestamp. Captures pre + infer + post
//               + any BoundedQueue / infer_mu_ wait time.
//   pre       : pre_end_tp - pre_start_tp — pure preprocessing time.
//   pre_wait  : infer_start_tp - pre_end_tp — gap between pre done
//               and graph->Run() starting (CopyDataToTensor / SwapHandle /
//               infer_mu_ acquisition / q_infer_ wait).
//   post_wait : post_start_tp - infer_end_tp — gap between Run done
//               and post starting (CopyDataFromTensor for cache
//               invalidate / q_post_ wait / mutex re-acquire).
//   post      : post_end_tp - post_start_tp — pure post processing.
//
// For every per-iter quantity we report mean ± 95% MoE (margin of
// error, computed as 1.96 × SE where SE = stddev / √n), and the SE
// itself for downstream callers that want to combine runs. Throughput
// and per-image-average are derived from inter-completion intervals
// (steady-state samples), so they have proper SE/MoE too.

#ifndef TIMVX_PIPELINE_BENCHMARK_H
#define TIMVX_PIPELINE_BENCHMARK_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <future>
#include <numeric>
#include <vector>

#include "pipeline/orchestrator.h"
#include "pipeline/postproc.h"

namespace timvx_pipeline {

struct BenchStats {
  size_t n_ok = 0;
  size_t n_fail = 0;
  double wall_ms = 0.0;
  // Per-iter latencies (each vector has n_ok entries when n_fail == 0).
  std::vector<double> infer_ms_per_iter;
  std::vector<double> e2e_ms_per_iter;
  std::vector<double> pre_ms_per_iter;
  std::vector<double> pre_wait_ms_per_iter;
  std::vector<double> post_wait_ms_per_iter;
  std::vector<double> post_ms_per_iter;
  // Inter-completion intervals: complete_tp[i+1] - complete_tp[i],
  // sorted by completion order (not submission order). N-1 samples.
  // Mean of these IS the steady-state per-image throughput-inverse;
  // its SE gives us a proper SE for throughput / per-image average.
  std::vector<double> interval_ms_per_iter;
  // From the LAST timed iteration — used for sanity-checking that the
  // mode produced the right argmax (e.g. cat105.jpg → 284 on resnet50).
  std::vector<TopKEntry> last_topk;
};

// `n_warmup` iters run first and are NOT included in the returned stats —
// useful for excluding cold-cache, NPU clock-ramp, and first-iter graph-
// kernel-JIT effects from the steady-state numbers.
inline BenchStats run_benchmark(Orchestrator& orch,
                                 std::vector<uint8_t> input_blob,
                                 size_t n_iters,
                                 size_t n_warmup = 0) {
  using Clock = std::chrono::steady_clock;
  // Warmup phase: submit + drain its futures, no timing collected.
  if (n_warmup > 0) {
    std::vector<std::future<InferenceResult>> wfutures;
    wfutures.reserve(n_warmup);
    for (size_t i = 0; i < n_warmup; ++i)
      wfutures.push_back(orch.submit(input_blob));
    for (auto& f : wfutures) f.get();
  }
  BenchStats stats;
  std::vector<std::future<InferenceResult>> futures;
  std::vector<Clock::time_point> submit_times;
  futures.reserve(n_iters);
  submit_times.reserve(n_iters);
  std::vector<Clock::time_point> complete_tps;
  complete_tps.reserve(n_iters);

  auto t0 = Clock::now();
  for (size_t i = 0; i < n_iters; ++i) {
    submit_times.push_back(Clock::now());
    futures.push_back(orch.submit(input_blob));
  }
  for (size_t i = 0; i < futures.size(); ++i) {
    auto r = futures[i].get();
    if (!r.error.empty()) { ++stats.n_fail; continue; }
    ++stats.n_ok;
    stats.infer_ms_per_iter.push_back(r.infer_ms);
    stats.e2e_ms_per_iter.push_back(
        std::chrono::duration<double, std::milli>(
            r.complete_tp - submit_times[i]).count());
    stats.pre_ms_per_iter.push_back(
        std::chrono::duration<double, std::milli>(
            r.pre_end_tp - r.pre_start_tp).count());
    stats.pre_wait_ms_per_iter.push_back(
        std::chrono::duration<double, std::milli>(
            r.infer_start_tp - r.pre_end_tp).count());
    stats.post_wait_ms_per_iter.push_back(
        std::chrono::duration<double, std::milli>(
            r.post_start_tp - r.infer_end_tp).count());
    stats.post_ms_per_iter.push_back(
        std::chrono::duration<double, std::milli>(
            r.post_end_tp - r.post_start_tp).count());
    complete_tps.push_back(r.complete_tp);
    stats.last_topk = std::move(r.topk);
  }
  stats.wall_ms = std::chrono::duration<double, std::milli>(
      Clock::now() - t0).count();
  orch.drain();

  // Inter-completion intervals → samples of steady-state per-image
  // time. Sort by actual completion time (workers may complete
  // out of submission order).
  std::sort(complete_tps.begin(), complete_tps.end());
  for (size_t i = 1; i < complete_tps.size(); ++i) {
    stats.interval_ms_per_iter.push_back(
        std::chrono::duration<double, std::milli>(
            complete_tps[i] - complete_tps[i - 1]).count());
  }
  return stats;
}

// One column of stats: per-iter mean / median / p99 / max plus
// standard error of the mean and 95% margin of error.
struct LatencyStats {
  double mean = 0.0;
  double median = 0.0;
  double p99 = 0.0;
  double max = 0.0;
  double se = 0.0;     // stddev / sqrt(n) — standard error of the mean
  double moe95 = 0.0;  // 1.96 * se — 95% CI half-width (large-n / normal)
};
inline LatencyStats _summarize(const std::vector<double>& xs) {
  LatencyStats s;
  if (xs.empty()) return s;
  s.mean = std::accumulate(xs.begin(), xs.end(), 0.0) / xs.size();
  std::vector<double> v = xs;
  std::sort(v.begin(), v.end());
  s.median = v[v.size() / 2];
  s.p99 = v[std::min(v.size() - 1, v.size() * 99 / 100)];
  s.max = v.back();
  if (xs.size() > 1) {
    double var = 0.0;
    for (double x : xs) var += (x - s.mean) * (x - s.mean);
    var /= static_cast<double>(xs.size() - 1);
    double stddev = std::sqrt(var);
    s.se = stddev / std::sqrt(static_cast<double>(xs.size()));
    s.moe95 = 1.96 * s.se;
  }
  return s;
}

inline void print_bench(const char* mode, const BenchStats& s) {
  std::printf("[bench %s] n_ok=%zu n_fail=%zu  wall=%.2f ms  per-img=%.3f ms"
              "  throughput=%.1f img/s\n",
              mode, s.n_ok, s.n_fail, s.wall_ms,
              s.n_ok ? s.wall_ms / s.n_ok : 0.0,
              s.n_ok ? s.n_ok * 1000.0 / s.wall_ms : 0.0);
  auto print_block = [&](const char* label, const LatencyStats& ls) {
    std::printf("              %-9s per iter: mean=%.3f±%.3f (SE=%.3f) "
                "median=%.3f p99=%.3f max=%.3f ms\n",
                label, ls.mean, ls.moe95, ls.se, ls.median, ls.p99, ls.max);
  };
  auto infer = _summarize(s.infer_ms_per_iter);
  auto e2e   = _summarize(s.e2e_ms_per_iter);
  auto pre   = _summarize(s.pre_ms_per_iter);
  auto pwait = _summarize(s.pre_wait_ms_per_iter);
  auto qwait = _summarize(s.post_wait_ms_per_iter);
  auto post  = _summarize(s.post_ms_per_iter);
  auto intv  = _summarize(s.interval_ms_per_iter);
  if (!s.infer_ms_per_iter.empty())     print_block("infer",     infer);
  if (!s.e2e_ms_per_iter.empty())       print_block("e2e",       e2e);
  if (!s.pre_ms_per_iter.empty())       print_block("pre",       pre);
  if (!s.pre_wait_ms_per_iter.empty())  print_block("pre_wait",  pwait);
  if (!s.post_wait_ms_per_iter.empty()) print_block("post_wait", qwait);
  if (!s.post_ms_per_iter.empty())      print_block("post",      post);

  // Machine-parseable single-line summary for `run_pipeline_bench.py`'s
  // LaTeX table emitter. Throughput / per-image are derived from the
  // inter-completion intervals (steady-state samples); throughput's
  // SE/MoE come from the delta method (∂(1000/x)/∂x = -1000/x²).
  if (!s.interval_ms_per_iter.empty() && intv.mean > 0.0) {
    double per_img_mean = intv.mean;
    double per_img_se   = intv.se;
    double per_img_moe  = intv.moe95;
    double thru_mean    = 1000.0 / per_img_mean;
    double thru_se      = 1000.0 * per_img_se  / (per_img_mean * per_img_mean);
    double thru_moe     = 1000.0 * per_img_moe / (per_img_mean * per_img_mean);
    std::printf(
        "[stats %s]"
        " thru_mean=%.4f thru_se=%.4f thru_moe=%.4f"
        " perimg_mean=%.4f perimg_se=%.4f perimg_moe=%.4f"
        " infer_mean=%.4f infer_se=%.4f infer_moe=%.4f infer_p99=%.4f"
        " e2e_mean=%.4f e2e_se=%.4f e2e_moe=%.4f e2e_p99=%.4f"
        " pre_mean=%.4f pre_se=%.4f pre_moe=%.4f"
        " prewait_mean=%.4f prewait_se=%.4f prewait_moe=%.4f"
        " postwait_mean=%.4f postwait_se=%.4f postwait_moe=%.4f"
        " post_mean=%.4f post_se=%.4f post_moe=%.4f"
        "\n",
        mode,
        thru_mean, thru_se, thru_moe,
        per_img_mean, per_img_se, per_img_moe,
        infer.mean, infer.se, infer.moe95, infer.p99,
        e2e.mean, e2e.se, e2e.moe95, e2e.p99,
        pre.mean, pre.se, pre.moe95,
        pwait.mean, pwait.se, pwait.moe95,
        qwait.mean, qwait.se, qwait.moe95,
        post.mean, post.se, post.moe95);
  }

  // Print top-K from the last timed iteration. Machine-parseable for the
  // sweep harness; the leading `[topk]` is the column key.
  if (!s.last_topk.empty()) {
    std::printf("[topk %s]", mode);
    for (const auto& e : s.last_topk)
      std::printf(" %u:%.4f", e.first, e.second);
    std::printf("\n");
  }
  std::fflush(stdout);
}

} // namespace timvx_pipeline

#endif // TIMVX_PIPELINE_BENCHMARK_H
