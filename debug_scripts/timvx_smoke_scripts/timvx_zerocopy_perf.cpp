// timvx_zerocopy_perf.cpp — measure the wall-clock benefit of the zero-copy
// path vs. the runner's current CopyDataToTensor / CopyDataFromTensor path,
// swept across six tensor sizes from 10^1 to 10^6 fp32 elements.
//
// Both variants drive Pow(x, 2.0) on a rank-1 fp32 tensor of N elements.
// Pow is elementwise so shape doesn't matter; rank-1 keeps the harness
// minimal. Sizes are powers of ten so the trend is easy to read.
//
//   variant "copy"  — Graph::CreateTensor(spec, nullptr) for INPUT/OUTPUT.
//                     Per inner cycle:
//                       in_t->CopyDataToTensor(src_buf, n)
//                       graph->Run()
//                       out_t->CopyDataFromTensor(dst_buf)
//                     This is exactly what runner_main.cpp.tpl does today.
//
//   variant "zero"  — Graph::CreateIOTensor(spec, host_buf). Per inner cycle:
//                       memcpy(in_buf, src_buf, n)        // app fills buf
//                       in_t->FlushCacheForHandle()
//                       graph->Run()
//                       out_t->CopyDataFromTensor(out_buf)  // self-memcpy +
//                                                           // cache invalidate
//                     Structurally identical to "copy" — still does one
//                     memcpy per cycle; only the direction changes.
//
//   variant "true"  — Same CreateIOTensor setup. in_buf pre-filled ONCE
//                     before the bench loop. Per inner cycle:
//                       in_t->FlushCacheForHandle()
//                       graph->Run()
//                       out_t->CopyDataFromTensor(out_buf)
//                     Models the case where the upstream producer (decoder,
//                     previous graph stage, mmap'd file) writes directly into
//                     our handle buffer — no per-cycle src→handle copy at all.
//                     This is the only variant that avoids memcpy on the hot
//                     path entirely. Output still needs CopyDataFromTensor as
//                     the cache-invalidate workaround (see timvx_zerocopy.cpp).
//
// Outer loop: kOuter iters; each iter runs kInner back-to-back cycles. Time
// the inner kInner. First kWarmup iters discarded; report stats over the rest.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"
#include "tim/vx/types.h"
#include "tim/vx/ops/elementwise.h"

namespace {

#define MEM_ALIGN(x, a) (((x) + ((a) - 1)) & ~((a) - 1))
constexpr size_t kAlign  = 64;
constexpr int    kOuter  = 1000;  // total outer iters (each runs all 3 variants)
constexpr int    kInner  = 10;    // NPU cycles timed per outer iter
constexpr int    kWarmup = 100;   // leading iters to discard (NPU clock ramp)

// Suites: powers of ten, 10^1 .. 10^6.
const uint32_t kSizes[] = {20000u, 40000u, 60000u, 80000u, 100000u, 120000u};

float* aligned_floats(size_t numel) {
  size_t nb = MEM_ALIGN(numel * sizeof(float), kAlign);
  void* p   = aligned_alloc(kAlign, nb);
  if (!p) { std::fprintf(stderr, "aligned_alloc failed\n"); std::abort(); }
  std::memset(p, 0, nb);
  return static_cast<float*>(p);
}

struct Stats { double mean, median, min, max, p99; };

Stats summarize(std::vector<double> us) {
  std::sort(us.begin(), us.end());
  Stats s{};
  s.min    = us.front();
  s.max    = us.back();
  s.median = us[us.size() / 2];
  s.p99    = us[std::min<size_t>(us.size() - 1, us.size() * 99 / 100)];
  s.mean   = std::accumulate(us.begin(), us.end(), 0.0) / us.size();
  return s;
}

struct Graph {
  std::shared_ptr<tim::vx::Context>   ctx;
  std::shared_ptr<tim::vx::Graph>     g;
  std::shared_ptr<tim::vx::Tensor>    in_t, out_t, exp_t;
  std::shared_ptr<tim::vx::Operation> op;
  float* in_buf{nullptr};   // only set for zero-copy variant
  float* out_buf{nullptr};  // only set for zero-copy variant
  float  exp_val{2.0f};
};

Graph build(uint32_t n, bool zero_copy) {
  Graph G;
  if (zero_copy) {
    G.in_buf  = aligned_floats(n);
    G.out_buf = aligned_floats(n);
  }
  G.ctx = tim::vx::Context::Create();
  G.g   = G.ctx->CreateGraph();
  tim::vx::TensorSpec is(tim::vx::DataType::FLOAT32, {n},
                         tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec os(tim::vx::DataType::FLOAT32, {n},
                         tim::vx::TensorAttribute::OUTPUT);
  tim::vx::TensorSpec es(tim::vx::DataType::FLOAT32, {1},
                         tim::vx::TensorAttribute::CONSTANT);
  G.in_t  = zero_copy ? G.g->CreateIOTensor(is, G.in_buf)
                      : G.g->CreateTensor(is);
  G.out_t = zero_copy ? G.g->CreateIOTensor(os, G.out_buf)
                      : G.g->CreateTensor(os);
  G.exp_t = G.g->CreateTensor(es, &G.exp_val);
  G.op    = G.g->CreateOperation<tim::vx::ops::Add>();
  (*G.op).BindInputs({G.in_t, G.exp_t}).BindOutput(G.out_t);
  if (!G.g->Compile()) {
    std::fprintf(stderr, "Compile failed (n=%u, zero=%d)\n", n, (int)zero_copy);
    std::abort();
  }
  return G;
}

struct TriStats { Stats copy, zero, pure; };

// All three variants are timed within the same outer iteration so they share
// NPU clock state and cache warmth — eliminating the sequential ordering bias
// where the first variant runs warm and later variants run in a different state.
template <typename C1, typename C2, typename C3>
TriStats bench3(C1 copy_fn, C2 zero_fn, C3 true_fn) {
  std::vector<double> us_copy, us_zero, us_true;
  us_copy.reserve(kOuter);
  us_zero.reserve(kOuter);
  us_true.reserve(kOuter);
  auto measure = [](std::vector<double>& dst, auto& fn) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kInner; ++i) fn();
    auto t1 = std::chrono::steady_clock::now();
    dst.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  };
  for (int o = 0; o < kOuter; ++o) {
    measure(us_copy, copy_fn);
    measure(us_zero, zero_fn);
    measure(us_true, true_fn);
  }
  auto keep = [](const std::vector<double>& v) {
    return std::vector<double>(v.begin() + kWarmup, v.end());
  };
  return TriStats{summarize(keep(us_copy)),
                  summarize(keep(us_zero)),
                  summarize(keep(us_true))};
}

void run_suite(uint32_t n) {
  size_t bytes = size_t(n) * sizeof(float);
  std::printf("\n--- N=%u (%.1f KiB per tensor) ---\n", n, bytes / 1024.0);

  std::vector<float> src(n);
  for (size_t i = 0; i < n; ++i)
    src[i] = static_cast<float>((i % 1024) - 512) / 64.0f;
  std::vector<float> dst_copy(n, 0.0f);

  Graph g_copy = build(n, /*zero_copy=*/false);
  Graph g_zero = build(n, /*zero_copy=*/true);
  Graph g_true = build(n, /*zero_copy=*/true);

  // Pre-fill g_true's handle buffer once — simulates a producer that writes
  // directly into our buffer (e.g. a decoder, upstream graph stage, or mmap'd
  // file). The bench loop then skips the src→handle memcpy entirely.
  std::memcpy(g_true.in_buf, src.data(), bytes);

  auto [s_copy, s_zero, s_true] = bench3(
    [&] {
      g_copy.in_t->CopyDataToTensor(src.data(), bytes);
      g_copy.g->Run();
      g_copy.out_t->CopyDataFromTensor(dst_copy.data());
    },
    [&] {
      std::memcpy(g_zero.in_buf, src.data(), bytes);
      g_zero.in_t->FlushCacheForHandle();
      g_zero.g->Run();
      g_zero.out_t->CopyDataFromTensor(g_zero.out_buf);
    },
    [&] {
      g_true.in_t->FlushCacheForHandle();
      g_true.g->Run();
      g_true.out_t->CopyDataFromTensor(g_true.out_buf);
    }
  );

  // Sanity check: all three outputs match src[i]^2.
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    float ref       = src[i] * src[i];
    float tol       = 1e-3f * std::fabs(ref) + 1e-4f;
    float diff_copy = std::fabs(dst_copy[i] - ref);
    float diff_zero = std::fabs(g_zero.out_buf[i] - ref);
    float diff_true = std::fabs(g_true.out_buf[i] - ref);
    if (diff_copy > tol || diff_zero > tol || diff_true > tol) ++bad;
  }

  auto print = [&](const char* tag, const Stats& s) {
    std::printf("  [%-4s] mean=%9.1f us  median=%9.1f  min=%9.1f  p99=%9.1f  "
                "max=%9.1f  per-cycle mean=%8.2f us\n",
                tag, s.mean, s.median, s.min, s.p99, s.max, s.mean / kInner);
  };
  auto delta = [&](const char* label, const Stats& a, const Stats& b) {
    double dpc = (a.mean - b.mean) / kInner;
    double pct = (a.mean - b.mean) * 100.0 / b.mean;
    std::printf("  %-4s vs copy: %+8.2f us/cycle  (%+5.1f%%, %s is %s)\n",
                label, dpc, pct, label, pct < 0 ? "FASTER" : "slower");
  };
  print("copy", s_copy);
  print("zero", s_zero);
  print("true", s_true);
  delta("zero", s_zero, s_copy);
  delta("true", s_true, s_copy);
  std::printf("  sanity=%s\n", bad == 0 ? "ok" : "BAD");

  std::free(g_zero.in_buf);
  std::free(g_zero.out_buf);
  std::free(g_true.in_buf);
  std::free(g_true.out_buf);
}

}  // namespace

int main() {
  std::printf("Add(x, 2.0)  fp32 rank-1 sweep — interleaved bench3\n"
              "%d outer * %d inner, drop first %d, report last %d\n",
              kOuter, kInner, kWarmup, kOuter - kWarmup);
  for (uint32_t n : kSizes) run_suite(n);
  std::printf("\n");
  return 0;
}
