// pipeline/orchestrator.h — `Orchestrator` interface + four strategies.
//
// An Orchestrator runs the pre→infer→post pipeline for each submitted
// input. The four strategies model the cells of the user's matrix:
//
//   Sequential  : caller-thread does pre, infer, post in order, then
//                 returns. Used as a baseline and for non-streaming
//                 (one-shot) modes.
//
//   Pipeline    : three dedicated worker threads (pre, infer, post)
//                 connected by bounded queues. Throughput in steady
//                 state is `1 / max(T_pre, T_infer, T_post)`.
//
//   Hybrid      : N worker threads, each running pre → infer → post.
//                 ALL workers share ONE tim::vx::Graph instance, so
//                 the infer stage is mutex-serialised — only one
//                 Run() at a time. Pre and post on different threads
//                 can overlap with each other's infer. (Renamed from
//                 the old "ThreadPool" — the original "Hybrid" with
//                 separate pre_workers/post_workers + a single infer
//                 thread was equivalent when pre_workers==post_workers,
//                 so the two collapsed into one strategy.)
//
//   Pool        : N worker threads, each owning its OWN tim::vx::Graph
//                 instance (one full copy of the model per worker).
//                 Multiple `graph_i->Run()` proceed in parallel — no
//                 internal mutex, the NPU driver handles dispatch
//                 ordering. Memory cost: N × weights. Throughput cost:
//                 the NPU has finite compute, so concurrent Runs may
//                 internally serialize; the win shows up when one
//                 Run stalls (cache invalidate, DMA setup) and another
//                 can advance.
//
// All four implement the same interface: `submit(raw)` returns a
// `std::future<InferenceResult>`, and `drain()` blocks until all
// in-flight submissions complete.
//
// Concurrency invariants
// ----------------------
//   * Within a single tim::vx::Graph, `Run()` is called from at most
//     one thread at a time. Sequential / Pipeline / Hybrid achieve
//     this structurally (one dedicated infer thread, or a mutex
//     around the shared graph). The Pool strategy sidesteps the
//     invariant by giving each worker its own graph instance.
//   * SwapHandle on a tensor is similarly per-graph; per-Job buffers
//     from IoBufferPool ensure no two workers ever SwapHandle the
//     same tensor concurrently.
//   * Cache-coherency steps (FlushCacheForHandle before Run,
//     CopyDataFromTensor-as-invalidate after Run) live in
//     `infer_one_zerocopy` — same regardless of strategy.

#ifndef TIMVX_PIPELINE_ORCHESTRATOR_H
#define TIMVX_PIPELINE_ORCHESTRATOR_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"

#include "pipeline/diag.h"
#include "pipeline/iobuffer.h"
#include "pipeline/postproc.h"
#include "pipeline/preproc.h"

namespace timvx_pipeline {

// One model graph instance + its input/output tensors. Sequential /
// Pipeline / Hybrid use a single WorkerGraph; ThreadPool uses N
// (one per worker thread) so that `graph->Run()` can fire in parallel
// with no internal locking. The runner constructs these from the
// generated `__FUNC_NAME__()` builder and hands a vector to the
// orchestrator factory; non-pool orchestrators just consume the
// first element.
struct WorkerGraph {
  std::shared_ptr<tim::vx::Context> ctx;
  std::shared_ptr<tim::vx::Graph>   graph;
  std::shared_ptr<tim::vx::Tensor>  in;
  std::shared_ptr<tim::vx::Tensor>  out;
};

class Orchestrator {
 public:
  virtual ~Orchestrator() = default;
  // Submit one input, get a future that resolves when post completes.
  virtual std::future<InferenceResult> submit(std::vector<uint8_t> raw) = 0;
  // Block until in-flight inputs finish. Safe to call multiple times.
  virtual void drain() = 0;
  virtual const char* name() const = 0;
};

// ── Bounded queue (used by Pipeline / Hybrid) ───────────────────────────
template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(size_t cap) : cap_(cap) {}
  // Returns false if the queue was shut down with `shutdown()`.
  bool push(T v) {
    std::unique_lock<std::mutex> lk(mu_);
    not_full_.wait(lk, [&]{ return q_.size() < cap_ || stop_; });
    if (stop_) return false;
    q_.push_back(std::move(v));
    not_empty_.notify_one();
    return true;
  }
  // Returns false on shutdown (queue drained).
  bool pop(T& out) {
    std::unique_lock<std::mutex> lk(mu_);
    not_empty_.wait(lk, [&]{ return !q_.empty() || stop_; });
    if (q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop_front();
    not_full_.notify_one();
    return true;
  }
  void shutdown() {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
    not_full_.notify_all();
    not_empty_.notify_all();
  }
 private:
  std::mutex mu_;
  std::condition_variable not_full_, not_empty_;
  std::deque<T> q_;
  size_t cap_;
  bool stop_ = false;
};

// Common job carried through every orchestrator's stages. Each stage
// fills part of the struct in turn; the postprocessor finalises and
// the orchestrator fulfils the promise.
//
// In the COPY path (`io_buf == nullptr`), `preproc_out` holds the
// tensor-ready bytes produced by the preprocessor; the infer step
// CopyDataToTensors them and the postprocessor reads from the
// model's output tensor via CopyDataFromTensor.
//
// In the ZERO-COPY path (`io_buf != nullptr`), the preprocessor
// writes its output directly into `io_buf->in_data`; the infer step
// SwapHandle-rebinds the model's input tensor to that buffer,
// FlushCacheForHandle()s, Run()s, and CopyDataFromTensors the model
// output INTO `io_buf->out_data` (which doubles as the cache-
// invalidate trigger — `InvalidateCacheForHandle()` is broken on
// VIP9000Nano-DI; see `timvx_zerocopy.cpp`). The postprocessor reads
// directly from `io_buf->out_data`. After post is done, the
// orchestrator releases the io_buf back to its pool.
struct Job {
  std::vector<uint8_t> raw;
  std::vector<uint8_t> preproc_out;        // unused if io_buf != nullptr
  IoBuffer* io_buf = nullptr;               // null in copy path
  std::shared_ptr<tim::vx::Tensor> output_tensor;
  InferenceResult result;
  std::promise<InferenceResult> promise;
};

// COPY-path infer: feed preproc_out → CopyDataToTensor → Run.
inline bool infer_one(const std::shared_ptr<tim::vx::Graph>& graph,
                      const std::shared_ptr<tim::vx::Tensor>& input,
                      const std::vector<uint8_t>& bytes,
                      InferenceResult& result) {
  using Clock = std::chrono::steady_clock;
  if (!input->CopyDataToTensor(
          bytes.data(), static_cast<uint32_t>(bytes.size()))) {
    result.error = "CopyDataToTensor failed";
    return false;
  }
  result.infer_start_tp = Clock::now();
  if (!graph->Run()) {
    result.error = "graph->Run failed";
    return false;
  }
  result.infer_end_tp = Clock::now();
  result.infer_ms = std::chrono::duration<double, std::milli>(
      result.infer_end_tp - result.infer_start_tp).count();
  return true;
}

// ZERO-COPY-path infer.
//
// In cpu_zero_copy modes, `job.io_buf->in_data` was written by the CPU
// preprocessor. In ppu_zero_copy modes, it was written by the PPU
// preprocessor's graph via DMA, and the CPU never touched it. Both
// cases share the same SwapHandle + Run + post-Run-cache-invalidate
// dance; only the "is the CPU cache for io_buf->in_data dirty?"
// question differs — that's encapsulated in `pre->sync_for_infer()`,
// which the CPU base impl resolves to `FlushCacheForHandle(input)` and
// PPU overrides to a no-op.
//
// SwapHandle is not thread-safe; the orchestrator guarantees only one
// thread reaches this function at a time (either the dedicated infer
// thread in pipeline/hybrid, or the infer-mutex holder in pool).
inline bool infer_one_zerocopy(const std::shared_ptr<tim::vx::Graph>& graph,
                                const std::shared_ptr<tim::vx::Tensor>& input,
                                const std::shared_ptr<tim::vx::Tensor>& output,
                                IoBuffer* io_buf,
                                PreProcessor* pre,
                                InferenceResult& result) {
  using Clock = std::chrono::steady_clock;
  if (!input->SwapHandle(io_buf->in_data,
                          /*is_new_ptr_malloc_by_ovxlib=*/false,
                          /*old_ptr_release=*/nullptr)) {
    result.error = "SwapHandle(input) failed";
    return false;
  }
  if (!output->SwapHandle(io_buf->out_data,
                           /*is_new_ptr_malloc_by_ovxlib=*/false,
                           /*old_ptr_release=*/nullptr)) {
    result.error = "SwapHandle(output) failed";
    return false;
  }
  // Bring io_buf->in_data into NPU-visible state. CPU pre flushes its
  // dirty cache lines; PPU pre no-ops (the buffer was DMA-written and
  // a CPU flush would clobber it with stale cache lines).
  if (pre) pre->sync_for_infer(io_buf, input);
  result.infer_start_tp = Clock::now();
  if (!graph->Run()) {
    result.error = "graph->Run failed";
    return false;
  }
  result.infer_end_tp = Clock::now();
  result.infer_ms = std::chrono::duration<double, std::milli>(
      result.infer_end_tp - result.infer_start_tp).count();
  // The documented incantation to get the CPU's view of the output
  // buffer consistent with the NPU's writes is
  // CopyDataFromTensor(handle_ptr) — InvalidateCacheForHandle() is
  // bugged on this chip and silently skips the real invalidate. Pass
  // the SAME pointer the tensor's bound to; the call's a self-memcpy
  // that exists only for its cache side-effect.
  if (!output->CopyDataFromTensor(io_buf->out_data)) {
    result.error = "CopyDataFromTensor(zerocopy invalidate) failed";
    return false;
  }
  return true;
}

// ── 1. Sequential ───────────────────────────────────────────────────────
// Set `pool != nullptr` to take the zero-copy path: preprocessor writes
// directly into a pool-acquired IoBuffer's in_data; infer SwapHandles
// the model tensors to that slot; postprocessor reads from out_data.
// Otherwise (the original path) preprocessor's `process()` returns a
// vector that gets CopyDataToTensor'd in.
class SequentialOrchestrator : public Orchestrator {
 public:
  SequentialOrchestrator(std::shared_ptr<tim::vx::Graph> g,
                          std::shared_ptr<tim::vx::Tensor> in,
                          std::shared_ptr<tim::vx::Tensor> out,
                          PreProcessor* pre, PostProcessor* post,
                          IoBufferPool* pool = nullptr)
      : g_(std::move(g)), in_(std::move(in)), out_(std::move(out)),
        pre_(pre), post_(post), pool_(pool) {}
  std::future<InferenceResult> submit(std::vector<uint8_t> raw) override {
    using Clock = std::chrono::steady_clock;
    std::promise<InferenceResult> p;
    auto fut = p.get_future();
    InferenceResult r;
    if (pool_) {
      IoBuffer* buf = pool_->acquire();
      r.pre_start_tp = Clock::now();
      if (!pre_->process_inplace(raw, buf)) {
        r.error = "preprocessor::process_inplace failed";
        pool_->release(buf);
        r.complete_tp = Clock::now();
        p.set_value(std::move(r));
        return fut;
      }
      r.pre_end_tp = Clock::now();
      if (!infer_one_zerocopy(g_, in_, out_, buf, pre_, r)) {
        pool_->release(buf);
        r.complete_tp = Clock::now();
        p.set_value(std::move(r));
        return fut;
      }
      r.post_start_tp = Clock::now();
      post_->process_inplace(buf->out_data, buf->out_size, out_, r);
      r.post_end_tp = Clock::now();
      pool_->release(buf);
    } else {
      r.pre_start_tp = Clock::now();
      auto bytes = pre_->process(raw);
      r.pre_end_tp = Clock::now();
      if (!infer_one(g_, in_, bytes, r)) {
        r.complete_tp = Clock::now();
        p.set_value(std::move(r));
        return fut;
      }
      r.post_start_tp = Clock::now();
      post_->process(out_, r);
      r.post_end_tp = Clock::now();
    }
    r.complete_tp = Clock::now();
    p.set_value(std::move(r));
    return fut;
  }
  void drain() override {}  // nothing in flight — submit is synchronous
  const char* name() const override { return "sequential"; }
 private:
  std::shared_ptr<tim::vx::Graph> g_;
  std::shared_ptr<tim::vx::Tensor> in_, out_;
  PreProcessor* pre_;
  PostProcessor* post_;
  IoBufferPool* pool_;  // nullptr = copy path; non-null = zero-copy path
};

// ── 2. Pipeline (3-stage: 1 pre, 1 infer, 1 post) ───────────────────────
//
// Zero-copy variant: when `pool != nullptr` is passed, each Job is
// associated with a pool-acquired IoBuffer on enter to `pre_loop()`.
// The buffer travels with the Job through the queues; the infer
// thread SwapHandle-rebinds the model tensors to it before Run; the
// post thread reads from the same buffer and releases it back to the
// pool. With multiple in-flight Jobs, multiple buffer slots are in
// use simultaneously — the pool size MUST be ≥ queue_cap to avoid
// starvation in the pipeline. The factory sizes the pool accordingly.
class PipelineOrchestrator : public Orchestrator {
 public:
  PipelineOrchestrator(std::shared_ptr<tim::vx::Graph> g,
                        std::shared_ptr<tim::vx::Tensor> in,
                        std::shared_ptr<tim::vx::Tensor> out,
                        PreProcessor* pre, PostProcessor* post,
                        size_t queue_cap = 4,
                        IoBufferPool* pool = nullptr)
      : g_(std::move(g)), in_(std::move(in)), out_(std::move(out)),
        pre_(pre), post_(post), pool_(pool),
        q_pre_(queue_cap), q_infer_(queue_cap) {
    th_pre_   = std::thread([this]{ pre_loop(); });
    th_infer_ = std::thread([this]{ infer_loop(); });
    th_post_  = std::thread([this]{ post_loop(); });
  }
  ~PipelineOrchestrator() override { drain(); }

  std::future<InferenceResult> submit(std::vector<uint8_t> raw) override {
    auto job = std::make_unique<Job>();
    job->raw = std::move(raw);
    auto fut = job->promise.get_future();
    q_pre_.push(std::move(job));
    return fut;
  }
  void drain() override {
    if (drained_) return;
    drained_ = true;
    q_pre_.shutdown();
    if (th_pre_.joinable())   th_pre_.join();
    q_infer_.shutdown();
    if (th_infer_.joinable()) th_infer_.join();
    if (th_post_.joinable())  th_post_.join();
  }
  const char* name() const override { return "pipeline"; }

 private:
  void pre_loop() {
    using Clock = std::chrono::steady_clock;
    std::unique_ptr<Job> j;
    while (q_pre_.pop(j)) {
      if (pool_) {
        j->io_buf = pool_->acquire();
        j->result.pre_start_tp = Clock::now();
        if (!pre_->process_inplace(j->raw, j->io_buf)) {
          j->result.error = "preprocessor::process_inplace failed";
          pool_->release(j->io_buf);
          j->io_buf = nullptr;
          j->result.complete_tp = Clock::now();
      j->promise.set_value(std::move(j->result));
          continue;
        }
        j->result.pre_end_tp = Clock::now();
      } else {
        j->result.pre_start_tp = Clock::now();
        j->preproc_out = pre_->process(j->raw);
        j->result.pre_end_tp = Clock::now();
      }
      q_infer_.push(std::move(j));
    }
  }
  void infer_loop() {
    using Clock = std::chrono::steady_clock;
    std::unique_ptr<Job> j;
    while (q_infer_.pop(j)) {
      j->output_tensor = out_;
      bool ok = pool_
          ? infer_one_zerocopy(g_, in_, out_, j->io_buf, pre_, j->result)
          : infer_one(g_, in_, j->preproc_out, j->result);
      if (!ok) {
        if (j->io_buf) { pool_->release(j->io_buf); j->io_buf = nullptr; }
        j->result.complete_tp = Clock::now();
      j->promise.set_value(std::move(j->result));
        continue;
      }
      q_post_.push(std::move(j));
    }
    q_post_.shutdown();
  }
  void post_loop() {
    using Clock = std::chrono::steady_clock;
    std::unique_ptr<Job> j;
    while (q_post_.pop(j)) {
      j->result.post_start_tp = Clock::now();
      if (pool_) {
        post_->process_inplace(j->io_buf->out_data, j->io_buf->out_size,
                                j->output_tensor, j->result);
        pool_->release(j->io_buf);
        j->io_buf = nullptr;
      } else {
        post_->process(j->output_tensor, j->result);
      }
      j->result.post_end_tp = Clock::now();
      j->result.complete_tp = Clock::now();
      j->promise.set_value(std::move(j->result));
    }
  }

  std::shared_ptr<tim::vx::Graph> g_;
  std::shared_ptr<tim::vx::Tensor> in_, out_;
  PreProcessor* pre_;
  PostProcessor* post_;
  IoBufferPool* pool_;
  BoundedQueue<std::unique_ptr<Job>> q_pre_;
  BoundedQueue<std::unique_ptr<Job>> q_infer_;
  BoundedQueue<std::unique_ptr<Job>> q_post_{8};
  std::thread th_pre_, th_infer_, th_post_;
  bool drained_ = false;
};

// ── 3. Hybrid (N workers, ONE shared graph, infer mutexed) ──────────────
//
// N worker threads, each running the full pre → infer → post chain. All
// workers share a single tim::vx::Graph instance, so the infer step is
// mutex-serialized (Run() is not reentrant on a single graph). Pre and
// post work on different threads can overlap with another worker's
// infer; the win is roughly `max(T_infer, max(T_pre,T_post)/N)`.
//
// Zero-copy variant: each Job acquires a pool buffer at start of pre,
// holds it across the infer critical section (SwapHandle + Run +
// CopyDataFromTensor under `infer_mu_`), then runs post (outside the
// mutex) reading from `io_buf->out_data`, and releases the buffer.
// Multiple workers can hold DIFFERENT pool slots simultaneously — they
// only contend on `infer_mu_`, not on the pool's slot mapping. Pool
// size must be ≥ num_workers + 1 to avoid pre/post stalls.
//
// Renamed from "ThreadPool" — the original "Hybrid" had separate
// pre_workers / 1 infer / post_workers threads, which was equivalent
// to this layout once you set pre_workers == post_workers, so the two
// collapsed.
class HybridPoolOrchestrator : public Orchestrator {
 public:
  HybridPoolOrchestrator(std::shared_ptr<tim::vx::Graph> g,
                          std::shared_ptr<tim::vx::Tensor> in,
                          std::shared_ptr<tim::vx::Tensor> out,
                          PreProcessor* pre, PostProcessor* post,
                          size_t num_workers = 4,
                          size_t queue_cap = 16,
                          IoBufferPool* pool = nullptr)
      : g_(std::move(g)), in_(std::move(in)), out_(std::move(out)),
        pre_(pre), post_(post), pool_(pool), q_(queue_cap) {
    workers_.reserve(num_workers);
    for (size_t i = 0; i < num_workers; ++i)
      workers_.emplace_back([this]{ worker_loop(); });
  }
  ~HybridPoolOrchestrator() override { drain(); }

  std::future<InferenceResult> submit(std::vector<uint8_t> raw) override {
    auto job = std::make_unique<Job>();
    job->raw = std::move(raw);
    auto fut = job->promise.get_future();
    q_.push(std::move(job));
    return fut;
  }
  void drain() override {
    if (drained_) return;
    drained_ = true;
    q_.shutdown();
    for (auto& t : workers_) if (t.joinable()) t.join();
  }
  const char* name() const override { return "hybrid"; }

 private:
  void worker_loop() {
    using Clock = std::chrono::steady_clock;
    std::unique_ptr<Job> j;
    while (q_.pop(j)) {
      if (pool_) {
        j->io_buf = pool_->acquire();
        j->result.pre_start_tp = Clock::now();
        if (!pre_->process_inplace(j->raw, j->io_buf)) {
          j->result.error = "preprocessor::process_inplace failed";
          pool_->release(j->io_buf);
          j->io_buf = nullptr;
          j->result.complete_tp = Clock::now();
      j->promise.set_value(std::move(j->result));
          continue;
        }
        j->result.pre_end_tp = Clock::now();
      } else {
        j->result.pre_start_tp = Clock::now();
        j->preproc_out = pre_->process(j->raw);
        j->result.pre_end_tp = Clock::now();
      }
      // Serialize the infer stage: one tim::vx::Graph cannot accept
      // concurrent Run() calls. The SwapHandle (zerocopy path) or
      // CopyDataToTensor (copy path) AND the Run() MUST be a single
      // critical section — otherwise a different worker can rebind
      // the tensors before our Run() reads them.
      bool ok;
      {
        std::lock_guard<std::mutex> lk(infer_mu_);
        ok = pool_
            ? infer_one_zerocopy(g_, in_, out_, j->io_buf, pre_, j->result)
            : infer_one(g_, in_, j->preproc_out, j->result);
      }
      j->output_tensor = out_;
      if (!ok) {
        if (j->io_buf) { pool_->release(j->io_buf); j->io_buf = nullptr; }
        j->result.complete_tp = Clock::now();
      j->promise.set_value(std::move(j->result));
        continue;
      }
      if (pool_) {
        // Post-process outside the mutex: the output bytes are in our
        // Job's slot's out_data (a distinct memory region from any
        // other worker's), so reads are safe to overlap with another
        // worker's pre/infer phase.
        j->result.post_start_tp = Clock::now();
        post_->process_inplace(j->io_buf->out_data, j->io_buf->out_size,
                                out_, j->result);
        j->result.post_end_tp = Clock::now();
        pool_->release(j->io_buf);
        j->io_buf = nullptr;
      } else {
        // Copy path: the output tensor is GRAPH-LEVEL state; reading
        // it post-Run-but-outside-the-mutex risks a different worker's
        // next Run() trampling the bytes. Keep this read INSIDE the
        // mutex (the original semantics — slower, but correct).
        std::lock_guard<std::mutex> lk(infer_mu_);
        j->result.post_start_tp = Clock::now();
        post_->process(out_, j->result);
        j->result.post_end_tp = Clock::now();
      }
      j->result.complete_tp = Clock::now();
      j->promise.set_value(std::move(j->result));
    }
  }

  std::shared_ptr<tim::vx::Graph> g_;
  std::shared_ptr<tim::vx::Tensor> in_, out_;
  PreProcessor* pre_;
  PostProcessor* post_;
  IoBufferPool* pool_;
  BoundedQueue<std::unique_ptr<Job>> q_;
  std::mutex infer_mu_;
  std::vector<std::thread> workers_;
  bool drained_ = false;
};

// ── 4. ThreadPool (N workers, N independent graphs, no infer mutex) ─────
//
// Each worker owns its OWN tim::vx::Graph instance (built from the same
// model body) with its own input/output tensors. There is NO infer
// mutex: `graph_i->Run()` calls from different workers fire concurrently
// and the NPU driver schedules them. Pre and post stages also run with
// no cross-worker coordination on graph state — each worker reads/
// writes only its own tensors.
//
// Trade-offs vs Hybrid (single shared graph):
//   + concurrent Runs: the NPU driver can pipeline DMA setup of Run i+1
//     against Run i still executing, hide cache-invalidate stalls, etc.
//   + post is always lock-free on the copy path too (each worker's
//     out_t is private, so reading it doesn't conflict with another
//     worker's next Run).
//   - memory cost: N × model weights. For resnet50 (~25 MiB weights)
//     and N=4, that's ~100 MiB. Linear in N.
//   - compile time: each graph independently compiles (~4s on rn50);
//     happens once at startup.
//
// The PreProcessor and PostProcessor are still shared across workers
// (with their own internal mutexes where they hold TIM-VX state, e.g.
// the PPU pre / PPU SoftMax post graphs). CPU pre/post are stateless
// and call freely from any thread.
//
// Zero-copy: same pattern as Hybrid — each Job acquires an IoBuffer
// from the shared pool, the worker SwapHandles ITS OWN graph's input
// and output tensors to that slot, runs, and post-processes out of
// io_buf->out_data before releasing. No two workers ever touch the
// same tensor (each has its own pair), and no two workers ever touch
// the same buffer slot (the pool guarantees exclusivity). Pool size
// must be ≥ num_workers + queue_cap so workers don't starve.
class ThreadPoolOrchestrator : public Orchestrator {
 public:
  ThreadPoolOrchestrator(std::vector<WorkerGraph> worker_graphs,
                          PreProcessor* pre, PostProcessor* post,
                          size_t queue_cap = 16,
                          IoBufferPool* pool = nullptr)
      : graphs_(std::move(worker_graphs)),
        pre_(pre), post_(post), pool_(pool), q_(queue_cap) {
    workers_.reserve(graphs_.size());
    for (size_t i = 0; i < graphs_.size(); ++i)
      workers_.emplace_back([this, i]{ worker_loop(i); });
  }
  ~ThreadPoolOrchestrator() override { drain(); }

  std::future<InferenceResult> submit(std::vector<uint8_t> raw) override {
    auto job = std::make_unique<Job>();
    job->raw = std::move(raw);
    auto fut = job->promise.get_future();
    q_.push(std::move(job));
    return fut;
  }
  void drain() override {
    if (drained_) return;
    drained_ = true;
    q_.shutdown();
    for (auto& t : workers_) if (t.joinable()) t.join();
  }
  const char* name() const override { return "threadpool"; }

 private:
  void worker_loop(size_t worker_id) {
    using Clock = std::chrono::steady_clock;
    WorkerGraph& wg = graphs_[worker_id];
    std::unique_ptr<Job> j;
    while (q_.pop(j)) {
      if (pool_) {
        j->io_buf = pool_->acquire();
        j->result.pre_start_tp = Clock::now();
        if (!pre_->process_inplace(j->raw, j->io_buf)) {
          j->result.error = "preprocessor::process_inplace failed";
          pool_->release(j->io_buf);
          j->io_buf = nullptr;
          j->result.complete_tp = Clock::now();
      j->promise.set_value(std::move(j->result));
          continue;
        }
        j->result.pre_end_tp = Clock::now();
      } else {
        j->result.pre_start_tp = Clock::now();
        j->preproc_out = pre_->process(j->raw);
        j->result.pre_end_tp = Clock::now();
      }
      // No infer mutex — wg.graph is private to this worker. Fire and
      // let the NPU driver schedule against other workers' Run()s.
      bool ok = pool_
          ? infer_one_zerocopy(wg.graph, wg.in, wg.out, j->io_buf, pre_,
                                j->result)
          : infer_one(wg.graph, wg.in, j->preproc_out, j->result);
      j->output_tensor = wg.out;
      if (!ok) {
        if (j->io_buf) { pool_->release(j->io_buf); j->io_buf = nullptr; }
        j->result.complete_tp = Clock::now();
      j->promise.set_value(std::move(j->result));
        continue;
      }
      j->result.post_start_tp = Clock::now();
      if (pool_) {
        post_->process_inplace(j->io_buf->out_data, j->io_buf->out_size,
                                wg.out, j->result);
        pool_->release(j->io_buf);
        j->io_buf = nullptr;
      } else {
        // Copy path: wg.out is OWNED by this worker — no other worker
        // can clobber it between Run and this read, so no mutex needed.
        post_->process(wg.out, j->result);
      }
      j->result.post_end_tp = Clock::now();
      j->result.complete_tp = Clock::now();
      j->promise.set_value(std::move(j->result));
    }
  }

  std::vector<WorkerGraph> graphs_;
  PreProcessor* pre_;
  PostProcessor* post_;
  IoBufferPool* pool_;
  BoundedQueue<std::unique_ptr<Job>> q_;
  std::vector<std::thread> workers_;
  bool drained_ = false;
};

} // namespace timvx_pipeline

#endif // TIMVX_PIPELINE_ORCHESTRATOR_H
