// pipeline/serve.h — `--serve <port>` plumbing for the runner.
//
// The legacy server (kept here intact, post-extraction) listens on TCP,
// accepts one inference request at a time, and uses a single inference
// thread fed by a mutex+cv FIFO queue. Networking lives on a separate
// accept thread plus one reader thread per accepted client; both push
// validated requests onto the queue.
//
// Wire format (little-endian, host order on aarch64 LE):
//   request : u64 req_id | u32 payload_len | <payload_len bytes>
//   response: u64 req_id | u8 status (1=OK, 0=FAIL)
//             if OK:   f32 inference_ms
//                      u32 K
//                      K × { u32 class_id, f32 score }   (top-K)
//             if FAIL: u32 err_len | <err_len bytes UTF-8>
//
// To keep this header self-contained, the model-specific bits
// (single-input 1×3×224×224 validation, payload-size derivation) take
// the kInputs vector by reference and validate at startup. The runner's
// main() decides whether `--serve` is appropriate for the deployed
// model.

#ifndef TIMVX_PIPELINE_SERVE_H
#define TIMVX_PIPELINE_SERVE_H

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <errno.h>
#include <memory>
#include <mutex>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"

#include "pipeline/byte_layout.h"
#include "pipeline/input_spec.h"
#include "pipeline/tensor_io.h"

namespace timvx_pipeline {

constexpr size_t kServeTopK = 5;
constexpr size_t kServeMaxPayloadBytes = 16 * 1024 * 1024;  // 16 MiB hard cap
constexpr size_t kServeQueueCap = 64;                        // back-pressure

// Reference-counted connection; close deferred until the last queued
// request that names this conn has been responded to.
struct ClientConn {
  int fd;
  explicit ClientConn(int f) : fd(f) {}
  ~ClientConn() { if (fd >= 0) ::close(fd); }
  ClientConn(const ClientConn&) = delete;
  ClientConn& operator=(const ClientConn&) = delete;
};

struct PendingRequest {
  std::shared_ptr<ClientConn> conn;
  uint64_t req_id = 0;
  std::vector<uint8_t> payload;
  // If non-empty, skip Run() and emit a FAIL response.
  std::string immediate_fail;
};

class RequestQueue {
 public:
  explicit RequestQueue(size_t cap) : cap_(cap) {}
  bool push(PendingRequest req) {
    std::lock_guard<std::mutex> lk(mu_);
    if (q_.size() >= cap_) return false;
    q_.push_back(std::move(req));
    cv_.notify_one();
    return true;
  }
  void push_force(PendingRequest req) {
    std::lock_guard<std::mutex> lk(mu_);
    q_.push_back(std::move(req));
    cv_.notify_one();
  }
  bool pop(PendingRequest& out) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&]{ return !q_.empty() || stop_; });
    if (q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop_front();
    return true;
  }
  void shutdown() {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
    cv_.notify_all();
  }
 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<PendingRequest> q_;
  size_t cap_;
  bool stop_ = false;
};

inline bool write_all(int fd, const void* buf, size_t n) {
  const uint8_t* p = reinterpret_cast<const uint8_t*>(buf);
  while (n > 0) {
    ssize_t w = ::send(fd, p, n, MSG_NOSIGNAL);
    if (w < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (w == 0) return false;
    p += w; n -= static_cast<size_t>(w);
  }
  return true;
}

inline bool read_all(int fd, void* buf, size_t n) {
  uint8_t* p = reinterpret_cast<uint8_t*>(buf);
  while (n > 0) {
    ssize_t r = ::recv(fd, p, n, 0);
    if (r < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (r == 0) return false;  // EOF
    p += r; n -= static_cast<size_t>(r);
  }
  return true;
}

inline bool send_fail(int fd, uint64_t req_id, const std::string& msg) {
  uint8_t hdr[8 + 1 + 4];
  uint8_t status = 0;
  uint32_t mlen = static_cast<uint32_t>(msg.size());
  std::memcpy(hdr + 0, &req_id, 8);
  hdr[8] = status;
  std::memcpy(hdr + 9, &mlen, 4);
  if (!write_all(fd, hdr, sizeof(hdr))) return false;
  if (mlen > 0 && !write_all(fd, msg.data(), mlen)) return false;
  return true;
}

inline bool send_ok(int fd, uint64_t req_id, float infer_ms,
                    const std::vector<TopKEntry>& topk) {
  uint8_t hdr[8 + 1 + 4 + 4];
  uint8_t status = 1;
  uint32_t K = static_cast<uint32_t>(topk.size());
  std::memcpy(hdr + 0, &req_id, 8);
  hdr[8] = status;
  std::memcpy(hdr + 9, &infer_ms, 4);
  std::memcpy(hdr + 13, &K, 4);
  if (!write_all(fd, hdr, sizeof(hdr))) return false;
  for (const auto& p : topk) {
    uint8_t b[8];
    std::memcpy(b + 0, &p.first,  4);
    std::memcpy(b + 4, &p.second, 4);
    if (!write_all(fd, b, sizeof(b))) return false;
  }
  return true;
}

inline void reader_loop(std::shared_ptr<ClientConn> conn,
                         RequestQueue& q,
                         size_t expected_payload_bytes) {
  const int fd = conn->fd;
  while (true) {
    uint8_t hdr[8 + 4];
    if (!read_all(fd, hdr, sizeof(hdr))) return;
    uint64_t req_id;
    uint32_t plen;
    std::memcpy(&req_id, hdr + 0, 8);
    std::memcpy(&plen,   hdr + 8, 4);
    if (plen > kServeMaxPayloadBytes) {
      PendingRequest pr;
      pr.conn = conn;
      pr.req_id = req_id;
      pr.immediate_fail = "payload too large";
      q.push_force(std::move(pr));
      return;
    }
    std::vector<uint8_t> payload(plen);
    if (plen > 0 && !read_all(fd, payload.data(), plen)) return;
    PendingRequest pr;
    pr.conn = conn;
    pr.req_id = req_id;
    if (plen != expected_payload_bytes) {
      char err[160];
      std::snprintf(err, sizeof(err),
                    "payload size mismatch: got %u bytes, expected %zu "
                    "(model wants 3x224x224)",
                    plen, expected_payload_bytes);
      pr.immediate_fail = err;
    } else {
      pr.payload = std::move(payload);
    }
    if (!q.push(std::move(pr))) {
      PendingRequest busy;
      busy.conn = conn;
      busy.req_id = req_id;
      busy.immediate_fail = "server queue full; try again later";
      q.push_force(std::move(busy));
    }
  }
}

inline void accept_loop(int listen_fd, RequestQueue& q,
                         size_t expected_payload_bytes,
                         std::atomic<bool>& running) {
  while (running.load(std::memory_order_acquire)) {
    sockaddr_in cli{};
    socklen_t cli_len = sizeof(cli);
    int cfd = ::accept(listen_fd, reinterpret_cast<sockaddr*>(&cli), &cli_len);
    if (cfd < 0) {
      if (errno == EINTR) continue;
      if (!running.load(std::memory_order_acquire)) return;
      std::fprintf(stderr, "[serve] accept failed: %s\n", std::strerror(errno));
      return;
    }
    int one = 1;
    ::setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    std::fprintf(stderr, "[serve] client connected fd=%d\n", cfd);
    std::fflush(stderr);
    auto conn = std::make_shared<ClientConn>(cfd);
    std::thread(reader_loop, conn, std::ref(q),
                expected_payload_bytes).detach();
  }
}

// Validate the deployed model is single-input ImageNet-shape (1×3×224×224
// NCHW or 1×224×224×3 NHWC). Returns expected payload size in bytes;
// 0 on validation failure (caller refuses to serve).
inline size_t serve_validate_model(const std::vector<InputSpec>& inputs) {
  if (inputs.size() != 1) {
    std::fprintf(stderr,
                 "[serve] refuse: --serve requires exactly one input tensor, "
                 "model has %zu\n", inputs.size());
    return 0;
  }
  const auto& shape = inputs[0].shape;
  bool nchw = (shape == std::vector<uint32_t>{1, 3, 224, 224});
  bool nhwc = (shape == std::vector<uint32_t>{1, 224, 224, 3});
  if (!nchw && !nhwc) {
    std::fprintf(stderr, "[serve] refuse: model input shape is {");
    for (size_t i = 0; i < shape.size(); ++i)
      std::fprintf(stderr, "%s%u", i ? "," : "", shape[i]);
    std::fprintf(stderr, "}, not 3x224x224 (NCHW or NHWC).\n");
    return 0;
  }
  size_t numel = 1;
  for (auto d : shape) numel *= d;
  return numel * bytesPerElem(inputs[0].dtype);
}

inline int run_serve(int port,
                      const std::shared_ptr<tim::vx::Graph>& graph,
                      const std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                      const std::shared_ptr<tim::vx::Tensor>& output,
                      const std::vector<InputSpec>& specs) {
  size_t expected_bytes = serve_validate_model(specs);
  if (expected_bytes == 0) return 4;
  ::signal(SIGPIPE, SIG_IGN);

  int listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd < 0) {
    std::fprintf(stderr, "[serve] socket() failed: %s\n", std::strerror(errno));
    return 4;
  }
  int one = 1;
  ::setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (::bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    std::fprintf(stderr, "[serve] bind(0.0.0.0:%d) failed: %s\n",
                 port, std::strerror(errno));
    ::close(listen_fd);
    return 4;
  }
  if (::listen(listen_fd, 16) < 0) {
    std::fprintf(stderr, "[serve] listen() failed: %s\n", std::strerror(errno));
    ::close(listen_fd);
    return 4;
  }

  std::printf("[serve] ready: 0.0.0.0:%d  expecting %zu-byte payloads "
              "(dtype=%d, queue_cap=%zu, top_k=%zu)\n",
              port, expected_bytes,
              static_cast<int>(specs[0].dtype),
              kServeQueueCap, kServeTopK);
  std::fflush(stdout);

  RequestQueue queue(kServeQueueCap);
  std::atomic<bool> running{true};
  std::thread acceptor(accept_loop, listen_fd, std::ref(queue),
                       expected_bytes, std::ref(running));

  using Clock = std::chrono::steady_clock;
  size_t elem = bytesPerElem(specs[0].dtype);

  while (true) {
    PendingRequest req;
    if (!queue.pop(req)) break;

    if (!req.immediate_fail.empty()) {
      send_fail(req.conn->fd, req.req_id, req.immediate_fail);
      continue;
    }

    std::vector<uint8_t> tvx_bytes(req.payload.size());
    layoutConvert(req.payload.data(), tvx_bytes.data(),
                  specs[0].shape, elem, /*from_mlir_to_tvx=*/true);
    if (!inputs[0]->CopyDataToTensor(
            tvx_bytes.data(),
            static_cast<uint32_t>(tvx_bytes.size()))) {
      send_fail(req.conn->fd, req.req_id, "CopyDataToTensor failed");
      continue;
    }

    auto t0 = Clock::now();
    if (!graph->Run()) {
      double ms = std::chrono::duration<double, std::milli>(
          Clock::now() - t0).count();
      std::fprintf(stderr,
                   "[serve] graph->Run failed after %.2f ms — NPU likely "
                   "hung; aborting server.\n", ms);
      std::fflush(stderr);
      send_fail(req.conn->fd, req.req_id,
                "graph->Run failed (NPU error); server aborting");
      std::_Exit(3);
    }
    double infer_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - t0).count();

    std::vector<float> out_buf;
    if (!dequantize_output(output, out_buf)) {
      send_fail(req.conn->fd, req.req_id, "dequantize_output failed");
      continue;
    }
    auto topk = compute_topk(out_buf, kServeTopK);
    send_ok(req.conn->fd, req.req_id,
            static_cast<float>(infer_ms), topk);
  }

  running.store(false, std::memory_order_release);
  ::shutdown(listen_fd, SHUT_RDWR);
  ::close(listen_fd);
  if (acceptor.joinable()) acceptor.join();
  return 0;
}

} // namespace timvx_pipeline

#endif // TIMVX_PIPELINE_SERVE_H
