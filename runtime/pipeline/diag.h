// pipeline/diag.h — small diagnostic helpers used by every run mode:
//
//   * `read_file(path)`            — slurp a binary file (or exit on error).
//   * `peakRssKb()`                — VmHWM scrape; used by the [stage] log.
//   * `Stage` / `print_stage()`    — uniform "[stage] X done in Y ms"
//                                    print, with optional peak-RSS suffix.
//   * `print_output(tensor)`       — dequantize-and-stats summary used by
//                                    the one-shot path. Honours the
//                                    `TIMVX_DUMP_OUTPUT` env var.
//
// Header-only; no shared state.

#ifndef TIMVX_PIPELINE_DIAG_H
#define TIMVX_PIPELINE_DIAG_H

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "tim/vx/tensor.h"

#include "pipeline/tensor_io.h"

namespace timvx_pipeline {

inline std::vector<uint8_t> read_file(const char* path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::fprintf(stderr, "cannot open input: %s\n", path);
    std::exit(1);
  }
  std::streamsize n = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<uint8_t> buf(static_cast<size_t>(n));
  f.read(reinterpret_cast<char*>(buf.data()), n);
  return buf;
}

// Cheap peak-RSS sampler. Compile-time memory usage by libCLC / the
// EVIS shader cache can spike non-monotonically — a leaky JIT path
// shows up as the per-stage delta blowing past expected runtime
// tensor sizes. Reads /proc/self/status' VmHWM line (high-water mark).
inline long peakRssKb() {
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line))
    if (line.rfind("VmHWM:", 0) == 0)
      return std::strtol(line.c_str() + 6, nullptr, 10);
  return -1;
}

inline void print_stage(const char* tag, double ms) {
  long rss = peakRssKb();
  if (rss >= 0)
    std::printf("[stage] %s in %.2f ms (peak RSS %.1f MiB)\n",
                tag, ms, rss / 1024.0);
  else
    std::printf("[stage] %s in %.2f ms\n", tag, ms);
  std::fflush(stdout);
}

inline double msSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
}

constexpr size_t kPrintFirst = 16;

// Stats summary for the one-shot path. Optionally writes the full
// dequantized fp32 buffer to `$TIMVX_DUMP_OUTPUT` for byte-level
// diagnostics by debug scripts.
inline void print_output(const std::shared_ptr<tim::vx::Tensor>& t) {
  const auto& shape = t->GetShape();
  size_t numel = 1;
  for (auto d : shape) numel *= d;

  std::printf("output shape: {");
  for (size_t i = 0; i < shape.size(); ++i)
    std::printf("%s%u", i ? ", " : "", shape[i]);
  std::printf("}, %zu elements\n", numel);
  if (numel == 0) return;

  std::vector<float> buf;
  if (!dequantize_output(t, buf)) {
    std::fprintf(stderr,
                 "print_output: unsupported dtype %d; raw bytes only.\n",
                 static_cast<int>(t->GetDataType()));
    return;
  }

  size_t print_n = std::min(numel, kPrintFirst);
  std::printf("first %zu values (dequantized): [", print_n);
  for (size_t i = 0; i < print_n; ++i)
    std::printf("%s%.6f", i ? ", " : "", buf[i]);
  std::printf("%s]\n", numel > print_n ? ", ..." : "");

  auto max_it = std::max_element(buf.begin(), buf.end());
  auto min_it = std::min_element(buf.begin(), buf.end());
  size_t argmax = static_cast<size_t>(std::distance(buf.begin(), max_it));
  float sum = std::accumulate(buf.begin(), buf.end(), 0.0f);
  std::printf("argmax: %zu (value=%.6f)\n", argmax, *max_it);
  std::printf("min=%.6f  max=%.6f  mean=%.6f\n",
              *min_it, *max_it, sum / static_cast<float>(numel));

  if (const char* dump = std::getenv("TIMVX_DUMP_OUTPUT")) {
    if (FILE* fp = std::fopen(dump, "wb")) {
      std::fwrite(buf.data(), sizeof(float), numel, fp);
      std::fclose(fp);
      std::fprintf(stderr, "[dump] wrote %zu fp32 values to %s\n",
                   numel, dump);
    } else {
      std::fprintf(stderr, "[dump] cannot open %s for write\n", dump);
    }
  }
}

} // namespace timvx_pipeline

#endif // TIMVX_PIPELINE_DIAG_H
