// Declaration: This is a benchmark-only file that 
// is AI-generated and reviewed by human.
// 
// pipeline/eval.h — `--eval-dir <dir> --labels <file>` plumbing for the
// runner. One-time ImageNet top-1/top-5 sweep mode.
//
// Hardcoded to mode = "cpu-sequential" (CPU preproc + NPU infer + CPU
// top-K postproc, no zero-copy, no pipelining). The pre-transformed
// images on disk are already 224x224 RGB (see prep_imagenet_val.py),
// so CpuJpegPreProcessor's cv::resize is a no-op and the only CPU
// work per image is libjpeg decode + mean/std + requantize + layout
// pack — well under the NPU's per-image time, so sequential is fine
// (no benefit to pipelining when CPU is far from the bottleneck).
//
// Wire format on disk
// -------------------
//   <eval_dir>/*.jpg    — images named like 00000.jpg, 00001.jpg, ...
//   <labels_path>       — JSON: {"00000": 91, "00001": 171, ...}
//
// The label value is the ImageNet class id (0..999); model output is a
// 1000-class softmax so no background offset.

#ifndef TIMVX_PIPELINE_EVAL_H
#define TIMVX_PIPELINE_EVAL_H

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"

#include "pipeline/factory.h"
#include "pipeline/input_spec.h"
#include "pipeline/orchestrator.h"
#include "pipeline/postproc.h"
#include "pipeline/tensor_io.h"

namespace timvx_pipeline {

// Minimal JSON parser for `{"key1": int1, "key2": int2, ...}`. Doesn't
// handle nested objects, arrays, strings-as-values, unicode escapes —
// labels.json is flat with int values, that's all we need.
inline std::unordered_map<std::string, int>
load_flat_int_labels(const std::string& path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    std::fprintf(stderr, "[eval] cannot open labels file: %s\n", path.c_str());
    return {};
  }
  std::stringstream ss;
  ss << f.rdbuf();
  std::string s = ss.str();

  std::unordered_map<std::string, int> out;
  size_t i = 0, n = s.size();
  auto skip_ws = [&]() { while (i < n && std::isspace((unsigned char)s[i])) ++i; };
  skip_ws();
  if (i >= n || s[i] != '{') {
    std::fprintf(stderr, "[eval] labels: expected '{' at start\n");
    return {};
  }
  ++i;
  while (i < n) {
    skip_ws();
    if (i < n && s[i] == '}') { ++i; break; }
    if (i >= n || s[i] != '"') {
      std::fprintf(stderr, "[eval] labels: expected '\"' at offset %zu\n", i);
      return {};
    }
    ++i;
    size_t key_start = i;
    while (i < n && s[i] != '"') ++i;
    if (i >= n) {
      std::fprintf(stderr, "[eval] labels: unterminated key\n");
      return {};
    }
    std::string key = s.substr(key_start, i - key_start);
    ++i;  // closing "
    skip_ws();
    if (i >= n || s[i] != ':') {
      std::fprintf(stderr, "[eval] labels: expected ':' after key '%s'\n",
                   key.c_str());
      return {};
    }
    ++i;
    skip_ws();
    size_t val_start = i;
    if (i < n && (s[i] == '-' || s[i] == '+')) ++i;
    while (i < n && std::isdigit((unsigned char)s[i])) ++i;
    if (val_start == i) {
      std::fprintf(stderr, "[eval] labels: expected int after key '%s'\n",
                   key.c_str());
      return {};
    }
    int v = std::atoi(s.substr(val_start, i - val_start).c_str());
    out[key] = v;
    skip_ws();
    if (i < n && s[i] == ',') { ++i; continue; }
    if (i < n && s[i] == '}') { ++i; break; }
    std::fprintf(stderr, "[eval] labels: expected ',' or '}' at offset %zu\n", i);
    return {};
  }
  return out;
}

// Sorted list of files in `dir` (basenames only, not paths) matching one
// of the lowercase extensions in `exts` (each must include the leading
// dot, e.g. ".jpg").
inline std::vector<std::string> list_dir_by_ext(
    const std::string& dir, std::initializer_list<const char*> exts) {
  std::vector<std::string> names;
  DIR* d = ::opendir(dir.c_str());
  if (!d) {
    std::fprintf(stderr, "[eval] cannot open dir: %s\n", dir.c_str());
    return names;
  }
  while (auto* ent = ::readdir(d)) {
    std::string name = ent->d_name;
    if (name.empty() || name[0] == '.') continue;
    std::string lc = name;
    for (auto& c : lc) c = static_cast<char>(std::tolower((unsigned char)c));
    for (auto* e : exts) {
      size_t elen = std::strlen(e);
      if (lc.size() > elen &&
          lc.compare(lc.size() - elen, elen, e) == 0) {
        names.push_back(name);
        break;
      }
    }
  }
  ::closedir(d);
  std::sort(names.begin(), names.end());
  return names;
}

inline std::vector<std::string> list_jpegs(const std::string& dir) {
  return list_dir_by_ext(dir, {".jpg", ".jpeg"});
}
inline std::vector<std::string> list_bins(const std::string& dir) {
  return list_dir_by_ext(dir, {".bin"});
}

inline std::vector<uint8_t> read_file_bytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) return {};
  f.seekg(0, std::ios::end);
  std::streamsize n = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<uint8_t> buf(static_cast<size_t>(n));
  f.read(reinterpret_cast<char*>(buf.data()), n);
  return buf;
}

// Back-compat alias.
inline std::vector<uint8_t> read_jpeg(const std::string& path) {
  return read_file_bytes(path);
}

// Run the eval sweep. Hardcoded to mode=cpu-sequential. Returns process
// exit code (0 = clean, 3 = some images failed inference, 4 = setup error).
//
// When `passthrough` is true, the runner expects `eval_dir` to contain
// `.bin` files whose bytes are the already-quantized, already-WHCN-packed
// model input (size = product(spec.shape) * bytes_per_element). The
// orchestrator's preprocessor becomes CpuPassthroughPreProcessor so no
// libjpeg / normalize / requantize / channel-pack runs on the C++ side.
// Use this to bisect "is the C++ preproc wrong?" vs "is the lowered
// model wrong?": feed identical bytes to both the TFLite reference and
// our runner and compare top-1.
inline int run_eval(const std::shared_ptr<tim::vx::Context>& /*ctx*/,
                    const std::shared_ptr<tim::vx::Graph>& graph,
                    const std::shared_ptr<tim::vx::Tensor>& input,
                    const std::shared_ptr<tim::vx::Tensor>& output,
                    const InputSpec& spec,
                    const std::string& eval_dir,
                    const std::string& labels_path,
                    size_t limit,
                    bool passthrough = false) {
  auto labels = load_flat_int_labels(labels_path);
  if (labels.empty()) {
    std::fprintf(stderr, "[eval] no labels loaded from %s\n",
                 labels_path.c_str());
    return 4;
  }
  std::printf("[eval] loaded %zu labels from %s\n", labels.size(),
              labels_path.c_str());

  auto names = passthrough ? list_bins(eval_dir) : list_jpegs(eval_dir);
  if (names.empty()) {
    std::fprintf(stderr, "[eval] no %s files found in %s\n",
                 passthrough ? ".bin" : "JPEG", eval_dir.c_str());
    return 4;
  }
  if (limit > 0 && limit < names.size()) names.resize(limit);
  std::printf("[eval] %zu %s files to evaluate in %s%s\n",
              names.size(), passthrough ? ".bin" : "JPEG",
              eval_dir.c_str(),
              passthrough ? "  (passthrough mode — no C++ preproc)" : "");

  // Build the cpu-sequential orchestrator (CPU pre + CPU post, no
  // zerocopy). num_classes / Sout / Zout are derived from the output
  // tensor; ppu_src_shape unused for the cpu backend.
  auto quant = output->GetQuantization();
  double Sout = 0.0; int32_t Zout = 0;
  if (quant.Type() != tim::vx::QuantType::NONE && !quant.Scales().empty()) {
    Sout = quant.Scales()[0];
    if (!quant.ZeroPoints().empty()) Zout = quant.ZeroPoints()[0];
  }
  uint32_t num_classes = 1;
  for (auto d : output->GetShape()) num_classes *= d;

  std::unique_ptr<PreProcessor>  pre_owned;
  std::unique_ptr<PostProcessor> post_owned;
  if (passthrough) {
    pre_owned  = std::make_unique<CpuPassthroughPreProcessor>(spec);
    post_owned = std::make_unique<CpuTopKPostProcessor>(/*K=*/5);
  } else {
    Backend backend = make_backend("cpu-sequential", spec, num_classes,
                                    Sout, Zout);
    if (!backend.pre || !backend.post) {
      std::fprintf(stderr, "[eval] make_backend failed\n");
      return 4;
    }
    pre_owned  = std::move(backend.pre);
    post_owned = std::move(backend.post);
  }

  // Sanity-check passthrough bin size before launching the sweep — easy
  // mistake to feed .bins from a different model spec.
  size_t expected_bytes = pre_owned->expected_input_bytes();
  if (passthrough && expected_bytes != 0) {
    auto probe = read_file_bytes(eval_dir + "/" + names.front());
    if (probe.size() != expected_bytes) {
      std::fprintf(stderr,
                   "[eval] passthrough: %s is %zu bytes, expected %zu "
                   "(shape product * %zu bytes/elem)\n",
                   names.front().c_str(), probe.size(), expected_bytes,
                   bytesPerElem(spec.dtype));
      return 4;
    }
  }

  ModeConfig cfg{"cpu-sequential"};
  std::vector<WorkerGraph> wgs;
  wgs.push_back({nullptr, graph, input, output});
  auto orch = make_orchestrator(cfg, std::move(wgs),
                                pre_owned.get(), post_owned.get(), nullptr);
  if (!orch) {
    std::fprintf(stderr, "[eval] make_orchestrator failed\n");
    return 4;
  }

  size_t total = 0, n_ok = 0, n_fail = 0, n_unlabeled = 0;
  size_t top1 = 0, top5 = 0;
  double infer_ms_sum = 0.0;
  auto t_start = std::chrono::steady_clock::now();

  for (size_t i = 0; i < names.size(); ++i) {
    const std::string& name = names[i];
    // Stem = filename without ".jpg" / ".jpeg" — used as labels.json key.
    std::string stem = name;
    auto dot = stem.find_last_of('.');
    if (dot != std::string::npos) stem.resize(dot);
    auto lit = labels.find(stem);
    if (lit == labels.end()) {
      ++n_unlabeled;
      continue;
    }
    int gt = lit->second;

    auto bytes = read_file_bytes(eval_dir + "/" + name);
    if (bytes.empty()) {
      ++n_fail;
      continue;
    }

    auto fut = orch->submit(std::move(bytes));
    auto res = fut.get();
    ++total;
    if (!res.error.empty()) {
      ++n_fail;
      if (n_fail < 10) {
        std::fprintf(stderr, "[eval] %s: %s\n",
                     name.c_str(), res.error.c_str());
      }
      continue;
    }
    ++n_ok;
    infer_ms_sum += res.infer_ms;

    // Top-K: vector of (class_id, score). Already sorted descending.
    if (!res.topk.empty()) {
      if (static_cast<int>(res.topk.front().first) == gt) ++top1;
      for (const auto& e : res.topk) {
        if (static_cast<int>(e.first) == gt) { ++top5; break; }
      }
    }

    if ((i + 1) % 500 == 0 || i + 1 == names.size()) {
      double elapsed_s = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - t_start).count();
      double rate = total / elapsed_s;
      double eta_s = rate > 0 ? (names.size() - (i + 1)) / rate : 0.0;
      std::printf("[eval] %zu/%zu  ok=%zu fail=%zu  "
                  "top1=%.2f%% top5=%.2f%%  "
                  "infer=%.2fms  rate=%.1f img/s  eta=%.0fs\n",
                  i + 1, names.size(), n_ok, n_fail,
                  n_ok ? 100.0 * top1 / n_ok : 0.0,
                  n_ok ? 100.0 * top5 / n_ok : 0.0,
                  n_ok ? infer_ms_sum / n_ok : 0.0,
                  rate, eta_s);
      std::fflush(stdout);
    }
  }

  double wall_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t_start).count();
  std::printf("\n[eval] === summary ===\n");
  std::printf("[eval]   images_seen   = %zu\n", names.size());
  std::printf("[eval]   labelled      = %zu\n", names.size() - n_unlabeled);
  std::printf("[eval]   unlabelled    = %zu\n", n_unlabeled);
  std::printf("[eval]   inferred_ok   = %zu\n", n_ok);
  std::printf("[eval]   inferred_fail = %zu\n", n_fail);
  std::printf("[eval]   top1_hits     = %zu\n", top1);
  std::printf("[eval]   top5_hits     = %zu\n", top5);
  if (n_ok > 0) {
    std::printf("[eval]   top1_accuracy = %.4f%%\n", 100.0 * top1 / n_ok);
    std::printf("[eval]   top5_accuracy = %.4f%%\n", 100.0 * top5 / n_ok);
    std::printf("[eval]   mean_infer_ms = %.3f\n", infer_ms_sum / n_ok);
  }
  std::printf("[eval]   wall_seconds  = %.1f\n", wall_s);
  std::fflush(stdout);
  return n_fail > 0 ? 3 : 0;
}

} // namespace timvx_pipeline

#endif // TIMVX_PIPELINE_EVAL_H
