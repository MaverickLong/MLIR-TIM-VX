// pipeline/preproc.h — `PreProcessor` interface + implementations.
//
// A PreProcessor turns raw input bytes (one image, in whatever encoding
// the caller chose to write to disk / read off the socket) into a byte
// buffer in the exact layout the model's input tensor expects (TIM-VX
// innermost-first byte order, quantized to the model's input dtype).
//
// Provided implementations
// ------------------------
//   CpuPassthroughPreProcessor   : input bytes are already-quantized
//                                  MLIR-row-major tensor bytes; just
//                                  layout-converts to TIM-VX innermost-
//                                  first. Matches the legacy
//                                  `run_timvx.py` pipeline.
//
//   CpuNormalizeQuantPreProcessor: input bytes are fp32, NHWC, in
//                                  [0,1] or similar pre-normalize
//                                  range. Applies (x - mean) / std
//                                  then quantize-to-u8 with the model's
//                                  (S, Z), then layout-converts.
//
//   CpuJpegPreProcessor          : libjpeg decode → resize → normalize
//                                  → quantize → layout convert. All
//                                  CPU. Default for cpu-* / cpu_zero_copy-*.
//
//   PpuJpegPreProcessor          : libjpeg decode on CPU, then a single
//                                  TIM-VX/PPU graph runs resize +
//                                  normalize + requantize end-to-end on
//                                  the shader unit and writes bytes in
//                                  the model's expected layout. Used for
//                                  ppu-* / ppu_zero_copy-* modes. The
//                                  PPU dispatch shares a queue with the
//                                  NN engine on this chip, so PPU pre
//                                  tends to serialize with NPU infer —
//                                  benchmark variant.
//
// Each PreProcessor instance owns whatever state it needs (TIM-VX
// graphs, scratch buffers, etc.). `process(raw)` must be reentrant —
// the Pool/Hybrid orchestrators call it from multiple threads.
//
// Zero-copy interface
// -------------------
// `process_inplace(raw, io_buf)` writes the NN-ready bytes directly
// into `io_buf->in_data` (the host buffer bound to the NN's input
// tensor) and avoids an intermediate `std::vector`. For PPU
// preprocessors the implementation also writes the CPU-decoded JPEG
// bytes into `io_buf->pre_in_data` (bound to the PPU graph's input
// tensor) and runs the PPU graph, so `io_buf->in_data` is filled by
// PPU DMA writes rather than by the CPU. The factory sizes the
// IoBufferPool's pre_in_bytes from `expected_pre_in_bytes()`.
//
// `sync_for_infer(io_buf, nn_input)` is a hook the orchestrator calls
// AFTER SwapHandle-ing the NN input/output tensors and BEFORE
// `graph->Run()`. The CPU base impl calls FlushCacheForHandle on the
// NN input tensor to push CPU writes of `io_buf->in_data` to DDR; the
// PPU override is a no-op because the PPU graph already wrote into
// `io_buf->in_data` via DMA, and a CPU flush there would overwrite
// fresh PPU data with stale CPU cache lines.

#ifndef TIMVX_PIPELINE_PREPROC_H
#define TIMVX_PIPELINE_PREPROC_H

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"
#include "tim/vx/types.h"
#include "tim/vx/ops/transpose.h"

#include "pipeline/byte_layout.h"
#include "pipeline/image.h"
#include "pipeline/input_spec.h"
#include "pipeline/iobuffer.h"
#include "pipeline/jpeg.h"
#include "pipeline/preproc_rgb_op.h"

namespace timvx_pipeline {

class PreProcessor {
 public:
  virtual ~PreProcessor() = default;
  // Map raw input bytes → tensor-ready bytes. Returned buffer's size
  // must equal `expected_output_bytes()`.
  virtual std::vector<uint8_t> process(const std::vector<uint8_t>& raw) = 0;
  // Zero-copy variant: write the NN-ready bytes DIRECTLY into
  // `io_buf->in_data` (the host buffer bound to the NN's input
  // tensor). For PPU preprocessors, the implementation also writes the
  // decoded JPEG bytes into `io_buf->pre_in_data` (bound to the PPU
  // graph's input tensor) and runs the PPU graph end-to-end, so the
  // CPU never touches `io_buf->in_data` — the PPU writes it via DMA.
  //
  // Default impl delegates to `process()` and memcpys into in_data;
  // derived classes that can avoid the intermediate vector or that
  // need the pre_in_data slot override.
  virtual bool process_inplace(const std::vector<uint8_t>& raw,
                                IoBuffer* io_buf) {
    if (!io_buf) return false;
    auto v = process(raw);
    if (v.size() != io_buf->in_size) return false;
    std::memcpy(io_buf->in_data, v.data(), v.size());
    return true;
  }
  // Bytes the model's input tensor expects. Caller uses this to
  // sanity-check the produced buffer before CopyDataToTensor.
  virtual size_t expected_output_bytes() const = 0;
  // Bytes the raw input is expected to be. Used in benchmark / eval modes
  // to size buffer allocations and to validate inputs.
  virtual size_t expected_input_bytes() const = 0;
  // Bytes the preprocessor needs in the per-slot "pre_in" buffer (the
  // PPU graph's input bytes — decoded RGB). Zero for CPU preprocessors
  // which don't use a separate pre_in slot. PPU preprocessors return
  // src_w * src_h * channels so the IoBufferPool can size each slot
  // accordingly. Querying happens once after preprocessor
  // construction (the factory uses it to size the pool).
  virtual size_t expected_pre_in_bytes() const { return 0; }
  // Identifier for diagnostic logs.
  virtual const char* name() const = 0;
  // Bridge between the pre stage and the NN's Run() in zerocopy mode.
  // Called AFTER `nn_input->SwapHandle(io_buf->in_data, ...)` and
  // BEFORE `graph->Run()`. The default impl flushes the CPU cache for
  // the NN input handle so the NPU sees the latest bytes the CPU just
  // wrote. PPU preprocessors override to a no-op: the PPU graph's
  // Run() already deposited bytes in DDR via DMA, so a CPU flush here
  // would overwrite them with stale cache lines.
  virtual void sync_for_infer(
      IoBuffer* /*io_buf*/,
      const std::shared_ptr<tim::vx::Tensor>& nn_input) {
    if (nn_input) nn_input->FlushCacheForHandle();
  }
};

// ── 1. CPU pass-through ────────────────────────────────────────────────
// Input == output bytes except for layout convert. Matches the legacy
// `run_timvx.py` flow where Python already normalized + quantized.
class CpuPassthroughPreProcessor : public PreProcessor {
 public:
  explicit CpuPassthroughPreProcessor(const InputSpec& spec) : spec_(spec) {
    elem_ = bytesPerElem(spec.dtype);
    numel_ = 1;
    for (auto d : spec.shape) numel_ *= d;
  }
  std::vector<uint8_t> process(const std::vector<uint8_t>& raw) override {
    // Native (folded-transpose) model: spec.shape is already TIM-VX
    // innermost-first, so the layout-convert no longer applies — the
    // caller is expected to hand us bytes already in that order. Pass
    // through verbatim.
    if (spec_.timvx_native_layout) return raw;
    std::vector<uint8_t> tvx(raw.size());
    layoutConvert(raw.data(), tvx.data(), spec_.shape, elem_,
                  /*from_mlir_to_tvx=*/true);
    return tvx;
  }
  size_t expected_output_bytes() const override { return numel_ * elem_; }
  size_t expected_input_bytes()  const override { return numel_ * elem_; }
  const char* name() const override { return "cpu-passthrough"; }

 private:
  InputSpec spec_;
  size_t elem_, numel_;
};

// ── 2. CPU normalize + quantize ─────────────────────────────────────────
// Input: fp32 NHWC in [0,1]. Computes (x - mean) / std elementwise, then
// quantizes to the model's u8 (S, Z), then layout-converts.
//
// Mean and std are passed at construction; ImageNet defaults are
// {0.485, 0.456, 0.406} and {0.229, 0.224, 0.225} for RGB.
class CpuNormalizeQuantPreProcessor : public PreProcessor {
 public:
  CpuNormalizeQuantPreProcessor(const InputSpec& spec,
                                 std::array<float, 3> mean,
                                 std::array<float, 3> std)
      : spec_(spec), mean_(mean), std_(std) {
    numel_ = 1;
    for (auto d : spec.shape) numel_ *= d;
    elem_ = bytesPerElem(spec.dtype);
    if (spec.dtype != tim::vx::DataType::UINT8 &&
        spec.dtype != tim::vx::DataType::INT8) {
      std::fprintf(stderr,
                   "CpuNormalizeQuantPreProcessor: only u8/i8 input "
                   "dtypes supported; got %d\n",
                   static_cast<int>(spec.dtype));
    }
  }
  std::vector<uint8_t> process(const std::vector<uint8_t>& raw) override {
    // raw is fp32 NHWC, MLIR row-major. Layout: shape[0..rank-1] with
    // the channel dim being the rightmost-but-not-last (typically
    // shape[rank-1] = 3 for NHWC). We don't enforce shape semantics
    // here — caller is responsible for matching mean/std to the right
    // channel layout.
    const float* src = reinterpret_cast<const float*>(raw.data());
    size_t channels = spec_.shape.empty() ? 1 : spec_.shape.back();
    if (channels > 3) channels = 3; // ImageNet RGB
    std::vector<uint8_t> mlir_bytes(numel_ * elem_);
    for (size_t i = 0; i < numel_; ++i) {
      size_t c = i % channels;
      float v = (src[i] - mean_[c]) / std_[c];
      // Quantize: byte = round(v / S + Z), saturate.
      double q = v / spec_.quant_scale + spec_.quant_zp;
      long iq = std::lround(q);
      if (spec_.dtype == tim::vx::DataType::UINT8) {
        if (iq < 0) iq = 0;
        if (iq > 255) iq = 255;
        mlir_bytes[i] = static_cast<uint8_t>(iq);
      } else {  // INT8
        if (iq < -128) iq = -128;
        if (iq > 127) iq = 127;
        mlir_bytes[i] = static_cast<uint8_t>(static_cast<int8_t>(iq));
      }
    }
    std::vector<uint8_t> tvx(mlir_bytes.size());
    layoutConvert(mlir_bytes.data(), tvx.data(), spec_.shape, elem_,
                  /*from_mlir_to_tvx=*/true);
    return tvx;
  }
  size_t expected_output_bytes() const override { return numel_ * elem_; }
  size_t expected_input_bytes()  const override { return numel_ * sizeof(float); }
  const char* name() const override { return "cpu-normalize-quant"; }

 private:
  InputSpec spec_;
  std::array<float, 3> mean_;
  std::array<float, 3> std_;
  size_t numel_, elem_;
};

// ── 3. CPU full JPEG pipeline ───────────────────────────────────────────
// Input: raw JPEG file bytes (the contents of e.g. cat105.jpg).
// Steps: libjpeg decode → bilinear resize → mean/std normalize →
// quantize → layout convert. All CPU.
//
// Per-iteration cost on the A733 big core for 480×360 source →
// 224×224 RGB888 → u8 tensor: ~1-2 ms decode + ~150 µs resize +
// ~600 µs normalize/quantize + ~50 µs layout convert ≈ 2-3 ms total.
// That's a real fraction of the resnet50 7.5 ms NPU time — exactly
// what the streaming pipeline needs to hide.
class CpuJpegPreProcessor : public PreProcessor {
 public:
  CpuJpegPreProcessor(const InputSpec& spec,
                       std::array<float, 3> mean,
                       std::array<float, 3> std)
      : spec_(spec), mean_(mean), std_(std),
        native_(spec.timvx_native_layout) {
    numel_ = 1;
    for (auto d : spec.shape) numel_ *= d;
    elem_ = bytesPerElem(spec.dtype);
    // Pre-compute vendor-form constants once. Vendor's PRE_PROCESS_RGB
    // op applies `(x - MEAN) * SCALE` directly on uint8 pixels, where
    // MEAN = mean*255 and SCALE = 1/(std*255). This is mathematically
    // equivalent to (x/255 - mean)/std but matches the vendor's float
    // arithmetic byte-for-byte. With imagenet defaults the constants are
    // MEAN  = {123.675, 116.28, 103.53}
    // SCALE = {0.0171247, 0.0175070, 0.0174291}
    // (verified against the vendor's NBG-baked params: max abs diff
    // ~1e-7).
    for (int c = 0; c < 3; ++c) {
      mean_x255_[c]    = mean_[c] * 255.0f;
      inv_std_x255_[c] = 1.0f / (std_[c] * 255.0f);
    }
    if (native_ && spec.shape.size() == 4) {
      // `timvx-fold-input-transpose` folded the entry transpose: shape is
      // already TIM-VX innermost-first WHCN = {W, H, C, N}. We emit the
      // channel-planar (CHW) byte order this implies directly, no
      // layout-convert (see process()).
      dst_w_ = spec.shape[0];
      dst_h_ = spec.shape[1];
      layout_nhwc_ = false;  // unused on the native path
    } else if (spec.shape.size() == 4) {
      // Heuristic: NCHW shape is {N, C, H, W} (C=3), NHWC is {N, H, W, C}.
      // Pick H, W accordingly so resize knows the target.
      if (spec.shape[1] == 3) {
        layout_nhwc_ = false;
        dst_h_ = spec.shape[2];
        dst_w_ = spec.shape[3];
      } else {
        layout_nhwc_ = true;
        dst_h_ = spec.shape[1];
        dst_w_ = spec.shape[2];
      }
    } else {
      layout_nhwc_ = true;
      dst_h_ = 224;
      dst_w_ = 224;
    }
  }

  std::vector<uint8_t> process(const std::vector<uint8_t>& raw) override {
    // 1+2) Decode + (optional) resize via OpenCV. cv::imdecode wraps
    // libjpeg-turbo with SIMD-optimised YCbCr→RGB; cv::resize uses
    // NEON intrinsics on aarch64. The vendor's reference CPU pre
    // (class_pre.cpp) uses the same OpenCV stack; matching it lets
    // us isolate the cost of our remaining normalize / quantize /
    // layout-convert passes when comparing against the vendor.
    cv::Mat raw_mat(1, static_cast<int>(raw.size()), CV_8UC1,
                    const_cast<uint8_t*>(raw.data()));
    cv::Mat bgr = cv::imdecode(raw_mat, cv::IMREAD_COLOR);
    if (bgr.empty()) {
      std::fprintf(stderr, "CpuJpegPreProcessor: cv::imdecode failed\n");
      return {};
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    if (rgb.rows != static_cast<int>(dst_h_) ||
        rgb.cols != static_cast<int>(dst_w_)) {
      cv::resize(rgb, rgb, cv::Size(static_cast<int>(dst_w_),
                                     static_cast<int>(dst_h_)));
    }
    // 3 + 4) normalize + quantize, written in the model's MLIR layout.
    //
    // For NHWC (shape = {N, H, W, C}), the destination iteration order
    // matches the source row order: idx = (y * W + x) * C + c.
    // For NCHW (shape = {N, C, H, W}), we shuffle: idx = (c * H + y) * W + x.
    //
    // The byte buffer here is in MLIR row-major (rightmost dim is
    // innermost in memory); layout_convert at the tail flips to TIM-VX
    // innermost-first.
    std::vector<uint8_t> mlir_bytes(numel_ * elem_);
    const uint8_t* src = rgb.data;
    auto quant_byte = [&](float v) -> uint8_t {
      double q = v / spec_.quant_scale + spec_.quant_zp;
      long iq = std::lround(q);
      if (spec_.dtype == tim::vx::DataType::UINT8) {
        if (iq < 0) iq = 0;
        if (iq > 255) iq = 255;
        return static_cast<uint8_t>(iq);
      }
      // INT8
      if (iq < -128) iq = -128;
      if (iq > 127) iq = 127;
      return static_cast<uint8_t>(static_cast<int8_t>(iq));
    };

    // Native (folded-transpose) path: write WHCN-canonical bytes directly
    // — channel-planar, W innermost: out[c*H*W + h*W + w]. That IS the
    // byte order conv2d reads off the input tensor once the entry
    // transpose is folded into the arg, so there's no layout-convert. The
    // index is identical to the legacy NCHW branch's `mlir_bytes` index;
    // we just skip the trailing TIM-VX flip (it would have undone exactly
    // what the now-gone device transpose used to re-do).
    if (native_) {
      const size_t plane = static_cast<size_t>(dst_w_) * dst_h_;
      for (int c = 0; c < 3; ++c) {
        for (uint32_t y = 0; y < dst_h_; ++y) {
          for (uint32_t x = 0; x < dst_w_; ++x) {
            float n = (static_cast<float>(src[(y * dst_w_ + x) * 3 + c])
                        - mean_x255_[c]) * inv_std_x255_[c];
            mlir_bytes[c * plane + y * dst_w_ + x] = quant_byte(n);
          }
        }
      }
      return mlir_bytes;  // already TIM-VX innermost-first
    }

    if (layout_nhwc_) {
      for (uint32_t y = 0; y < dst_h_; ++y) {
        for (uint32_t x = 0; x < dst_w_; ++x) {
          for (int c = 0; c < 3; ++c) {
            float n = (static_cast<float>(src[(y * dst_w_ + x) * 3 + c])
                        - mean_x255_[c]) * inv_std_x255_[c];
            size_t off = (y * dst_w_ + x) * 3 + c;
            mlir_bytes[off] = quant_byte(n);
          }
        }
      }
    } else {
      for (int c = 0; c < 3; ++c) {
        for (uint32_t y = 0; y < dst_h_; ++y) {
          for (uint32_t x = 0; x < dst_w_; ++x) {
            float n = (static_cast<float>(src[(y * dst_w_ + x) * 3 + c])
                        - mean_x255_[c]) * inv_std_x255_[c];
            size_t off = ((c * dst_h_) + y) * dst_w_ + x;
            mlir_bytes[off] = quant_byte(n);
          }
        }
      }
    }
    // 5) layout convert to TIM-VX innermost-first.
    std::vector<uint8_t> tvx(mlir_bytes.size());
    layoutConvert(mlir_bytes.data(), tvx.data(), spec_.shape, elem_,
                  /*from_mlir_to_tvx=*/true);
    return tvx;
  }
  size_t expected_output_bytes() const override { return numel_ * elem_; }
  // Variable: depends on the JPEG's compressed size. Use a sentinel; the
  // bench harness just feeds the file's actual size and the JPEG header
  // tells the decoder the real dims.
  size_t expected_input_bytes() const override { return 0; /* variable */ }
  const char* name() const override { return "cpu-jpeg"; }

 private:
  InputSpec spec_;
  std::array<float, 3> mean_, std_;
  // Vendor-form constants pre-computed from (mean, std). Used in the
  // hot normalize loop to apply `(x - MEAN) * SCALE` byte-for-byte
  // identical to the PPU's PRE_PROCESS_RGB op.
  std::array<float, 3> mean_x255_{}, inv_std_x255_{};
  size_t numel_, elem_;
  uint32_t dst_h_, dst_w_;
  bool layout_nhwc_;
  bool native_;  // shape is folded TIM-VX WHCN; emit bytes directly
};

// ── 4. PPU JPEG pipeline (FULL-PPU: PRE_PROCESS_RGB + Transpose) ────────
// Input: raw JPEG file bytes (same as CpuJpegPreProcessor).
//
// Pipeline
// --------
//   CPU (OpenCV): cv::imdecode → cv::cvtColor BGR→RGB. The resulting
//                 cv::Mat::data IS HWC-interleaved row-major u8, which
//                 matches TIM-VX innermost-first byte layout for shape
//                 `{3, src_W, src_H, 1}` — so we hand the Mat's data
//                 straight to the PPU input without any rearrange.
//   PPU graph:  PRE_PROCESS_RGB (single shader dispatch that fuses
//               bilinear resize + per-channel (x-mean)*scale + cast-
//               to-(S,Z) quantized u8) → Transpose (perm derived from
//               spec.shape: [3,2,1,0] for NCHW, [3,1,0,2] for NHWC)
//               which converts the op's channel-planar
//               `{dst_W, dst_H, 3, 1}` output into the spec.shape-
//               driven byte layout the NN input tensor expects.
//
// Why PRE_PROCESS_RGB and not the previous Resize+DataConvert chain
// ----------------------------------------------------------------
// PRE_PROCESS_RGB is one shader dispatch that does everything in fp16
// in registers; the alternative — Resize → DataConvert(u8→fp32) →
// Sub({1,1,3,1}) → Multiply({1,1,3,1}) → DataConvert(fp32→u8 q) →
// Permute — is N dispatches and fp32 traffic through DDR. More
// importantly, fp32 eltwise against per-channel constants trips a
// `vivante.nn.tensorcopy` driver bug on this chip (VIP9000):
// implicit broadcast, explicit Broadcast, AND pre-tiled full-shape
// constants all hit the same failure. BatchNorm trips `dim 65 exceeds
// 6`. Falling back to a CPU finish-step (the previous shipping
// state) cost ~5 ms/frame in the normalize+quant+layoutConvert loop.
// PRE_PROCESS_RGB sidesteps all of these — it's the op the vendor
// uses themselves, just exposed via a thin BuiltinOp subclass.
//
// Per-channel parameter mapping
// -----------------------------
// PRE_PROCESS_RGB computes, per channel c:
//   out[c] = (rgb_u8[c] - mean[c]) * scale[c] * (1/Q.s) + Q.zp
// For the ImageNet recipe `y = (x/255 - imagenet_mean) / imagenet_std`:
//   mean[c]  = 255 * imagenet_mean[c]
//   scale[c] = 1.0 / (255 * imagenet_std[c])
// `(Q.s, Q.zp)` is read off the OUTPUT tensor — we set it to
// `(spec.quant_scale, spec.quant_zp)`, the NN's input quant.
//
// Zero-copy
// ---------
//   ppu-* (copy)         : cvtColor writes into a temp Mat →
//                           CopyDataToTensor(PPU in) → PPU Run →
//                           CopyDataFromTensor(NN-ready bytes) →
//                           CopyDataToTensor(NN in).
//   ppu_zero_copy-* (zc) : cvtColor writes DIRECTLY into
//                           io_buf->pre_in_data → SwapHandle(PPU in)
//                           → PPU Run with PPU OUTPUT bound (via
//                           SwapHandle) to io_buf->in_data → NN reads
//                           io_buf->in_data without any memcpy. All
//                           three edges (CPU→PPU, PPU→NN, NN→CPU)
//                           are zerocopy.
//
// The source image dimensions (src_w, src_h) are fixed at construction.
// For a benchmark that feeds the same JPEG repeatedly, the runner
// probes the JPEG header once and passes the result in.
class PpuJpegPreProcessor : public PreProcessor {
 public:
  // `shader_mu` is the shader-unit serialization mutex — shared with
  // any other TIM-VX graph that dispatches on the same shader engine
  // (currently only this preprocessor; the previous
  // PpuSoftMaxTopKPostProcessor used to as well but has been removed
  // to match the vendor pipeline). Concurrent shader-graph dispatch
  // wedges the driver on this chip (see CLAUDE.md's "graph->Run
  // failed after 23222.64 ms" note); holding `*shader_mu` around any
  // shader-graph Run() / SwapHandle prevents that. If null, the
  // preprocessor allocates its own mutex (the convenience default
  // for single-shader-user setups).
  PpuJpegPreProcessor(const InputSpec& spec,
                       std::array<float, 3> mean,
                       std::array<float, 3> std,
                       uint32_t src_w, uint32_t src_h,
                       bool zerocopy = false,
                       std::shared_ptr<std::mutex> shader_mu = nullptr)
      : spec_(spec), mean_(mean), std_(std),
        src_w_(src_w), src_h_(src_h), zerocopy_(zerocopy),
        shader_mu_(shader_mu ? std::move(shader_mu)
                             : std::make_shared<std::mutex>()) {
    numel_ = 1;
    for (auto d : spec.shape) numel_ *= d;
    elem_ = bytesPerElem(spec.dtype);
    native_ = spec.timvx_native_layout;
    if (native_ && spec.shape.size() == 4) {
      // Folded transpose: spec.shape is TIM-VX WHCN {W, H, C, N}. That's
      // EXACTLY the channel-planar layout PRE_PROCESS_RGB produces, so
      // the PPU graph needs no trailing Transpose (see build_graph()).
      dst_w_ = spec.shape[0]; dst_h_ = spec.shape[1];
      layout_nhwc_ = false;  // unused on the native path
    } else if (spec.shape.size() == 4) {
      if (spec.shape[1] == 3) {
        layout_nhwc_ = false; dst_h_ = spec.shape[2]; dst_w_ = spec.shape[3];
      } else {
        layout_nhwc_ = true;  dst_h_ = spec.shape[1]; dst_w_ = spec.shape[2];
      }
    } else {
      layout_nhwc_ = true; dst_h_ = 224; dst_w_ = 224;
    }
    build_graph();
  }

  // Decode the JPEG via OpenCV (BGR→RGB) into `dst`. The resulting
  // bytes are HWC-interleaved row-major u8 — bit-identical to the
  // byte order TIM-VX's `{3, src_W, src_H, 1}` tensor (channel
  // innermost, then W, then H) reads. So `dst` is directly the PPU
  // input buffer. Returns false on decode or dimension mismatch.
  // Thread-safe — cv::imdecode / cv::cvtColor are reentrant.
  bool decode_rgb_into(const std::vector<uint8_t>& raw,
                        uint8_t* dst, size_t dst_size) {
    const size_t need = static_cast<size_t>(src_w_) * src_h_ * 3;
    if (dst_size < need) {
      std::fprintf(stderr,
                   "PpuJpegPreProcessor: dst_size=%zu < %zu\n",
                   dst_size, need);
      return false;
    }
    cv::Mat raw_mat(1, static_cast<int>(raw.size()), CV_8UC1,
                    const_cast<uint8_t*>(raw.data()));
    cv::Mat bgr = cv::imdecode(raw_mat, cv::IMREAD_COLOR);
    if (bgr.empty()) {
      std::fprintf(stderr, "PpuJpegPreProcessor: cv::imdecode failed\n");
      return false;
    }
    if (bgr.rows != static_cast<int>(src_h_) ||
        bgr.cols != static_cast<int>(src_w_)) {
      std::fprintf(stderr,
                   "PpuJpegPreProcessor: JPEG dims %dx%d != fixed src %ux%u\n",
                   bgr.cols, bgr.rows, src_w_, src_h_);
      return false;
    }
    // Pre-allocated cv::Mat over `dst` — cvtColor writes the RGB
    // bytes straight into the PPU input buffer. No intermediate
    // allocation. cvtColor is happy to write into a destination of
    // matching size/type; the resulting bytes are HWC-interleaved.
    cv::Mat rgb_view(static_cast<int>(src_h_), static_cast<int>(src_w_),
                     CV_8UC3, dst);
    cv::cvtColor(bgr, rgb_view, cv::COLOR_BGR2RGB);
    return true;
  }

  // Copy path (ppu-*): cvtColor into a temp buffer, push into PPU in,
  // run, pull NN-ready bytes back. No CPU normalize/quant — the
  // PRE_PROCESS_RGB op already produced bytes in the model's quant.
  std::vector<uint8_t> process(const std::vector<uint8_t>& raw) override {
    const size_t pre_bytes = static_cast<size_t>(src_w_) * src_h_ * 3;
    std::vector<uint8_t> pre_buf(pre_bytes);
    if (!decode_rgb_into(raw, pre_buf.data(), pre_buf.size())) return {};

    std::lock_guard<std::mutex> lk(*shader_mu_);
    if (!in_t_->CopyDataToTensor(
            pre_buf.data(), static_cast<uint32_t>(pre_buf.size()))) {
      std::fprintf(stderr, "PpuJpegPreProcessor: CopyDataToTensor failed\n");
      return {};
    }
    if (!graph_->Run()) {
      std::fprintf(stderr, "PpuJpegPreProcessor: graph->Run failed\n");
      return {};
    }
    std::vector<uint8_t> out(numel_ * elem_);
    if (!out_t_->CopyDataFromTensor(out.data())) {
      std::fprintf(stderr, "PpuJpegPreProcessor: CopyDataFromTensor failed\n");
      return {};
    }
    return out;
  }

  // Zero-copy path: cvtColor writes RGB bytes DIRECTLY into
  // io_buf->pre_in_data; PPU graph reads them and writes the model-
  // ready bytes DIRECTLY into io_buf->in_data (the NN's bound
  // buffer) via SwapHandle on the PPU OUTPUT tensor. No CPU touch
  // of the NN buffer at all.
  bool process_inplace(const std::vector<uint8_t>& raw,
                        IoBuffer* io_buf) override {
    if (!zerocopy_ || !io_buf || !io_buf->pre_in_data) {
      return PreProcessor::process_inplace(raw, io_buf);
    }
    if (!decode_rgb_into(raw, io_buf->pre_in_data, io_buf->pre_in_size))
      return false;
    if (io_buf->in_size < numel_ * elem_) {
      std::fprintf(stderr,
                   "PpuJpegPreProcessor: in_size=%zu < expected=%zu\n",
                   io_buf->in_size, numel_ * elem_);
      return false;
    }
    std::lock_guard<std::mutex> lk(*shader_mu_);
    if (!in_t_->SwapHandle(io_buf->pre_in_data,
                            /*is_new_ptr_malloc_by_ovxlib=*/false,
                            /*old_ptr_release=*/nullptr)) {
      std::fprintf(stderr, "PpuJpegPreProcessor: SwapHandle(in) failed\n");
      return false;
    }
    if (!out_t_->SwapHandle(io_buf->in_data,
                             /*is_new_ptr_malloc_by_ovxlib=*/false,
                             /*old_ptr_release=*/nullptr)) {
      std::fprintf(stderr, "PpuJpegPreProcessor: SwapHandle(out) failed\n");
      return false;
    }
    in_t_->FlushCacheForHandle();
    if (!graph_->Run()) {
      std::fprintf(stderr, "PpuJpegPreProcessor: graph->Run failed\n");
      return false;
    }
    // PPU wrote model-ready bytes into io_buf->in_data via DMA. The
    // base `sync_for_infer` override (no-op for PPU preprocessors)
    // skips the would-be CPU flush; the NN's Run will hit the same
    // memory the PPU just published.
    return true;
  }

  size_t expected_output_bytes() const override { return numel_ * elem_; }
  size_t expected_input_bytes() const override { return 0; /* JPEG: variable */ }
  size_t expected_pre_in_bytes() const override {
    if (!zerocopy_) return 0;
    return static_cast<size_t>(src_w_) * src_h_ * 3;
  }
  const char* name() const override { return "ppu-jpeg"; }

  // PPU graph already wrote into io_buf->in_data via DMA in
  // process_inplace; a CPU flush here would overwrite the PPU's
  // bytes with stale cache lines.
  void sync_for_infer(IoBuffer* /*io_buf*/,
                      const std::shared_ptr<tim::vx::Tensor>& /*nn_input*/)
      override {}

 private:
  void build_graph() {
    using namespace tim::vx;
    ctx_ = Context::Create();
    graph_ = ctx_->CreateGraph();

    // Input: u8 in `{3, src_W, src_H, 1}` (channel-innermost) with no
    // quant — PRE_PROCESS_RGB reads raw 0-255 byte values; the per-
    // channel mean is in 0-255 units (255 * imagenet_mean).
    TensorSpec in_spec(DataType::UINT8, {3, src_w_, src_h_, 1},
                       TensorAttribute::INPUT);
    in_t_ = zerocopy_
        ? graph_->CreateIOTensor(in_spec, /*data=*/nullptr)
        : graph_->CreateTensor(in_spec);

    // PRE_PROCESS_RGB output quant: the model's input (S, Z) so the
    // kernel's final cast `* (1/Q.s) + Q.zp` lands in NN-ready u8 bytes.
    Quantization out_quant(spec_.quant_scale != 0.0
                               ? QuantType::ASYMMETRIC
                               : QuantType::NONE,
                           static_cast<float>(
                               spec_.quant_scale != 0.0 ? spec_.quant_scale
                                                         : 1.0),
                           spec_.quant_zp);

    // PRE_PROCESS_RGB params. Mean/scale pre-multiplied for the
    // [0,255] input range (see file header for derivation).
    PreProcessRgbOp::Params pp{};
    pp.src_w = src_w_; pp.src_h = src_h_;
    pp.dst_w = dst_w_; pp.dst_h = dst_h_;
    pp.r_mean = 255.f * mean_[0];
    pp.g_mean = 255.f * mean_[1];
    pp.b_mean = 255.f * mean_[2];
    pp.r_scale = 1.f / (255.f * std_[0]);
    pp.g_scale = 1.f / (255.f * std_[1]);
    pp.b_scale = 1.f / (255.f * std_[2]);
    pp.reverse_channel = false;  // cv::cvtColor already gave us RGB.
    auto preproc_op = graph_->CreateOperation<PreProcessRgbOp>(pp);

    if (native_) {
      // Folded-transpose model: the NN input tensor IS WHCN
      // `{dst_W, dst_H, 3, 1}` (== spec_.shape), which is precisely the
      // channel-planar layout PRE_PROCESS_RGB produces. So the op's
      // output is the graph output directly — no trailing Transpose.
      // This is the device-side mirror of the host-side layout-convert
      // we dropped in the CPU preprocessor: both transposes are gone.
      out_t_ = zerocopy_
          ? graph_->CreateIOTensor(
                TensorSpec(DataType::UINT8, {dst_w_, dst_h_, 3, 1},
                           TensorAttribute::OUTPUT, out_quant),
                /*data=*/nullptr)
          : graph_->CreateTensor(
                TensorSpec(DataType::UINT8, {dst_w_, dst_h_, 3, 1},
                           TensorAttribute::OUTPUT, out_quant));
      (*preproc_op).BindInput(in_t_).BindOutput(out_t_);
      if (!graph_->Compile()) {
        std::fprintf(stderr, "PpuJpegPreProcessor: graph compile failed\n");
      }
      return;
    }

    // Legacy (non-folded) model: PRE_PROCESS_RGB writes a channel-planar
    // `{dst_W, dst_H, 3, 1}` intermediate; a Transpose then permutes it
    // into the NN input's spec.shape byte layout (the NN tensor carries
    // the MLIR-text shape — NCHW {1,3,224,224} or NHWC {1,224,224,3} —
    // which TIM-VX reads innermost-first).
    auto preproc_out = graph_->CreateTensor(TensorSpec(
        DataType::UINT8, {dst_w_, dst_h_, 3, 1},
        TensorAttribute::TRANSIENT, out_quant));
    (*preproc_op).BindInput(in_t_).BindOutput(preproc_out);

    std::vector<uint32_t> perm;
    if (layout_nhwc_) {
      // Output shape {1, H, W, 3}. Output dim i = input dim perm[i]:
      //   dim0=N(1) ← input dim3=N    perm[0]=3
      //   dim1=H    ← input dim1=H    perm[1]=1
      //   dim2=W    ← input dim0=W    perm[2]=0
      //   dim3=C(3) ← input dim2=C    perm[3]=2
      perm = {3, 1, 0, 2};
    } else {
      // Output shape {1, 3, H, W} (NCHW). perm = full reverse.
      perm = {3, 2, 1, 0};
    }
    out_t_ = zerocopy_
        ? graph_->CreateIOTensor(
              TensorSpec(DataType::UINT8, spec_.shape,
                         TensorAttribute::OUTPUT, out_quant),
              /*data=*/nullptr)
        : graph_->CreateTensor(
              TensorSpec(DataType::UINT8, spec_.shape,
                         TensorAttribute::OUTPUT, out_quant));
    auto tp_op = graph_->CreateOperation<ops::Transpose>(perm);
    (*tp_op).BindInput(preproc_out).BindOutput(out_t_);

    if (!graph_->Compile()) {
      std::fprintf(stderr, "PpuJpegPreProcessor: graph compile failed\n");
    }
  }

  InputSpec spec_;
  std::array<float, 3> mean_, std_;
  uint32_t src_w_, src_h_;
  uint32_t dst_h_, dst_w_;
  size_t numel_, elem_;
  bool layout_nhwc_;
  bool zerocopy_;
  bool native_;  // folded-transpose model: PPU output IS the NN input
  std::shared_ptr<tim::vx::Context> ctx_;
  std::shared_ptr<tim::vx::Graph> graph_;
  std::shared_ptr<tim::vx::Tensor> in_t_, out_t_;
  // Shader-unit serialization mutex (see constructor doc). Shared
  // with any future shader post-processor; today it's just this
  // preprocessor's own dispatch.
  std::shared_ptr<std::mutex> shader_mu_;
};

} // namespace timvx_pipeline

#endif // TIMVX_PIPELINE_PREPROC_H
