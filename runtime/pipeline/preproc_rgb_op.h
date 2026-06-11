// Declaration: AI involved in this file by proofreading the TIM-VX
// source code and providing suggestions on the custom op implementation.
// pipeline/preproc_rgb_op.h — TIM-VX C++ wrapper around OVXLIB's
// `VSI_NN_OP_PRE_PROCESS_RGB` op (a single shader-dispatch primitive
// that fuses resize + per-channel normalize + quantize into one node).
//
// Why this exists
// ---------------
// TIM-VX's public C++ surface (`tim::vx::ops::*`) doesn't expose
// PRE_PROCESS_RGB; this wrapper reaches one layer down to the C
// OVXLIB primitive that all the C++ ops are themselves built on.
// `BuiltinOp(graph, kind, in, out)` is the public escape hatch — it
// calls `vsi_nn_AddNode(graph, kind, ...)`, which both registers the
// op and runs its `op_init` (so `nn_param.pre_process_rgb.local` is
// allocated for us). After construction we set the op's per-channel
// mean / scale, crop rect, and output shape via the union-typed
// `node->nn_param.pre_process_rgb.*` accessor — exactly the same
// pattern OVXLIB's internal `vsi_nn_op_pre_process.c` uses when it
// fans `VSI_NN_SOURCE_FORMAT_IMAGE_RGB` out into a real node.
//
// What PRE_PROCESS_RGB does
// -------------------------
// Single shader-unit dispatch that fuses, per channel c ∈ {R,G,B}:
//
//   out[c] = (rgb_u8[c] - mean[c]) * scale[c] * (1/Q.s) + Q.zp
//
// where `(Q.s, Q.zp)` is the OUTPUT tensor's TIM-VX quantization (we
// set it to the NN's input quant). For the standard ImageNet recipe
//
//   y = (x/255 - imagenet_mean[c]) / imagenet_std[c]
//
// the mapping is
//
//   mean[c]  = 255 * imagenet_mean[c]
//   scale[c] = 1.0 / (255 * imagenet_std[c])
//
// Input expected
// --------------
// Raw `{3, src_W, src_H, 1}` U8 bytes in TIM-VX innermost-first
// order (channel-innermost, then W, then H) — bit-identical to what
// `cv::cvtColor(BGR→RGB)` writes into its `cv::Mat::data` (HWC
// interleaved row-major). No quantization on the input.
//
// Output produced
// ---------------
// `{dst_W, dst_H, 3, 1}` channel-PLANAR U8 in TIM-VX innermost-first
// order (W innermost, then H, then C). The OUTPUT tensor's caller-
// supplied (S, Z) drives the final cast-to-byte. For our use case
// it's the NN's input quant.
//
// Caveat — output layout vs the NN input
// --------------------------------------
// The NN input tensor is created with the MLIR-text spec.shape
// (e.g. `{1, 3, 224, 224}` for NCHW), which TIM-VX reads innermost-
// first as `[N=inner, C, H, W=outer]` (the lowering pass adds an
// internal Transpose right after to flip into the WHCN-canonical
// view the rest of the graph wants). PRE_PROCESS_RGB writes WHCN-
// canonical channel-planar, which is NOT the same byte layout. So
// the PPU pipeline appends a `tim::vx::ops::Transpose` after this
// op to permute into the spec.shape layout — the NN's binding then
// reads the bytes as if they were the standard input format.

#ifndef TIMVX_PIPELINE_PREPROC_RGB_OP_H
#define TIMVX_PIPELINE_PREPROC_RGB_OP_H

#include <array>
#include <cstdint>
#include <memory>

#include "tim/vx/builtin_op.h"
#include "tim/vx/graph.h"

// TIM-VX internal headers: `op_impl.h` for `OpImpl::node()` (BuiltinOp's
// `impl_` is an OpImpl*); the OVXLIB chain for the op enum
// (`VSI_NN_OP_PRE_PROCESS_RGB`) and the `nn_param.pre_process_rgb`
// union field type. Both live in the TIM-VX source tree (not in the
// public install); the build adds them via `-I` to the compile flags
// (see `lower_to_timvx.py`).
#include "op_impl.h"  // tim::vx::OpImpl::node()
extern "C" {
#include "vsi_nn_pub.h"  // VSI_NN_OP_PRE_PROCESS_RGB + nn_param union
}

namespace timvx_pipeline {

class PreProcessRgbOp : public tim::vx::BuiltinOp {
 public:
  // Caller-facing params. `dst_w/dst_h` set the output spatial dims
  // (PRE_PROCESS_RGB resizes on the fly via `scale_x/scale_y`). Per-
  // channel mean/scale must already be PRE-multiplied for the [0,255]
  // input range — i.e. `mean = 255 * imagenet_mean`, `scale = 1 / (255
  // * imagenet_std)`. The output tensor's TIM-VX quant supplies the
  // final cast-to-u8 (Q.s, Q.zp). See file header for the full formula.
  struct Params {
    uint32_t src_w, src_h;
    uint32_t dst_w, dst_h;
    float r_mean = 0.f, g_mean = 0.f, b_mean = 0.f;
    float r_scale = 1.f, g_scale = 1.f, b_scale = 1.f;
    // RGB→BGR channel swap (vendor's class_pre.cpp does this if the
    // model was trained on BGR but we feed RGB). We feed RGB through
    // cv::cvtColor(BGR2RGB), so default false matches our pipeline.
    bool reverse_channel = false;
    // Crop rect in the SOURCE image (left, top, width, height). We
    // resize the full source by default; left/top stay 0 and
    // width/height are src_w/src_h.
    uint32_t rect_left = 0, rect_top = 0;
  };

  PreProcessRgbOp(tim::vx::Graph* graph, const Params& params)
      : tim::vx::BuiltinOp(graph,
                            VSI_NN_OP_PRE_PROCESS_RGB,
                            /*in_cnt=*/1, /*out_cnt=*/1,
                            tim::vx::DataLayout::ANY),
        params_(params) {
    // `output_attr.size` is a `vsi_size_t*` the OVXLIB op reads at
    // setup-time. It MUST stay alive past graph->Compile(). Hold it
    // in a member so the lifetime tracks the op object.
    output_size_storage_[0] = params_.dst_w;
    output_size_storage_[1] = params_.dst_h;
    output_size_storage_[2] = 3;
    output_size_storage_[3] = 1;

    vsi_nn_node_t* node = impl_->node();
    auto& p = node->nn_param.pre_process_rgb;
    p.rect.left = params_.rect_left;
    p.rect.top = params_.rect_top;
    p.rect.width = params_.src_w;
    p.rect.height = params_.src_h;
    p.output_attr.size = output_size_storage_.data();
    p.output_attr.dim_num = 4;
    p.r_mean = params_.r_mean;
    p.g_mean = params_.g_mean;
    p.b_mean = params_.b_mean;
    p.r_scale = params_.r_scale;
    p.g_scale = params_.g_scale;
    p.b_scale = params_.b_scale;
    p.reverse_channel = params_.reverse_channel ? TRUE : FALSE;
    // OVXLIB exposes `perm` for an in-op output permute, but we don't
    // need it: we follow with an explicit `tim::vx::ops::Transpose`
    // because the spec.shape-driven NN input layout isn't one of the
    // simple permutations PRE_PROCESS_RGB supports natively. Leave
    // null + dim_num=0 so the kernel's `enable_perm` check stays FALSE.
    p.perm = nullptr;
    p.dim_num = 0;
  }

  std::shared_ptr<tim::vx::Operation> Clone(
      std::shared_ptr<tim::vx::Graph>& graph) const override {
    return graph->CreateOperation<PreProcessRgbOp>(params_);
  }

 private:
  Params params_;
  // `vsi_size_t` is `size_t` on Linux aarch64; we hold our own array
  // and hand a stable pointer to OVXLIB. std::array so we get default
  // value-init and a clean .data().
  std::array<vsi_size_t, 4> output_size_storage_{};
};

}  // namespace timvx_pipeline

#endif  // TIMVX_PIPELINE_PREPROC_RGB_OP_H
