// timvx_runtime.h — thin one-shot wrappers around tim::vx ops.
//
// Each helper does the same four-step dance:
//   1) materialize the output tensor from `output_spec`,
//   2) construct the corresponding tim::vx op,
//   3) wire inputs / outputs,
//   4) return the output tensor.
//
// This file exists because EmitC's `call_opaque` is a one-call-one-result
// expression and can't represent that 4-statement sequence directly. Each
// timvx.* op in the lowering emits a single `call_opaque` against one of
// these helpers.
//
// Layouts: tensors arrive already in TIM-VX's native innermost-first WHCN
// (activations) / WHIcOc (kernels) form -- the spatial-op converters in
// `--tosa-to-timvx` (`Conv2DOpConversion`, `MaxPool2DConversion`,
// `AvgPool2DConversion`, `RescaleConvFusion`) wrap their TIM-VX op in
// explicit `timvx.transpose` ops that bridge NHWC <-> WHCN, and the
// transpose canonicalizers fold those into permuted constants / single
// composed transposes. We pass through TIM-VX's default DataLayout
// arguments rather than declaring CWHN/IcWHOc.

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"
#include "tim/vx/types.h"
#include "tim/vx/ops/clip.h"
#include "tim/vx/ops/conv2d.h"
#include "tim/vx/ops/elementwise.h"      // Add, Sub, Pow, Multiply
#include "tim/vx/ops/fullyconnected.h"
#include "tim/vx/ops/pad.h"
#include "tim/vx/ops/pool2d.h"
#include "tim/vx/ops/reduce.h"           // ReduceSum
#include "tim/vx/ops/reshape.h"
#include "tim/vx/ops/simple_operations.h" // Rcp, Cast
#include "tim/vx/ops/slice.h"
#include "tim/vx/ops/transpose.h"

// Project-owned custom OpenCL ops. Both live under example/custom_ops/
// in this repo (TIM-VX's customized_op.md explicitly supports hosting
// custom ops outside the TIM-VX tree). The build step adds that dir to
// the include path and links the matching .cc files (which carry the
// static kernel_id_/kernel_name_ definitions).
#include "custom_ops/custom_gemm.h"
#include "custom_ops/custom_reduce_sum.h"

namespace timvx_runtime {

using TensorPtr = std::shared_ptr<tim::vx::Tensor>;
using GraphPtr  = std::shared_ptr<tim::vx::Graph>;

// const_tensor: backing data is supplied by the lowered `static const T[]`
// arrays the timvx-to-emitc pass emits ahead of each call site. The arrays
// are pre-baked in TIM-VX innermost-first byte order at lowering time
// (see reorderMlirToTvx in TimvxToEmitC.cpp), so we bind them directly
// without any runtime layout fixup.
inline TensorPtr const_tensor(GraphPtr graph,
                              tim::vx::TensorSpec spec,
                              const void* data) {
  return graph->CreateTensor(spec, data);
}

inline TensorPtr clip(GraphPtr graph, TensorPtr input,
                      float min_val, float max_val,
                      tim::vx::TensorSpec output_spec) {
  auto out = graph->CreateTensor(output_spec);
  auto op  = graph->CreateOperation<tim::vx::ops::Clip>(min_val, max_val);
  (*op).BindInput(input).BindOutput(out);
  return out;
}

inline TensorPtr conv2d(GraphPtr graph,
                        TensorPtr input, TensorPtr weight, TensorPtr bias,
                        std::array<uint32_t, 4> pad,
                        std::array<uint32_t, 2> stride,
                        std::array<uint32_t, 2> dilation,
                        tim::vx::TensorSpec output_spec) {
  auto out = graph->CreateTensor(output_spec);
  auto op  = graph->CreateOperation<tim::vx::ops::Conv2d>(
      pad, stride, dilation, /*multiplier=*/0);
  (*op).BindInputs({input, weight, bias}).BindOutputs({out});
  return out;
}

inline TensorPtr pool2d(GraphPtr graph, TensorPtr input,
                        tim::vx::PoolType type,
                        std::array<uint32_t, 2> kernel,
                        std::array<uint32_t, 2> stride,
                        std::array<uint32_t, 4> pad,
                        tim::vx::TensorSpec output_spec) {
  auto out = graph->CreateTensor(output_spec);
  auto op  = graph->CreateOperation<tim::vx::ops::Pool2d>(
      type, pad, kernel, stride, tim::vx::RoundType::FLOOR);
  (*op).BindInput(input).BindOutput(out);
  return out;
}


// Fully-connected: Y = X @ W^T + b. Input is rank-2 [batch, in_features],
// weight rank-2 [out_features, in_features], bias rank-1 [out_features].
// axis=1 means dim 1 onward of the input is treated as features (input is
// already coerced to 2-D by the matmul→FC rewrite in TIMVXPasses.cpp).
// `weights` (out_features) is required: TIM-VX's FCL2 op_setup writes
// `output.size[0] = nn_param.fcl.weights`, so leaving it 0 (the default
// ctor's value) collapses the output to a 0-element tensor.
inline TensorPtr fully_connected(GraphPtr graph,
                                 TensorPtr input, TensorPtr weight, TensorPtr bias,
                                 tim::vx::TensorSpec output_spec) {
  // The matmul→FC rewrite emits an output type of `[Ba*M, N]` where N is
  // the out_features count (trailing dim).
  uint32_t out_features = output_spec.shape_.empty()
                              ? 0u
                              : static_cast<uint32_t>(output_spec.shape_.back());
  auto out = graph->CreateTensor(output_spec);
  auto op  = graph->CreateOperation<tim::vx::ops::FullyConnected>(
      /*axis=*/1u, out_features);
  (*op).BindInputs({input, weight, bias}).BindOutput(out);
  return out;
}

// Runtime-runtime matmul (activation × activation). Const-weight matmul
// is rewritten to fully_connected upstream, so this path only fires for
// cases like attention's Q @ K^T.
//
// Implemented via tim::vx::ops::CustomGemm rather than tim::vx::ops::Matmul:
// the NN-core Matmul was unreliable in our smoke tests; CustomGemm runs an
// OpenCL GEMM kernel on the GC GPU, which is FP32-native and works pre-
// quantization. Caveats inherited from the sample kernel:
//   - rank-2 only (the kernel reads via image2d_t with 2-D coords),
//   - FP32 only (the kernel is gemm_F32F32toF32_2D),
//   - quant params left at identity; not yet a quantized path.
//
// tim::vx shapes are innermost-first: logical [M, K] is stored as {K, M}.
inline TensorPtr matmul(GraphPtr graph, TensorPtr a, TensorPtr b,
                        tim::vx::TensorSpec output_spec) {
  auto a_shape = a->GetShape();
  auto b_shape = b->GetShape();
  int K = static_cast<int>(a_shape[0]);
  int M = static_cast<int>(a_shape[1]);
  int N = static_cast<int>(b_shape[0]);

  tim::vx::ops::CustomGemm::ParamTuple params(
      M, K, N,
      /*ac2zero=*/0, /*bc2zero=*/0,
      /*scale_a=*/1.0f,   /*zp_a=*/0.0f,
      /*scale_b=*/1.0f,   /*zp_b=*/0.0f,
      /*scale_out=*/1.0f, /*zp_out=*/0.0f);

  auto out = graph->CreateTensor(output_spec);
  auto op  = graph->CreateOperation<tim::vx::ops::CustomGemm>(
      /*trans_a=*/false, /*trans_b=*/false, params);
  (*op).BindInputs({a, b}).BindOutputs({out});
  return out;
}

// Binary elementwise helpers -------------------------------------------------
template <typename Op>
inline TensorPtr binary_elt(GraphPtr graph, TensorPtr a, TensorPtr b,
                            tim::vx::TensorSpec output_spec) {
  auto out = graph->CreateTensor(output_spec);
  auto op  = graph->CreateOperation<Op>();
  (*op).BindInputs({a, b}).BindOutput(out);
  return out;
}
inline TensorPtr add(GraphPtr g, TensorPtr a, TensorPtr b, tim::vx::TensorSpec s) {
  return binary_elt<tim::vx::ops::Add>(g, a, b, s);
}
inline TensorPtr sub(GraphPtr g, TensorPtr a, TensorPtr b, tim::vx::TensorSpec s) {
  return binary_elt<tim::vx::ops::Sub>(g, a, b, s);
}
inline TensorPtr pow_op(GraphPtr g, TensorPtr a, TensorPtr b, tim::vx::TensorSpec s) {
  return binary_elt<tim::vx::ops::Pow>(g, a, b, s);
}
inline TensorPtr multiply(GraphPtr g, TensorPtr a, TensorPtr b, tim::vx::TensorSpec s) {
  // Multiply has an extra `scale` parameter; default 1.0f.
  auto out = g->CreateTensor(s);
  auto op  = g->CreateOperation<tim::vx::ops::Multiply>(1.0f);
  (*op).BindInputs({a, b}).BindOutput(out);
  return out;
}
inline TensorPtr maximum(GraphPtr g, TensorPtr a, TensorPtr b,
                         tim::vx::TensorSpec s) {
  return binary_elt<tim::vx::ops::Maximum>(g, a, b, s);
}
inline TensorPtr minimum(GraphPtr g, TensorPtr a, TensorPtr b,
                         tim::vx::TensorSpec s) {
  return binary_elt<tim::vx::ops::Minimum>(g, a, b, s);
}

// Unary elementwise helpers --------------------------------------------------
template <typename Op>
inline TensorPtr unary_elt(GraphPtr graph, TensorPtr input,
                           tim::vx::TensorSpec output_spec) {
  auto out = graph->CreateTensor(output_spec);
  auto op  = graph->CreateOperation<Op>();
  (*op).BindInput(input).BindOutput(out);
  return out;
}
inline TensorPtr rcp(GraphPtr g, TensorPtr x, tim::vx::TensorSpec s) {
  return unary_elt<tim::vx::ops::Rcp>(g, x, s);
}

// Shape / data movement ------------------------------------------------------
inline TensorPtr reshape(GraphPtr graph, TensorPtr input,
                         std::vector<uint32_t> shape,
                         tim::vx::TensorSpec output_spec) {
  auto out = graph->CreateTensor(output_spec);
  auto op  = graph->CreateOperation<tim::vx::ops::Reshape>(shape);
  (*op).BindInput(input).BindOutput(out);
  return out;
}

inline TensorPtr slice(GraphPtr graph, TensorPtr input,
                       std::vector<uint32_t> start,
                       std::vector<uint32_t> length,
                       tim::vx::TensorSpec output_spec) {
  // tim::vx::ops::Slice uses int32 start/length.
  std::vector<int32_t> s(start.begin(), start.end());
  std::vector<int32_t> l(length.begin(), length.end());
  auto out = graph->CreateTensor(output_spec);
  auto op  = graph->CreateOperation<tim::vx::ops::Slice>(
      static_cast<uint32_t>(s.size()), s, l);
  (*op).BindInput(input).BindOutput(out);
  return out;
}

// ReduceSum: tim::vx expects keep_dims explicitly; keep_dims=true keeps
// the reduced axis as a singleton dim (matches tosa.reduce_sum semantics).
//
// FP32 fast path: VIP9000Nano-DI's NN-engine REDUCE only ships INT8/INT32
// kernels — fp32/fp16/u8 fail at Compile() (op_probe table). For fp32
// single-axis keep_dims reductions we route through CustomReduceSum's
// OpenCL kernel on the GC GPU. The custom path handles any inner stride
// (in-memory innermost OR a non-innermost axis like W of an NHWC tensor
// where C=10 sits inside) as long as `inner * axis_size` fits within the
// chip's image2d width limit. Anything outside the envelope (multi-axis,
// non-fp32, keep_dims=false, or projection too wide) falls through to
// tim::vx::ops::ReduceSum.
inline TensorPtr reduce_sum(GraphPtr graph, TensorPtr input,
                             std::vector<int32_t> axes, bool keep_dims,
                             tim::vx::TensorSpec output_spec) {
  bool fp32_single_axis_keep_dims =
      input->GetDataType() == tim::vx::DataType::FLOAT32 &&
      axes.size() == 1 && keep_dims;

  uint32_t inner = 1;
  uint32_t axis_size = 1;
  uint32_t outer = 1;
  if (fp32_single_axis_keep_dims) {
    // Bytes inside TIM-VX are in innermost-first order (the harness
    // layoutConverts at the function boundary). For shape {s_0,…,s_{R-1}},
    // dim k has byte stride = product(s[0..k-1]); dims to the LEFT of K
    // are inner and dims to the RIGHT are outer — the OPPOSITE of MLIR
    // row-major. Compute strides in the actual TIM-VX layout the kernel
    // will walk.
    const auto& s = input->GetShape();
    int K = axes[0];
    for (int i = 0; i < K; ++i) inner *= s[i];
    axis_size = s[K];
    for (size_t i = static_cast<size_t>(K) + 1; i < s.size(); ++i)
      outer *= s[i];
  }

  // image2d on this chip's GC GPU caps each dim at 8192 for FP32.
  // Layout A (axis-along-width): input image {inner*axis_size, outer},
  //   limited by inner*axis_size <= W_max.
  // Layout B (axis-along-height): input image {inner, axis_size*outer},
  //   limited by axis_size*outer <= H_max (and inner <= W_max, but
  //   inner is typically much smaller than the W_max product).
  // Pick the layout that fits; fall through if neither does (the
  // builtin REDUCE will likely COMPILE_FAIL on fp32, but that's the
  // honest signal — better than silently producing wrong values).
  constexpr uint32_t kMaxImageDim = 8192;
  uint64_t inner_x_axis = static_cast<uint64_t>(inner) * axis_size;
  uint64_t axis_x_outer = static_cast<uint64_t>(axis_size) * outer;
  bool layoutA_fits = fp32_single_axis_keep_dims &&
                      inner_x_axis <= kMaxImageDim && outer <= kMaxImageDim;
  bool layoutB_fits = fp32_single_axis_keep_dims &&
                      inner <= kMaxImageDim && axis_x_outer <= kMaxImageDim;

  if (!layoutA_fits && !layoutB_fits) {
    auto out = graph->CreateTensor(output_spec);
    auto op = graph->CreateOperation<tim::vx::ops::ReduceSum>(axes, keep_dims);
    (*op).BindInput(input).BindOutput(out);
    return out;
  }

  // Output is the same {inner, outer} 2-D shape under either layout —
  // both kernels write at (gx=inner_idx, gy=outer_idx).
  tim::vx::TensorSpec out2d_spec(tim::vx::DataType::FLOAT32,
                                 tim::vx::ShapeType{inner, outer},
                                 tim::vx::TensorAttribute::TRANSIENT);
  auto out2d = graph->CreateTensor(out2d_spec);

  if (layoutA_fits) {
    // Reshape input to {inner * axis_size, outer}; kernel reads at
    // (gx + k*inner, gy). Byte at TIM-VX (i0, i1) = i0 + i1 *
    // (inner*axis_size); for (i0=inner_idx + axis_idx*inner, i1=outer_idx)
    // this lands on inner_idx + axis_idx*inner + outer_idx*inner*axis_size,
    // matching TIM-VX innermost-first byte order.
    const uint32_t in2d_w = inner * axis_size;
    tim::vx::TensorSpec in2d_spec(tim::vx::DataType::FLOAT32,
                                  tim::vx::ShapeType{in2d_w, outer},
                                  tim::vx::TensorAttribute::TRANSIENT);
    auto in2d = graph->CreateTensor(in2d_spec);
    auto reshape_in = graph->CreateOperation<tim::vx::ops::Reshape>(
        std::vector<uint32_t>{in2d_w, outer});
    (*reshape_in).BindInput(input).BindOutput(in2d);

    auto op = graph->CreateOperation<tim::vx::ops::CustomReduceSum>(
        static_cast<int>(axis_size), static_cast<int>(inner),
        static_cast<int>(outer));
    (*op).BindInput(in2d).BindOutput(out2d);
  } else {
    // Reshape input to {inner, axis_size * outer}; kernel reads at
    // (gx, gy*axis_size + k). Byte at TIM-VX (i0, i1) = i0 + i1*inner;
    // for (i0=inner_idx, i1=axis_idx + outer_idx*axis_size) this lands
    // on inner_idx + axis_idx*inner + outer_idx*axis_size*inner — same
    // bytes as layout A, just reshaped tall.
    const uint32_t in2d_h = axis_size * outer;
    tim::vx::TensorSpec in2d_spec(tim::vx::DataType::FLOAT32,
                                  tim::vx::ShapeType{inner, in2d_h},
                                  tim::vx::TensorAttribute::TRANSIENT);
    auto in2d = graph->CreateTensor(in2d_spec);
    auto reshape_in = graph->CreateOperation<tim::vx::ops::Reshape>(
        std::vector<uint32_t>{inner, in2d_h});
    (*reshape_in).BindInput(input).BindOutput(in2d);

    auto op = graph->CreateOperation<tim::vx::ops::CustomReduceSumTall>(
        static_cast<int>(axis_size), static_cast<int>(inner),
        static_cast<int>(outer));
    (*op).BindInput(in2d).BindOutput(out2d);
  }

  // Reshape back to the keep_dims output shape requested by the caller.
  // Bytes don't move; harness reads them MLIR row-major and lands on
  // sum_over_axis at MLIR (n, ..., c) for the matching coord→byte map.
  auto out = graph->CreateTensor(output_spec);
  auto reshape_out = graph->CreateOperation<tim::vx::ops::Reshape>(
      std::vector<uint32_t>(output_spec.shape_.begin(),
                             output_spec.shape_.end()));
  (*reshape_out).BindInput(out2d).BindOutput(out);
  return out;
}

// Cast: value-preserving dtype convert. Routes through the GPU `cast`
// kernel (tim::vx::ops::Cast) which is what every `tosa.cast` lowers
// to — covers f32 ↔ {i8|asym, u8|asym, i32 raw} on this chip. The
// output tensor's quant metadata (scale,zp on output_spec) is honored
// by downstream ops but not by the cast itself.
inline TensorPtr cast(GraphPtr graph, TensorPtr input,
                      tim::vx::TensorSpec output_spec) {
  auto out = graph->CreateTensor(output_spec);
  auto op = graph->CreateOperation<tim::vx::ops::Cast>();
  (*op).BindInput(input).BindOutput(out);
  return out;
}

// DataConvert: int↔int requantize between (Si,Zi) on the input spec and
// (So,Zo) on output_spec. Routes through `tim::vx::ops::DataConvert`
// (internally vxTensorCopyNode). Used to lower standalone `tosa.rescale`
// (e.g. the conv→rescale→pad→rescale→pool pattern tflite emits, where the
// second rescale aligns scales). On VIP9000Nano-DI: int↔int and int→f32
// PASS, f32→int FAILs — keep f32 boundaries on `cast` instead.
inline TensorPtr dataconvert(GraphPtr graph, TensorPtr input,
                             tim::vx::TensorSpec output_spec) {
  auto out = graph->CreateTensor(output_spec);
  auto op = graph->CreateOperation<tim::vx::ops::DataConvert>();
  (*op).BindInput(input).BindOutput(out);
  return out;
}

// Pad: front/back size derive from a [rank, 2] index tensor that was
// produced by a `tosa.const_shape` and rebuilt by the lowering. Quantized
// TOSA passes the input zero-point as `pad_const` so the padded region
// reads as zero in real-value space.
inline TensorPtr pad(GraphPtr graph, TensorPtr input,
                     std::vector<uint32_t> padding /* [rank*2], laid out as
                     interleaved [front, back] per dim */,
                     float pad_const, tim::vx::TensorSpec output_spec) {
  size_t rank = padding.size() / 2;
  std::vector<uint32_t> front(rank), back(rank);
  for (size_t i = 0; i < rank; ++i) {
    front[i] = padding[i * 2 + 0];
    back[i]  = padding[i * 2 + 1];
  }
  // tim::vx::ops::Pad takes an i32 const value. Cast pad_const (the input
  // zp) — TIM-VX will interpret it as the integer fill in storage space.
  int32_t const_val = static_cast<int32_t>(pad_const);
  auto out = graph->CreateTensor(output_spec);
  auto op = graph->CreateOperation<tim::vx::ops::Pad>(front, back, const_val);
  (*op).BindInput(input).BindOutput(out);
  return out;
}

inline TensorPtr transpose(GraphPtr graph, TensorPtr input,
                           std::vector<uint32_t> perms,
                           tim::vx::TensorSpec output_spec) {
  auto out = graph->CreateTensor(output_spec);
  auto op  = graph->CreateOperation<tim::vx::ops::Transpose>(perms);
  (*op).BindInput(input).BindOutput(out);
  return out;
}

}  // namespace timvx_runtime
