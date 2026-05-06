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
// (activations) / WHIcOc (kernels) form -- the IR-side `--tosa-layout-to-whcn`
// pass does the permutation. We pass through TIM-VX's default DataLayout
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
#include "tim/vx/ops/simple_operations.h" // Rcp, DataConvert
#include "tim/vx/ops/slice.h"
#include "tim/vx/ops/transpose.h"

// CustomGemm lives in TIM-VX's samples tree, not the public install. Build
// must add `-I$TIM_VX_DIR/samples/custom_op_test` and link
// `$TIM_VX_DIR/samples/custom_op_test/custom_gemm.cc` (it carries the
// static kernel_id_/kernel_name_ definitions). Driven from lower_sample.py.
#include "custom_gemm.h"

namespace timvx_runtime {

using TensorPtr = std::shared_ptr<tim::vx::Tensor>;
using GraphPtr  = std::shared_ptr<tim::vx::Graph>;

// const_tensor: backing data is supplied by the lowered `static const T[]`
// arrays the timvx-to-emitc pass emits ahead of each call site.
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

// ReduceSum: tim::vx's expects keep_dims explicitly; keep_dims=true keeps
// the reduced axis as a singleton dim (matches tosa.reduce_sum semantics).
inline TensorPtr reduce_sum(GraphPtr graph, TensorPtr input,
                             std::vector<int32_t> axes, bool keep_dims,
                             tim::vx::TensorSpec output_spec) {
  auto out = graph->CreateTensor(output_spec);
  auto op = graph->CreateOperation<tim::vx::ops::ReduceSum>(axes, keep_dims);
  (*op).BindInput(input).BindOutput(out);
  return out;
}

// DataConvert: dtype change with scale/zp ignored — i.e., the integer
// values are reinterpreted in the new dtype. Used to lower `tosa.cast`
// (e.g., i8 → f32 dequantize cast or f32 → i32 → i8 quantize cast); the
// surrounding tosa.sub / tosa.mul ops do the actual dequantization with
// explicit scale/zp constants in fp32.
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
