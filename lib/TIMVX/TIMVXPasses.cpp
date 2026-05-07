//===- TIMVXPasses.cpp - TIMVX passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// tosa-to-timvx pass.
//
// Mapping table (TOSA op  →  TIM-VX op  →  tim::vx C++ class):
//
//   tosa.clamp → timvx.clip → tim::vx::ops::Clip
//   tosa.const → timvx.const → tensor w/ TensorAttribute::CONSTANT
//   tosa.const_shape → timvx.const_shape → (compile-time shape)
//   tosa.conv2d → timvx.conv2d → tim::vx::ops::Conv2d
//   tosa.matmul → timvx.fully_connected → tim::vx::ops::FullyConnected
//                  (when B is a tosa.const; weight is transposed at
//                   compile time, zero bias is synthesized)
//   tosa.matmul → timvx.matmul → tim::vx::ops::Matmul
//                  (fallback when B is a runtime activation)
//   tosa.max_pool2d → timvx.pool2d → tim::vx::ops::Pool2d (PoolType::MAX)
//   tosa.avg_pool2d → timvx.pool2d → tim::vx::ops::Pool2d (PoolType::AVG)
//   tosa.mul → timvx.multiply → tim::vx::ops::Multiply
//   tosa.pow → timvx.pow → tim::vx::ops::Pow
//   tosa.reciprocal → timvx.rcp → tim::vx::ops::Rcp
//   tosa.reshape → timvx.reshape → tim::vx::ops::Reshape
//   tosa.slice → timvx.slice → tim::vx::ops::Slice
//   tosa.sub → timvx.sub → tim::vx::ops::Sub
//   tosa.add → timvx.add → tim::vx::ops::Add
//   tosa.transpose → timvx.transpose → tim::vx::ops::Transpose
//
//===----------------------------------------------------------------------===//

#include "TIMVX/TIMVXPasses.h"

#include "TIMVX/TIMVXDialect.h"
#include "TIMVX/TIMVXOps.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>

namespace mlir {
namespace timvx {

#define GEN_PASS_DEF_TOSALAYOUTTAGPASS
#define GEN_PASS_DEF_TOSALAYOUTTOWHCNPASS
#define GEN_PASS_DEF_TOSACONSTFOLDPASS
#define GEN_PASS_DEF_TOSAFOLDAVGPOOLREDUCEPASS
#define GEN_PASS_DEF_TOSATOTIMVXPASS
#define GEN_PASS_DEF_TIMVXCONV1X1TOFCPASS
#define GEN_PASS_DEF_TIMVXTOEMITCPASS
#include "TIMVX/TIMVXPasses.h.inc"

namespace {

// Helper for finding zero-valued tensors
static bool isConstantZero(Value v) {
  return matchPattern(v, m_Zero()) || matchPattern(v, m_AnyZeroFloat());
}

// Read a scalar / 1-element integer from a `tosa.const` (or anything that
// folds to a `DenseIntElementsAttr`). Returns nullopt if `v` doesn't trace
// back to such a constant.
//
// Used to extract the scalar zp / multiplier / shift operands that quantized
// TOSA carries on conv2d / rescale / pad in operand form.
static std::optional<int64_t> matchConstScalarInt(Value v) {
  ElementsAttr attr;
  if (!matchPattern(v, m_Constant(&attr)))
    return std::nullopt;
  auto dense = dyn_cast<DenseIntElementsAttr>(attr);
  if (!dense || dense.getNumElements() < 1)
    return std::nullopt;
  return (*dense.getValues<APInt>().begin()).getSExtValue();
}

//===----------------------------------------------------------------------===//
// tosa-const-fold (driver below)
//===----------------------------------------------------------------------===//
//
// Pure tosa→tosa rewrites that fold elementwise / reshape ops on constant
// tensors. Run via the greedy driver to fixed point, so chains collapse
// inside-out (e.g. BatchNorm's `var → +eps → ^0.5 → 1/x → *gamma`).
//
// Why this pass exists:  TIM-VX's eltwise kernels reject same-rank,
// different-size broadcast (`tensor<1xf32> + tensor<64xf32>`) even
// though the per-dim compatibility check in vsi_nn_op_eltwise_setup
// would let it through. The BatchNorm scalar chain hits exactly that
// pattern, and every op in the chain has constant operands — so the
// cleanest fix is to fold them away at the tosa level before lowering.
//
// Why it's separate from --tosa-to-timvx: the conversion driver is
// one-shot per op; an upstream fold won't trigger re-matching of its
// consumers. The greedy driver iterates to fixed point.
//
// Why it's separate from upstream's --tosa-layerwise-constant-fold:
// upstream covers `transpose`, `reciprocal`, and `reduce*`, but not
// the binary elementwise ops or `reshape`.
//
// FP32 only (matches the immediate need).

// Materialized view of an FP32 constant tensor: the shape from its type,
// plus an owned `float[]` of values. We materialize a copy here (rather
// than refer back to the underlying attribute) so the same code path
// works for both inline `dense<...>` and out-of-line `dense_resource<...>`
// storage. Kept FP32-only — that's all the BatchNorm chain needs.
struct ConstF32View {
  ArrayRef<int64_t> shape;
  SmallVector<float> data;
};

// Materialize an FP32 const view from `v`. Reads from a `tosa.const`,
// covering both inline `DenseElementsAttr` and out-of-line
// `DenseF32ResourceElementsAttr`. Returns nullopt if `v` isn't a
// foldable FP32 constant.
static std::optional<ConstF32View> getConstF32(Value v) {
  auto c = v.getDefiningOp<tosa::ConstOp>();
  if (!c)
    return std::nullopt;
  auto attr = dyn_cast<ElementsAttr>(c.getValuesAttr());
  if (!attr)
    return std::nullopt;

  auto rt = dyn_cast<RankedTensorType>(attr.getType());
  if (!rt || !rt.hasStaticShape() || !rt.getElementType().isF32())
    return std::nullopt;

  ConstF32View view;
  view.shape = rt.getShape();
  view.data.reserve(rt.getNumElements());

  if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
    // Covers both regular dense<> and splats; getValues<APFloat>() walks
    // the logical element sequence regardless.
    for (APFloat f : dense.getValues<APFloat>())
      view.data.push_back(f.convertToFloat());
  } else if (auto r = dyn_cast<DenseF32ResourceElementsAttr>(attr)) {
    auto arr = r.tryGetAsArrayRef();
    if (!arr)
      return std::nullopt;
    view.data.assign(arr->begin(), arr->end());
  } else {
    return std::nullopt;
  }
  return view;
}

// Numpy-style broadcast shape; empty SmallVector on incompatible inputs.
// Right-aligns shapes and broadcasts size-1 dims.
static SmallVector<int64_t> broadcastShape(ArrayRef<int64_t> a,
                                            ArrayRef<int64_t> b) {
  size_t maxRank = std::max(a.size(), b.size());
  SmallVector<int64_t> out(maxRank);
  for (size_t i = 0; i < maxRank; ++i) {
    int64_t ai = i < a.size() ? a[a.size() - 1 - i] : 1;
    int64_t bi = i < b.size() ? b[b.size() - 1 - i] : 1;
    if (ai != 1 && bi != 1 && ai != bi)
      return {};
    out[maxRank - 1 - i] = std::max(ai, bi);
  }
  return out;
}

// Source linear index when broadcasting `inShape` into `outShape` for the
// row-major output position `dst`. Right-aligns shapes; clamps coordinates
// to 0 in dims where inShape has size 1.
static size_t broadcastSrc(size_t dst, ArrayRef<int64_t> outShape,
                           ArrayRef<int64_t> inShape) {
  SmallVector<int64_t, 8> coord(outShape.size());
  for (int64_t i = (int64_t)outShape.size() - 1; i >= 0; --i) {
    coord[i] = (int64_t)(dst % outShape[i]);
    dst /= outShape[i];
  }
  size_t pad = outShape.size() - inShape.size();
  size_t src = 0;
  for (size_t i = 0; i < inShape.size(); ++i) {
    int64_t c = inShape[i] == 1 ? 0 : coord[i + pad];
    src = src * inShape[i] + c;
  }
  return src;
}

// Build a tosa.const from `outShape` + `data` and replace `op`. Pure
// tosa→tosa, so the downstream --tosa-to-timvx pass picks it up via
// the existing ConstOpConversion.
static void replaceWithF32Const(Operation *op, ArrayRef<int64_t> outShape,
                                ArrayRef<float> data,
                                PatternRewriter &rewriter) {
  auto f32 = rewriter.getF32Type();
  auto outTy = RankedTensorType::get(outShape, f32);
  rewriter.replaceOpWithNewOp<tosa::ConstOp>(
      op, outTy, DenseElementsAttr::get(outTy, data));
}

// Generic binary elementwise fold over FP32 constants with broadcasting.
// On success, replaces `op` with a `timvx.const` carrying the computed
// values. On failure (operands non-const, dynamic shape, non-FP32
// element type, broadcast incompatible), returns failure() and leaves
// the IR untouched.
template <typename Compute>
static LogicalResult foldBinaryFP(Operation *op, Value lhs, Value rhs,
                                  Compute compute,
                                  PatternRewriter &rewriter) {
  auto lhsC = getConstF32(lhs);
  auto rhsC = getConstF32(rhs);
  if (!lhsC || !rhsC)
    return failure();

  SmallVector<int64_t> outShape = broadcastShape(lhsC->shape, rhsC->shape);
  if (outShape.empty() && (!lhsC->shape.empty() || !rhsC->shape.empty()))
    return failure();

  int64_t numel = 1;
  for (int64_t d : outShape)
    numel *= d;
  if (numel == 0)
    numel = 1; // both rank-0 scalars

  SmallVector<float> result(numel);
  for (size_t i = 0; i < (size_t)numel; ++i) {
    size_t li = broadcastSrc(i, outShape, lhsC->shape);
    size_t ri = broadcastSrc(i, outShape, rhsC->shape);
    result[i] = compute(lhsC->data[li], rhsC->data[ri]);
  }
  replaceWithF32Const(op, outShape, result, rewriter);
  return success();
}

template <typename Compute>
static LogicalResult foldUnaryFP(Operation *op, Value src, Compute compute,
                                 PatternRewriter &rewriter) {
  auto srcC = getConstF32(src);
  if (!srcC)
    return failure();

  SmallVector<float> result(srcC->data.size());
  for (size_t i = 0; i < srcC->data.size(); ++i)
    result[i] = compute(srcC->data[i]);
  replaceWithF32Const(op, srcC->shape, result, rewriter);
  return success();
}

// Compute primitives.
static float fAdd(float a, float b) { return a + b; }
static float fSub(float a, float b) { return a - b; }
static float fMul(float a, float b) { return a * b; }
static float fPow(float a, float b) { return std::pow(a, b); }
static float fReciprocal(float x) { return 1.0f / x; }

// Pattern stamps. Higher benefit so each one wins over the regular
// tosa→timvx convert pattern when both operands are constants.
// Patterns. Driven by applyPatternsGreedily in TosaConstFoldPass below,
// which iterates to fixed point so upstream folds trigger re-matching of
// their consumers and the whole chain collapses.
struct AddConstFold : public OpRewritePattern<tosa::AddOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::AddOp op,
                                PatternRewriter &rewriter) const final {
    return foldBinaryFP(op, op.getInput1(), op.getInput2(), fAdd, rewriter);
  }
};
struct SubConstFold : public OpRewritePattern<tosa::SubOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::SubOp op,
                                PatternRewriter &rewriter) const final {
    return foldBinaryFP(op, op.getInput1(), op.getInput2(), fSub, rewriter);
  }
};
struct MulConstFold : public OpRewritePattern<tosa::MulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::MulOp op,
                                PatternRewriter &rewriter) const final {
    if (!isConstantZero(op.getShift()))
      return failure();
    return foldBinaryFP(op, op.getInput1(), op.getInput2(), fMul, rewriter);
  }
};
struct PowConstFold : public OpRewritePattern<tosa::PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::PowOp op,
                                PatternRewriter &rewriter) const final {
    return foldBinaryFP(op, op.getInput1(), op.getInput2(), fPow, rewriter);
  }
};
struct ReciprocalConstFold : public OpRewritePattern<tosa::ReciprocalOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::ReciprocalOp op,
                                PatternRewriter &rewriter) const final {
    return foldUnaryFP(op, op.getInput1(), fReciprocal, rewriter);
  }
};
// tosa.reshape of a constant is a pure layout change. Materialize the
// values from whatever storage the source const uses (dense_resource is
// common for large weight tensors) and emit a fresh timvx.const at the
// target shape.
struct ReshapeConstFold : public OpRewritePattern<tosa::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::ReshapeOp op,
                                PatternRewriter &rewriter) const final {
    auto src = getConstF32(op.getInput1());
    if (!src)
      return failure();
    auto outTy = cast<RankedTensorType>(op.getType());
    if (!outTy.hasStaticShape() ||
        (size_t)outTy.getNumElements() != src->data.size())
      return failure();
    replaceWithF32Const(op, outTy.getShape(), src->data, rewriter);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Boilerplate
//===----------------------------------------------------------------------===//

/// Lower a TOSA op to a TIM-VX op when the source op takes only tensor
/// operands and no attributes.
template <typename TosaOp, typename TIMVXOp>
struct TensorOnlyOpConversion : public OpConversionPattern<TosaOp> {
  using OpConversionPattern<TosaOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<TosaOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(TosaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<TIMVXOp>(op, op->getResultTypes(),
                                         adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Bespoke patterns for ops whose attributes need translation
//===----------------------------------------------------------------------===//

/// tosa.clamp -> timvx.clip
/// Drops `nan_mode` (TIM-VX has no per-op NaN policy).
/// TODO: look into this and find if nan_mode breaks anything.
struct ClampOpConversion : public OpConversionPattern<tosa::ClampOp> {
  using OpConversionPattern<tosa::ClampOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::ClampOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto minVal = dyn_cast<FloatAttr>(op.getMinValAttr());
    auto maxVal = dyn_cast<FloatAttr>(op.getMaxValAttr());
    if (!minVal || !maxVal)
      return rewriter.notifyMatchFailure(
          op, "integer clamp not yet supported by tosa-to-timvx");

    rewriter.replaceOpWithNewOp<ClipOp>(op, op.getType(), adaptor.getInput(),
                                        minVal, maxVal);
    return success();
  }
};

/// tosa.mul -> timvx.multiply
///
/// TODO: TOSA computes (input1 * input2) >> shift; tim::vx::Multiply has no
/// shift.
///
/// We reject the rewrite at non-zero shift since otherwise the semantics
/// change. We will handle with adding an extra shift operation later.
struct MulOpConversion : public OpConversionPattern<tosa::MulOp> {
  using OpConversionPattern<tosa::MulOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isConstantZero(op.getShift()))
      return rewriter.notifyMatchFailure(
          op,
          "non-zero or non-constant shift not yet supported by tosa-to-timvx");

    rewriter.replaceOpWithNewOp<MultiplyOp>(
        op, op.getType(), adaptor.getInput1(), adaptor.getInput2());
    return success();
  }
};

/// Const passthrough
struct ConstOpConversion : public OpConversionPattern<tosa::ConstOp> {
  using OpConversionPattern<tosa::ConstOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::ConstOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto values = dyn_cast<ElementsAttr>(op.getValuesAttr());
    if (!values)
      return rewriter.notifyMatchFailure(
          op, "expected dense integer elements for const_shape values");

    rewriter.replaceOpWithNewOp<ConstOp>(op, op.getType(), values,
                                          /*quant_scale=*/FloatAttr{},
                                          /*quant_zp=*/IntegerAttr{});
    return success();
  }
};

/// const_shape passthrough
/// We are casting into a tensor of 1xIndex instead of custom shape for
/// simplicity.
struct ConstShapeOpConversion : public OpConversionPattern<tosa::ConstShapeOp> {
  using OpConversionPattern<tosa::ConstShapeOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::ConstShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto values = dyn_cast<DenseIntElementsAttr>(op.getValuesAttr());
    if (!values)
      return rewriter.notifyMatchFailure(op, "non-dense const_shape values");

    // Result type changes from !tosa.shape<N> to tensor<Nxindex>.
    auto resultType =
        RankedTensorType::get({static_cast<int64_t>(values.getNumElements())},
                              rewriter.getIndexType());

    // Re-attribute the dense data into the new tensor type.
    auto reTyped = DenseIntElementsAttr::get(
        resultType, llvm::to_vector(values.getValues<APInt>()));

    rewriter.replaceOpWithNewOp<ConstShapeOp>(op, resultType, reTyped);
    return success();
  }
};

/// tosa.reshape / tosa.slice carry shape operands typed as `!tosa.shape<N>`,
/// while their timvx counterparts take `tensor<Nxindex>`. We rely on
/// ConstShapeOpConversion having already rewritten the producer; the adaptor
/// then exposes the shape operand with its new tensor<Nxindex> type and we
/// hand it through. If the producer hasn't been rewritten (e.g. it isn't a
/// const_shape), the adaptor still hands back the !tosa.shape<N> value and
/// the rewrite fails the type check below — better than silently building an
/// op that can't verify.
// Forward-declare for quant-passthrough ops.
struct QuantInfoMap;

// Helper: convert a {scale, zp} pair from QuantInfoMap to MLIR attrs, or
// return nulls when the value isn't on the int8 quant chain (FP path).
template <typename Builder>
static std::pair<FloatAttr, IntegerAttr> qmapAttrs(const QuantInfoMap &qm,
                                                     Value v, Builder &b);

struct ReshapeOpConversion : public OpConversionPattern<tosa::ReshapeOp> {
  ReshapeOpConversion(MLIRContext *ctx, const QuantInfoMap &qm)
      : OpConversionPattern(ctx), quantInfo(qm) {}
  const QuantInfoMap &quantInfo;
  LogicalResult
  matchAndRewrite(tosa::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value shape = adaptor.getShape();
    if (!isa<RankedTensorType>(shape.getType()))
      return rewriter.notifyMatchFailure(
          op, "shape operand not converted to tensor<Nxindex>");
    auto [s, z] = qmapAttrs(quantInfo, op.getResult(), rewriter);
    rewriter.replaceOpWithNewOp<ReshapeOp>(
        op, op.getType(), adaptor.getInput1(), shape, s, z);
    return success();
  }
};

struct SliceOpConversion : public OpConversionPattern<tosa::SliceOp> {
  SliceOpConversion(MLIRContext *ctx, const QuantInfoMap &qm)
      : OpConversionPattern(ctx), quantInfo(qm) {}
  const QuantInfoMap &quantInfo;
  LogicalResult
  matchAndRewrite(tosa::SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value start = adaptor.getStart();
    Value size = adaptor.getSize();
    if (!isa<RankedTensorType>(start.getType()) ||
        !isa<RankedTensorType>(size.getType()))
      return rewriter.notifyMatchFailure(
          op, "start/size operands not converted to tensor<Nxindex>");
    auto [s, z] = qmapAttrs(quantInfo, op.getResult(), rewriter);
    rewriter.replaceOpWithNewOp<SliceOp>(op, op.getType(), adaptor.getInput1(),
                                         start, size, s, z);
    return success();
  }
};

/// Transpose a `[1, K, N]`-shaped dense weight constant down to `[N, K]`,
/// dropping the leading batch dim and swapping the inner two. Used by the
/// matmul→fully_connected rewrite. Only handles inline DenseElementsAttr
/// (the form tosa-import emits for inline `dense<"0x...">`); returns null
/// for splats or DenseResourceElementsAttr so the caller falls back to the
/// general matmul path.
static DenseElementsAttr transposeWeightForFC(ElementsAttr in, int64_t K,
                                              int64_t N) {
  auto dense = dyn_cast<DenseElementsAttr>(in);
  if (!dense || dense.isSplat())
    return {};

  auto inTy = cast<RankedTensorType>(in.getType());
  Type elemTy = inTy.getElementType();
  // TIM-VX FC reads weights as `{in_features, out_features}` innermost-first
  // (`vsi_nn_op_fullconnect2.c` takes `ofm = weights_size[dim_num-1]`, i.e.
  // out_features lives in the *outermost* slot). Our data is rearranged so
  // K is the fastest-varying dim, so the correct shape declaration is
  // `[K, N]` (with K at index 0 = innermost in TIM-VX's view).
  auto outTy = RankedTensorType::get({K, N}, elemTy);

  // Linear index in source `[1, K, N]` (b=0):           k * N + n.
  // Linear index in destination buffer (K fastest):     n * K + k.
  if (isa<FloatType>(elemTy)) {
    SmallVector<APFloat> values;
    values.reserve(static_cast<size_t>(N) * K);
    auto src = llvm::to_vector(dense.getValues<APFloat>());
    if (static_cast<int64_t>(src.size()) != K * N)
      return {};
    for (int64_t n = 0; n < N; ++n)
      for (int64_t k = 0; k < K; ++k)
        values.push_back(src[k * N + n]);
    return DenseElementsAttr::get(outTy, values);
  }
  if (isa<IntegerType>(elemTy)) {
    SmallVector<APInt> values;
    values.reserve(static_cast<size_t>(N) * K);
    auto src = llvm::to_vector(dense.getValues<APInt>());
    if (static_cast<int64_t>(src.size()) != K * N)
      return {};
    for (int64_t n = 0; n < N; ++n)
      for (int64_t k = 0; k < K; ++k)
        values.push_back(src[k * N + n]);
    return DenseElementsAttr::get(outTy, values);
  }
  return {};
}

/// Build `timvx.reshape(src, timvx.const_shape(newShape))` and return the
/// result. The constant_shape op carries an `IndexElementsAttr` per the
/// dialect's contract.
static Value emitReshape(OpBuilder &b, Location loc, Value src,
                         ArrayRef<int64_t> newShape) {
  auto srcTy = cast<RankedTensorType>(src.getType());
  auto shapeTy = RankedTensorType::get({static_cast<int64_t>(newShape.size())},
                                       b.getIndexType());
  SmallVector<APInt> shapeInts;
  shapeInts.reserve(newShape.size());
  for (int64_t s : newShape)
    shapeInts.emplace_back(/*numBits=*/64, /*val=*/static_cast<uint64_t>(s),
                           /*isSigned=*/true);
  auto shapeAttr = DenseIntElementsAttr::get(shapeTy, shapeInts);
  Value shapeConst = ConstShapeOp::create(b, loc, shapeTy, shapeAttr);

  auto outTy = RankedTensorType::get(newShape, srcTy.getElementType());
  return ReshapeOp::create(b, loc, outTy, src, shapeConst,
                            /*output_scale=*/FloatAttr{},
                            /*output_zp=*/IntegerAttr{});
}

/// tosa.matmul → timvx.fully_connected (preferred when B is a constant
/// weight) or timvx.matmul (fallback for activation×activation).
///
/// TOSA's matmul has scalar-tensor zero-point operands (a_zp / b_zp); TIM-VX's
/// matmul carries quant params on the tensor type instead, so we drop them.
/// As with conv2d, only the constant-zero zp case is handled here — non-zero
/// or non-constant zp would change semantics.
///
/// FC fast path: when B traces back to a `tosa.const` we compile-time
/// transpose the weight from `[1, K, N]` to `[N, K]`, synthesize a zero
/// bias of shape `[N]`, and reshape A from `[Ba, M, K]` to `[Ba*M, K]` to
/// satisfy FC's rank-2 contract. The output is reshaped back to the
/// original `[Ba, M, N]`. FC is faster than Matmul on Vivante NPUs because
/// the weight is pre-tiled at bind time.
struct MatMulOpConversion : public OpConversionPattern<tosa::MatMulOp> {
  using OpConversionPattern<tosa::MatMulOp>::OpConversionPattern;

  // Try to rewrite as fully_connected. Returns success on rewrite, failure
  // (without modifying IR) when shapes/attr storage don't fit the pattern,
  // so the caller can fall back to the general matmul path.
  LogicalResult tryRewriteAsFC(tosa::MatMulOp op,
                               ConversionPatternRewriter &rewriter) const {
    auto bConst = op.getB().getDefiningOp<tosa::ConstOp>();
    if (!bConst)
      return failure();

    auto aTy = dyn_cast<RankedTensorType>(op.getA().getType());
    auto bTy = dyn_cast<RankedTensorType>(op.getB().getType());
    auto outTy = dyn_cast<RankedTensorType>(op.getType());
    if (!aTy || !bTy || !outTy || aTy.getRank() != 3 || bTy.getRank() != 3)
      return failure();
    int64_t Ba = aTy.getDimSize(0), M = aTy.getDimSize(1), K = aTy.getDimSize(2);
    int64_t Bb = bTy.getDimSize(0), Kb = bTy.getDimSize(1), N = bTy.getDimSize(2);
    if (Ba == ShapedType::kDynamic || M == ShapedType::kDynamic ||
        K == ShapedType::kDynamic || Bb != 1 || Kb != K)
      return failure();
    if (aTy.getElementType() != bTy.getElementType())
      return failure();

    auto values = dyn_cast<ElementsAttr>(bConst.getValuesAttr());
    if (!values)
      return failure();
    auto transposed = transposeWeightForFC(values, K, N);
    if (!transposed)
      return failure();

    Location loc = op.getLoc();
    Type elemTy = bTy.getElementType();

    // Transposed weight: type `[K, N]` to match TIM-VX innermost-first
    // expectation (in_features innermost, out_features outermost).
    auto weightTy = RankedTensorType::get({K, N}, elemTy);
    Value weight = ConstOp::create(rewriter, loc, weightTy, transposed,
                                    /*quant_scale=*/FloatAttr{},
                                    /*quant_zp=*/IntegerAttr{});

    // Zero bias: [N]. Splat keeps the IR small even for large N.
    auto biasTy = RankedTensorType::get({N}, elemTy);
    auto biasAttr = DenseElementsAttr::get(
        biasTy, cast<TypedAttr>(rewriter.getZeroAttr(elemTy)));
    Value bias = ConstOp::create(rewriter, loc, biasTy, biasAttr,
                                  /*quant_scale=*/FloatAttr{},
                                  /*quant_zp=*/IntegerAttr{});

    // Coerce A to rank-2 [Ba*M, K] and run FC.
    Value reshapedA = emitReshape(rewriter, loc, op.getA(), {Ba * M, K});
    auto fcOutTy = RankedTensorType::get({Ba * M, N}, outTy.getElementType());
    Value fc = FullyConnectedOp::create(rewriter, loc, fcOutTy, reshapedA,
                                         weight, bias,
                                         /*output_scale=*/FloatAttr{},
                                         /*output_zp=*/IntegerAttr{});

    // Restore the original rank-3 output shape.
    Value restored = emitReshape(rewriter, loc, fc, outTy.getShape());
    rewriter.replaceOp(op, restored);
    return success();
  }

  LogicalResult
  matchAndRewrite(tosa::MatMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isConstantZero(op.getAZp()))
      return rewriter.notifyMatchFailure(op, "non-zero a zero-point");
    if (!isConstantZero(op.getBZp()))
      return rewriter.notifyMatchFailure(op, "non-zero b zero-point");

    if (succeeded(tryRewriteAsFC(op, rewriter)))
      return success();

    rewriter.replaceOpWithNewOp<MatMulOp>(op, op.getType(), adaptor.getA(),
                                          adaptor.getB());
    return success();
  }
};

/// tosa.conv2d -> timvx.conv2d
///
/// TODO: Dropped attributes / operands:
///   - input_zp / weight_zp: handled at the tensor type via quant params on
///     the timvx side, so dropped here. Constant-zero only — non-zero would
///     change semantics.
///   - acc_type: tosa carries an explicit accumulator type;
///     tim::vx::ops::Conv2d picks its accumulator internally (i32 for INT8, f32
//      for FP). We have no knob to forward this onto, so we drop it. If a
//      frontend ever requests an accumulator that diverges from TIM-VX's
//      default the result will be wrong; revisit if that comes up.
///   - local_bound: rare flag (default false); we don't model it. Match-fail
///     when set so it isn't dropped silently.
struct Conv2DOpConversion : public OpConversionPattern<tosa::Conv2DOp> {
  using OpConversionPattern<tosa::Conv2DOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::Conv2DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isConstantZero(op.getInputZp()))
      return rewriter.notifyMatchFailure(op, "non-zero input zero-point");
    if (!isConstantZero(op.getWeightZp()))
      return rewriter.notifyMatchFailure(op, "non-zero weight zero-point");
    if (op.getLocalBound())
      return rewriter.notifyMatchFailure(op, "local_bound=true not supported");

    rewriter.replaceOpWithNewOp<Conv2DOp>(
        op, op.getType(), adaptor.getInput(), adaptor.getWeight(),
        adaptor.getBias(), op.getPadAttr(), op.getStrideAttr(),
        op.getDilationAttr(),
        /*output_scale=*/FloatAttr{}, /*output_zp=*/IntegerAttr{});
    return success();
  }
};

/// pool2d rewrites. nan_mode / acc_type dropped
struct MaxPool2DConversion : public OpConversionPattern<tosa::MaxPool2dOp> {
  MaxPool2DConversion(MLIRContext *ctx, const QuantInfoMap &qm)
      : OpConversionPattern(ctx), quantInfo(qm) {}
  const QuantInfoMap &quantInfo;
  LogicalResult
  matchAndRewrite(tosa::MaxPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto [s, z] = qmapAttrs(quantInfo, op.getResult(), rewriter);
    rewriter.replaceOpWithNewOp<Pool2DOp>(
        op, op.getType(), adaptor.getInput(), PoolType::MAX,
        adaptor.getKernelAttr(), adaptor.getStrideAttr(), op.getPadAttr(),
        s, z);
    return success();
  }
};
struct AvgPool2DConversion : public OpConversionPattern<tosa::AvgPool2dOp> {
  AvgPool2DConversion(MLIRContext *ctx, const QuantInfoMap &qm)
      : OpConversionPattern(ctx), quantInfo(qm) {}
  const QuantInfoMap &quantInfo;
  LogicalResult
  matchAndRewrite(tosa::AvgPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // For a quantized avg-pool, input_zp == output_zp must hold (avg is
    // linear in real-value space, so the zp passes through). We don't
    // assert that here — the upstream peephole that emits this op
    // already chooses the same zp on both sides; for hand-written TOSA
    // the user's responsibility.
    auto [s, z] = qmapAttrs(quantInfo, op.getResult(), rewriter);
    rewriter.replaceOpWithNewOp<Pool2DOp>(
        op, op.getType(), adaptor.getInput(), PoolType::AVG,
        adaptor.getKernelAttr(), adaptor.getStrideAttr(), op.getPadAttr(),
        s, z);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Quantization plumbing
//===----------------------------------------------------------------------===//
//
// Quantized TOSA from the tflite import has the structure
//
//     tosa.conv2d (i8 in, i8 weight, i32 bias) -> i32 (accumulator)
//     tosa.rescale  i32 -> i8 (multiplier, shift, output_zp)
//
// TIM-VX wants the conv to be a single op carrying per-tensor (Si, Sw, So,
// in_zp, out_zp); it derives the integer multiplier+shift internally from
// those scales. So we fuse the pair, choosing canonical absolute scales:
//
//   Sw  = 1/128                          (symmetric int8 weight, weight_zp = 0)
//   Si  = (chained from upstream)        (input scale; anchor is 1.0 at func arg)
//   So  = Si * Sw / M_eff,  where M_eff = mul / 2^shift
//   Sb  = Si * Sw                        (i32 bias: scale = Si*Sw, zp = 0)
//
// TIM-VX only consumes the *ratio* Si*Sw/So, not the individual scales, so
// the choice of Si=1.0 at the model input is harmless to the integer math.
// The fp32 dequantize/quantize dance around residuals reads its own
// scale/zp from constants, decoupled from this chain.
struct QuantInfo {
  double scale;
  int64_t zp;
};

// Builds a `Value -> {scale, zp}` map by walking the IR top-down, so each
// conversion pattern can look up the (S, Z) of any of its operands or its
// result without re-deriving. Populated once before the partial conversion.
struct QuantInfoMap {
  llvm::DenseMap<Value, QuantInfo> map;

  void seedFuncArgs(func::FuncOp f) {
    for (BlockArgument a : f.getArguments()) {
      auto rt = dyn_cast<RankedTensorType>(a.getType());
      if (!rt) continue;
      auto it = dyn_cast<IntegerType>(rt.getElementType());
      if (!it || it.getWidth() > 8) continue;
      // Anchor at scale=1.0 zp=0 — the int8 conv chain only depends on
      // scale ratios, so any fixed anchor produces the same int8 byte
      // pattern.  An optional `tim::vx::input_scale` / `..._zp` arg attr
      // overrides this if the front-end supplied one.
      double s = 1.0;
      int64_t z = 0;
      if (auto sa = f.getArgAttrOfType<FloatAttr>(a.getArgNumber(),
                                                   "tim::vx::input_scale"))
        s = sa.getValueAsDouble();
      if (auto za = f.getArgAttrOfType<IntegerAttr>(a.getArgNumber(),
                                                     "tim::vx::input_zp"))
        z = za.getInt();
      map[a] = {s, z};
    }
  }

  std::optional<QuantInfo> lookup(Value v) const {
    auto it = map.find(v);
    if (it == map.end()) return std::nullopt;
    return it->second;
  }

  void set(Value v, QuantInfo qi) { map[v] = qi; }
};

// Definition for the forward-declared template above. Looks up `v` in
// the qmap and returns matching FloatAttr / IntegerAttr suitable for
// `output_scale` / `output_zp`. Returns null attrs when `v` isn't on
// the int8 quant chain (FP32 path).
template <typename Builder>
std::pair<FloatAttr, IntegerAttr> qmapAttrs(const QuantInfoMap &qm, Value v,
                                              Builder &b) {
  auto qi = qm.lookup(v);
  if (!qi) return {FloatAttr{}, IntegerAttr{}};
  return {b.getF64FloatAttr(qi->scale), b.getI64IntegerAttr(qi->zp)};
}

// Walk forward through layout-/range-preserving ops looking for a
// consumer whose attributes pin the expected zero-point of `v`. Returns
// the first match (e.g. `tosa.conv2d.input_zp` or `tosa.pad.pad_const`).
//
// Used to recover the func-arg zp from the model: TOSA doesn't store
// quant on the type system in this dialect version, so the only reliable
// signal is the consumer's `input_zp` operand.
static std::optional<int64_t> consumerExpectedZp(Value v) {
  llvm::SmallPtrSet<Operation *, 16> visited;
  llvm::SmallVector<Value, 8> stack{v};
  while (!stack.empty()) {
    Value cur = stack.pop_back_val();
    for (Operation *user : cur.getUsers()) {
      if (!visited.insert(user).second) continue;
      if (auto conv = dyn_cast<tosa::Conv2DOp>(user))
        if (auto z = matchConstScalarInt(conv.getInputZp()))
          return *z;
      if (auto pad = dyn_cast<tosa::PadOp>(user)) {
        if (auto pc = pad.getPadConst().getDefiningOp<tosa::ConstOp>()) {
          if (auto attr = dyn_cast<DenseIntElementsAttr>(pc.getValuesAttr())) {
            if (attr.getNumElements() >= 1)
              return (*attr.getValues<APInt>().begin()).getSExtValue();
          }
        }
      }
      // Layout-passthrough ops: keep walking through them.
      if (isa<tosa::TransposeOp, tosa::ReshapeOp, tosa::SliceOp>(user))
        if (user->getNumResults() == 1)
          stack.push_back(user->getResult(0));
    }
  }
  return std::nullopt;
}

// Top-down propagation: pool / cast (i8 path) / pad pass quant through
// unchanged; tosa.rescale-after-conv2d is the place where a new (S, Z)
// is computed. Constants (weight, bias) are tagged in the rescale handler
// because their scales depend on the consuming conv's chain.
static void buildQuantInfoMap(func::FuncOp f, QuantInfoMap &qm) {
  qm.seedFuncArgs(f);

  // After seeding, refine each func arg's zp using the first consumer's
  // expectation. Without this we'd anchor at zp=0, which mismatches models
  // whose tflite quantization assigns a non-zero input_zp.
  for (BlockArgument a : f.getArguments()) {
    auto qi = qm.lookup(a);
    if (!qi) continue;
    if (auto z = consumerExpectedZp(a)) qm.set(a, {qi->scale, *z});
  }

  f.walk([&](Operation *op) {
    if (auto resc = dyn_cast<tosa::RescaleOp>(op)) {
      auto conv = resc.getInput().getDefiningOp<tosa::Conv2DOp>();
      if (!conv) {
        // Not a conv-rescale fusion target — leave for a future pattern.
        return;
      }
      auto siOpt = qm.lookup(conv.getInput());
      QuantInfo si = siOpt.value_or(QuantInfo{1.0, 0});

      auto mul = matchConstScalarInt(resc.getMultiplier());
      auto shift = matchConstScalarInt(resc.getShift());
      auto outZp = matchConstScalarInt(resc.getOutputZp());
      if (!mul || !shift || !outZp) return;

      double M = static_cast<double>(*mul) * std::pow(2.0, -double(*shift));
      double Sw = 1.0 / 128.0;
      double So = si.scale * Sw / M;
      qm.set(resc.getResult(), {So, *outZp});
      return;
    }
    // Layout-/range-preserving ops: result inherits operand quant.
    // (Avg pool is linear in real-value space — quant passes through
    // unchanged — provided input_zp == output_zp on the tosa op, which
    // the AvgPoolReduceFold rewriter guarantees.)
    if (isa<tosa::MaxPool2dOp, tosa::AvgPool2dOp, tosa::PadOp,
            tosa::TransposeOp, tosa::ReshapeOp, tosa::SliceOp>(op)) {
      if (op->getNumOperands() < 1 || op->getNumResults() != 1) return;
      auto qi = qm.lookup(op->getOperand(0));
      if (qi) qm.set(op->getResult(0), *qi);
      return;
    }
    // tosa.cast: drops quant in general (the i8→f32 dequant cast, or
    // f32→i32 intermediate cast inside the requantize chain). The one
    // exception is the *final* requantize cast — `tosa.cast f32 →
    // narrow-int` produced by RequantI32SkipFold — which the peephole
    // tags with `timvx.output_scale` / `timvx.output_zp` discardable
    // attrs encoding the (output_scale, output_zp) pair the downstream
    // conv2d expects on its input. Without reading these here, the cast
    // result would land in the IR with no quant, the EmitC path would
    // build an INT8 spec without Quantization(), and the next conv2d
    // would receive a no-quant tensor (kernel rejects it).
    if (auto cast = dyn_cast<tosa::CastOp>(op)) {
      auto scaleAttr = cast->getAttrOfType<FloatAttr>("timvx.output_scale");
      auto zpAttr = cast->getAttrOfType<IntegerAttr>("timvx.output_zp");
      if (scaleAttr && zpAttr)
        qm.set(cast.getResult(),
               {scaleAttr.getValueAsDouble(), zpAttr.getInt()});
      return;
    }
  });
}

// Matches `tosa.rescale` whose input is a `tosa.conv2d`, and replaces both
// with a single quantized `timvx.conv2d` carrying output_scale / output_zp.
// The new conv's weight + bias consts pick up their own quant attrs derived
// from the canonical Sw = 1/128 (and Sb = Si*Sw).
struct RescaleConvFusion : public OpConversionPattern<tosa::RescaleOp> {
  RescaleConvFusion(MLIRContext *ctx, const QuantInfoMap &qm)
      : OpConversionPattern(ctx, /*benefit=*/10), quantInfo(qm) {}
  const QuantInfoMap &quantInfo;

  LogicalResult
  matchAndRewrite(tosa::RescaleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto conv = op.getInput().getDefiningOp<tosa::Conv2DOp>();
    if (!conv)
      return rewriter.notifyMatchFailure(op, "input is not a tosa.conv2d");
    if (!isa<IntegerType>(
            cast<RankedTensorType>(conv.getType()).getElementType()))
      return rewriter.notifyMatchFailure(op, "conv2d output not integer");
    if (op.getPerChannel())
      return rewriter.notifyMatchFailure(
          op, "per-channel rescale not yet supported (requires per-channel "
              "weight requant; see project_a733_fp32_conv_unsupported.md)");
    if (conv.getLocalBound())
      return rewriter.notifyMatchFailure(op, "local_bound=true not supported");

    auto inZpVal = matchConstScalarInt(conv.getInputZp());
    auto wZpVal = matchConstScalarInt(conv.getWeightZp());
    auto outZpVal = matchConstScalarInt(op.getOutputZp());
    if (!inZpVal || !wZpVal || !outZpVal)
      return rewriter.notifyMatchFailure(
          op, "non-constant zp on conv or rescale");
    if (*wZpVal != 0)
      return rewriter.notifyMatchFailure(
          op, "non-zero weight zp not supported (need symmetric int8 weight)");

    // Look up the input scale (chained from upstream) and compute output
    // scale via the conv-rescale fusion math.
    QuantInfo si =
        quantInfo.lookup(conv.getInput()).value_or(QuantInfo{1.0, 0});
    if (auto inferred = quantInfo.lookup(op.getResult()); !inferred)
      return rewriter.notifyMatchFailure(
          op, "rescale result missing from QuantInfoMap "
              "(buildQuantInfoMap should have populated it)");
    QuantInfo so = *quantInfo.lookup(op.getResult());
    double sw = 1.0 / 128.0;
    double sb = si.scale * sw;

    // Re-emit the weight const with symmetric-int8 quant attrs.
    auto weightConst = conv.getWeight().getDefiningOp<tosa::ConstOp>();
    if (!weightConst)
      return rewriter.notifyMatchFailure(op, "weight is not a tosa.const");
    auto weightVals = dyn_cast<ElementsAttr>(weightConst.getValuesAttr());
    if (!weightVals)
      return rewriter.notifyMatchFailure(op, "weight const lacks ElementsAttr");
    Value weight = ConstOp::create(
        rewriter, conv.getLoc(), weightConst.getType(), weightVals,
        rewriter.getF64FloatAttr(sw), rewriter.getI64IntegerAttr(0));

    auto biasConst = conv.getBias().getDefiningOp<tosa::ConstOp>();
    if (!biasConst)
      return rewriter.notifyMatchFailure(op, "bias is not a tosa.const");
    auto biasVals = dyn_cast<ElementsAttr>(biasConst.getValuesAttr());
    if (!biasVals)
      return rewriter.notifyMatchFailure(op, "bias const lacks ElementsAttr");
    Value bias = ConstOp::create(
        rewriter, conv.getLoc(), biasConst.getType(), biasVals,
        rewriter.getF64FloatAttr(sb), rewriter.getI64IntegerAttr(0));

    // Build the quant timvx.conv2d that subsumes both ops. Output type
    // mirrors the rescale's i8 result.
    rewriter.replaceOpWithNewOp<Conv2DOp>(
        op, op.getType(), conv.getInput(), weight, bias, conv.getPadAttr(),
        conv.getStrideAttr(), conv.getDilationAttr(),
        rewriter.getF64FloatAttr(so.scale),
        rewriter.getI64IntegerAttr(so.zp));

    // The original tosa.conv2d's only user was this rescale; with the
    // rescale replaced its result is dead. Erase explicitly (the
    // conversion driver doesn't auto-DCE).
    rewriter.eraseOp(conv);
    return success();
  }
};

// `tosa.cast` -> `timvx.cast`. TIM-VX's Cast is the value-cast op (it
// dispatches to the GPU `cast` kernel, ignores scale/zp on either end
// for the cast itself). DataConvert is NOT used here — on this chip the
// `vivante.nn.tensorcopy` path COMPILE_FAILs for every f32→int direction,
// even when the upstream op_check table allows it. Cast handles all the
// pairs the residual quantize chain needs: f32→i32 raw, f32→i8|asym,
// i32 raw→i8|asym, plus the corresponding dequantize directions.
//
// We forward (scale, zp) from the QuantInfoMap to the result spec when
// available so the produced tensor still binds with proper Quantization
// metadata for downstream ops, even though the cast itself is
// value-preserving.
struct CastConversion : public OpConversionPattern<tosa::CastOp> {
  CastConversion(MLIRContext *ctx, const QuantInfoMap &qm)
      : OpConversionPattern(ctx), quantInfo(qm) {}
  const QuantInfoMap &quantInfo;
  LogicalResult
  matchAndRewrite(tosa::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    FloatAttr scale;
    IntegerAttr zp;
    if (auto qi = quantInfo.lookup(op.getResult())) {
      scale = rewriter.getF64FloatAttr(qi->scale);
      zp = rewriter.getI64IntegerAttr(qi->zp);
    }
    rewriter.replaceOpWithNewOp<CastOp>(op, op.getType(),
                                         adaptor.getInput(), scale, zp);
    return success();
  }
};

// `tosa.pad` -> `timvx.pad`. The padding spec is a `!tosa.shape<2N>`
// produced by `tosa.const_shape`, which the generic ConstShapeOpConversion
// has rewritten to a `tensor<2Nxindex>` constant. Quantized TOSA's
// `pad_const` operand is a 1-element tensor whose value is the input zero
// point (so the padded region is zero in real-value space); we pull it out
// as a scalar and stash it on `timvx.pad`'s F32Attr.
struct PadConversion : public OpConversionPattern<tosa::PadOp> {
  PadConversion(MLIRContext *ctx, const QuantInfoMap &qm)
      : OpConversionPattern(ctx), quantInfo(qm) {}
  const QuantInfoMap &quantInfo;
  LogicalResult
  matchAndRewrite(tosa::PadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value padding = adaptor.getPadding();
    if (!isa<RankedTensorType>(padding.getType()))
      return rewriter.notifyMatchFailure(
          op, "padding shape operand not yet converted to tensor<Nxindex>");

    // pad_const is a 1-element tensor; cast to f32 for the F32Attr.
    double padConstF = 0.0;
    Value padConst = adaptor.getPadConst();
    ElementsAttr padAttr;
    if (matchPattern(padConst, m_Constant(&padAttr)) &&
        padAttr.getNumElements() >= 1) {
      if (auto inty =
              dyn_cast<IntegerType>(padAttr.getShapedType().getElementType())) {
        padConstF = static_cast<double>(
            (*padAttr.getValues<APInt>().begin()).getSExtValue());
      } else {
        padConstF = static_cast<double>(
            (*padAttr.getValues<APFloat>().begin()).convertToDouble());
      }
    }

    // Pad's output shares quant with input; pull from qmap so the bound
    // tensor matches what the consumer (likely a conv2d) expects.
    FloatAttr scale;
    IntegerAttr zp;
    if (auto qi = quantInfo.lookup(op.getResult())) {
      scale = rewriter.getF64FloatAttr(qi->scale);
      zp = rewriter.getI64IntegerAttr(qi->zp);
    }

    // The padding spec is already a 1-D index tensor produced by
    // ConstShapeOpConversion; the runtime helper consumes the flat
    // [front,back,front,back,...] layout directly.
    rewriter.replaceOpWithNewOp<PadOp>(
        op, op.getType(), adaptor.getInput1(), padding,
        rewriter.getF32FloatAttr(static_cast<float>(padConstF)), scale, zp);
    return success();
  }
};

// Decompose `tosa.reduce_sum` over an i8/f32 tensor into a slice+add chain.
//
// The NPU's REDUCE op decomposes to REDUCE_MEAN_INTERNAL at compile time,
// which lacks an FP32 kernel on this chip (`Not found kernel "reduce_mean"`).
// Slice + Add are FP32-PASS per timvx_op_probe, so we emit `dim_K - 1` adds
// of `dim_K` unit-thickness slices along the reduce axis. Math is identical
// (sum of N elements is the same regardless of how the additions are
// associated). For typical avg-pool tails (7×7) this is 12 adds total —
// negligible runtime cost vs. an unrunnable reduce.
struct ReduceSumConversion : public OpConversionPattern<tosa::ReduceSumOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::ReduceSumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto inTy = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto outTy = dyn_cast<RankedTensorType>(op.getType());
    if (!inTy || !inTy.hasStaticShape() || !outTy)
      return rewriter.notifyMatchFailure(op, "non-static input shape");

    int64_t axis = static_cast<int64_t>(op.getAxis());
    int64_t rank = inTy.getRank();
    if (axis < 0 || axis >= rank)
      return rewriter.notifyMatchFailure(op, "axis out of range");

    int64_t reduceDim = inTy.getShape()[axis];
    if (reduceDim <= 0)
      return rewriter.notifyMatchFailure(op, "non-positive reduce dim");
    if (reduceDim == 1) {
      // Already singleton along this axis; just retype if needed.
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }

    Location loc = op.getLoc();
    Type elemTy = inTy.getElementType();
    auto sliceShape = llvm::to_vector(inTy.getShape());
    sliceShape[axis] = 1;
    auto sliceTy = RankedTensorType::get(sliceShape, elemTy);

    // tosa.slice takes a `!tosa.shape<rank>` for `start` and `size`.
    // Since slice support in our timvx pipeline goes through a
    // tosa.const_shape, we synthesize one per slice (start vector varies).
    auto shapeIdxTy = RankedTensorType::get({rank}, rewriter.getIndexType());
    SmallVector<int64_t> startBase(rank, 0);

    auto makeShape = [&](ArrayRef<int64_t> v) {
      auto attr = DenseIntElementsAttr::get(shapeIdxTy, v);
      auto shapeTy = tosa::shapeType::get(rewriter.getContext(), rank);
      return tosa::ConstShapeOp::create(rewriter, loc, shapeTy, attr)
          .getResult();
    };

    Value sizeShape = makeShape(sliceShape);

    SmallVector<Value> slices;
    slices.reserve(reduceDim);
    for (int64_t i = 0; i < reduceDim; ++i) {
      auto start = startBase;
      start[axis] = i;
      Value startShape = makeShape(start);
      Value sl = tosa::SliceOp::create(rewriter, loc, sliceTy,
                                        adaptor.getInput(), startShape,
                                        sizeShape);
      slices.push_back(sl);
    }

    Value acc = slices[0];
    for (size_t i = 1; i < slices.size(); ++i) {
      acc = tosa::AddOp::create(rewriter, loc, sliceTy, acc, slices[i]);
    }
    // The reduce_sum's result type equals slice type (both have the reduced
    // axis as size 1 with keep_dims).
    rewriter.replaceOp(op, acc);
    return success();
  }
};

/// transpose rewrite. nan_mode dropped
struct TransposeConversion : public OpConversionPattern<tosa::TransposeOp> {
  TransposeConversion(MLIRContext *ctx, const QuantInfoMap &qm)
      : OpConversionPattern(ctx), quantInfo(qm) {}
  const QuantInfoMap &quantInfo;
  LogicalResult
  matchAndRewrite(tosa::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto [s, z] = qmapAttrs(quantInfo, op.getResult(), rewriter);
    rewriter.replaceOpWithNewOp<TransposeOp>(
        op, op.getType(), adaptor.getInput1(), adaptor.getPermsAttr(), s, z);
    return success();
  }
};

// Fold `tosa.pad → tosa.conv2d` (or `tosa.pad → tosa.max_pool2d`) into the
// consumer's `pad` attribute. The standalone NN-core pad kernel
// (`vivante.nn.tensor.pad`) refuses to initialize on this hardware, but
// conv2d / pool2d carry their own padding internally — so absorbing the
// pad into the spatial op sidesteps that codepath.
//
// Pre-conditions:
//   * pad's only user is the conv2d / max_pool2d
//   * pad's spec const_shape carries `[N=0/0, H=fH/bH, W=fW/bW, C=0/0]`
//     (NHWC; spatial-only padding). Anything else (channel pad, batch pad)
//     bails out — those would need a different lowering anyway.
//   * pad_const matches the conv's `input_zp` (so the padded region is the
//     same value the conv would have inserted itself). If it doesn't match,
//     we leave pad alone — the standalone op might still work depending on
//     hardware.
struct PadFoldIntoConv : public OpRewritePattern<tosa::PadOp> {
  using OpRewritePattern<tosa::PadOp>::OpRewritePattern;

  // Read the rank-2 padding shape ([rank, 2] of front/back per dim).
  // Returns nullopt if the shape isn't a const_shape we can read.
  static std::optional<SmallVector<int64_t>>
  readPaddingPairs(tosa::PadOp op) {
    auto shapeOp = op.getPadding().getDefiningOp<tosa::ConstShapeOp>();
    if (!shapeOp) return std::nullopt;
    auto attr = dyn_cast<DenseIntElementsAttr>(shapeOp.getValuesAttr());
    if (!attr) return std::nullopt;
    SmallVector<int64_t> v;
    for (APInt i : attr.getValues<APInt>()) v.push_back(i.getSExtValue());
    return v;
  }

  LogicalResult matchAndRewrite(tosa::PadOp op,
                                 PatternRewriter &rewriter) const final {
    if (!op->hasOneUse())
      return failure();
    Operation *user = *op->getUsers().begin();

    auto pairs = readPaddingPairs(op);
    if (!pairs || pairs->size() != 8)
      return failure(); // only the NHWC rank-4 case is supported here.
    // NHWC: pairs = [N front/back, H f/b, W f/b, C f/b].
    if ((*pairs)[0] != 0 || (*pairs)[1] != 0 ||
        (*pairs)[6] != 0 || (*pairs)[7] != 0)
      return failure(); // non-spatial padding can't be folded into conv/pool.

    int64_t hFront = (*pairs)[2], hBack = (*pairs)[3];
    int64_t wFront = (*pairs)[4], wBack = (*pairs)[5];

    if (auto conv = dyn_cast<tosa::Conv2DOp>(user)) {
      // Conv only consumes the pad as `input` (operand 0).
      if (op.getResult() != conv.getInput())
        return failure();
      auto cur = conv.getPadAttr().asArrayRef();
      if (cur.size() != 4) return failure();
      auto newPad = rewriter.getDenseI64ArrayAttr(
          {cur[0] + hFront, cur[1] + hBack,
           cur[2] + wFront, cur[3] + wBack});
      conv.setPadAttr(newPad);
      conv->setOperand(0, op.getInput1());
      rewriter.eraseOp(op);
      return success();
    }
    if (auto pool = dyn_cast<tosa::MaxPool2dOp>(user)) {
      auto cur = pool.getPadAttr().asArrayRef();
      if (cur.size() != 4) return failure();
      auto newPad = rewriter.getDenseI64ArrayAttr(
          {cur[0] + hFront, cur[1] + hBack,
           cur[2] + wFront, cur[3] + wBack});
      pool.setPadAttr(newPad);
      pool->setOperand(0, op.getInput1());
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

// Skip the i32 detour in TOSA's quantize-tail decomposition. The chain
// TOSA's rescale lowering emits is:
//
//   %a   = tosa.mul %x, %inv_scale_f32, %shift   : f32   (1/output_scale)
//   %b   = tosa.cast %a   : f32 → i32
//   %z   = tosa.const dense<zp_int>              : i32   (rank-0 / broadcast)
//   %c   = tosa.add %b, %z                       : i32
//   %out = tosa.cast %c   : i32 → i8 (or u8/i16) : narrow int
//
// On VIP9000Nano-DI plain int32 is a near-dead column (Add/Sub/Mul/Pow/Rcp,
// Reshape/Slice, Pool/Conv/Matmul/FC all COMPILE_FAIL — see the i32
// column of the matrix in project_a733_fp32_conv_unsupported.md). The
// equivalent fp32 form runs end-to-end on supported kernels:
//
//   %z_f = tosa.const dense<float(zp_int)>       : f32   (same shape as %z)
//   %c_f = tosa.add %a, %z_f                     : f32
//   %out = tosa.cast %c_f : f32 → narrow int
//
// `Cast f32 → i8|asym` does saturate(round(input)), which equals the
// pre-existing chain's behavior because %a was already produced as
// `(s * 1/scale)`: adding zp in fp32 then truncating gives the same
// narrow-int values as casting to i32, adding zp, then truncating.
//
// We also recover the (output_scale, output_zp) pair here and stash
// them on the new `tosa.cast` as discardable `timvx.output_scale` /
// `timvx.output_zp` attributes. `buildQuantInfoMap` reads those attrs
// so `CastConversion` can attach the right Quantization() to the
// downstream tensor spec — without them, the i8 result feeding into
// the next conv2d would have no quant info on the operand spec.
struct RequantI32SkipFold : public OpRewritePattern<tosa::CastOp> {
  using OpRewritePattern<tosa::CastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::CastOp castOut,
                                 PatternRewriter &rewriter) const final {
    auto srcTy = dyn_cast<RankedTensorType>(castOut.getInput().getType());
    auto dstTy = dyn_cast<RankedTensorType>(castOut.getType());
    if (!srcTy || !dstTy) return failure();
    if (!srcTy.getElementType().isInteger(32)) return failure();
    auto dstInt = dyn_cast<IntegerType>(dstTy.getElementType());
    if (!dstInt || dstInt.getWidth() >= 32) return failure();

    auto add = castOut.getInput().getDefiningOp<tosa::AddOp>();
    if (!add || !add->hasOneUse()) return failure();

    // Identify the (cast-from-f32-to-i32) operand and the (i32 zp const)
    // operand of the inner add.
    auto isI32CastFromF32 = [](Value v, tosa::CastOp &out) {
      auto c = v.getDefiningOp<tosa::CastOp>();
      if (!c) return false;
      auto in = dyn_cast<RankedTensorType>(c.getInput().getType());
      auto outT = dyn_cast<RankedTensorType>(c.getType());
      if (!in || !outT) return false;
      if (!in.getElementType().isF32()) return false;
      if (!outT.getElementType().isInteger(32)) return false;
      out = c;
      return true;
    };
    tosa::CastOp innerCast;
    Value zpSide;
    if (isI32CastFromF32(add.getInput1(), innerCast)) zpSide = add.getInput2();
    else if (isI32CastFromF32(add.getInput2(), innerCast)) zpSide = add.getInput1();
    else return failure();
    if (!innerCast->hasOneUse()) return failure();

    // Read zp (must be constant; broadcast-shaped is fine — we'll reuse the
    // shape for the f32 replacement).
    auto zpConst = zpSide.getDefiningOp<tosa::ConstOp>();
    if (!zpConst) return failure();
    auto zpAttr = dyn_cast<DenseIntElementsAttr>(zpConst.getValuesAttr());
    if (!zpAttr || !zpAttr.getElementType().isInteger(32)) return failure();
    auto zpTy = dyn_cast<RankedTensorType>(zpConst.getType());
    if (!zpTy || zpAttr.getNumElements() < 1) return failure();
    int64_t zpScalar =
        (*zpAttr.getValues<APInt>().begin()).getSExtValue();

    // Read the upstream multiplier const (the `1/output_scale` factor) so
    // we can derive the output scale that downstream conv2d expects on
    // its input. The mul operand can be either input — try both.
    auto mul = innerCast.getInput().getDefiningOp<tosa::MulOp>();
    std::optional<double> invScale;
    Value mulValue;
    if (mul) {
      auto tryConst = [&](Value v) -> std::optional<double> {
        auto view = getConstF32(v);
        if (!view || view->data.empty()) return std::nullopt;
        return static_cast<double>(view->data[0]);
      };
      if ((invScale = tryConst(mul.getInput2()))) {
        mulValue = mul.getInput1();
      } else if ((invScale = tryConst(mul.getInput1()))) {
        mulValue = mul.getInput2();
      }
    }
    // The peephole's correctness doesn't depend on recovering invScale —
    // we'll still rewrite the chain. We only skip the `output_scale`
    // attribute when invScale isn't available, which leaves the cast
    // result untagged and falls back to the existing CastConversion
    // path's quant-less spec. (Practically every chain we lower has the
    // const operand, so this should always succeed.)
    (void)mulValue;

    // Build f32 const with the same shape; value-cast each i32 element to
    // f32. zp is small so the cast is exact.
    SmallVector<float> zpFp;
    zpFp.reserve(zpAttr.getNumElements());
    for (APInt v : zpAttr.getValues<APInt>())
      zpFp.push_back(static_cast<float>(v.getSExtValue()));
    auto f32 = rewriter.getF32Type();
    auto zpFpTy = RankedTensorType::get(zpTy.getShape(), f32);
    auto zpFpConst = tosa::ConstOp::create(
        rewriter, zpConst.getLoc(), zpFpTy,
        DenseElementsAttr::get(zpFpTy, ArrayRef<float>(zpFp)));

    // f32 add (same shape as the original i32 add result).
    auto i32AddTy = dyn_cast<RankedTensorType>(add.getType());
    if (!i32AddTy) return failure();
    auto fAddTy = RankedTensorType::get(i32AddTy.getShape(), f32);
    auto newAdd = tosa::AddOp::create(rewriter, add.getLoc(), fAddTy,
                                       innerCast.getInput(),
                                       zpFpConst.getResult());

    // Final cast: replaces the outer i32 → narrow-int cast with a f32 →
    // narrow-int cast. Stash output_scale / output_zp as discardable
    // attrs so buildQuantInfoMap can pick them up.
    auto newCast = tosa::CastOp::create(rewriter, castOut.getLoc(), dstTy,
                                         newAdd.getResult());
    if (invScale && *invScale != 0.0) {
      double scale = 1.0 / *invScale;
      newCast->setAttr("timvx.output_scale", rewriter.getF64FloatAttr(scale));
      newCast->setAttr("timvx.output_zp",
                        rewriter.getI64IntegerAttr(zpScalar));
    }
    rewriter.replaceOp(castOut, newCast.getResult());
    return success();
  }
};

}; // namespace

//===----------------------------------------------------------------------===//
// Pass driver: tosa-const-fold
//===----------------------------------------------------------------------===//

struct TosaConstFoldPass
    : public impl::TosaConstFoldPassBase<TosaConstFoldPass> {
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<AddConstFold, SubConstFold, MulConstFold, PowConstFold,
                 ReciprocalConstFold, ReshapeConstFold, PadFoldIntoConv,
                 RequantI32SkipFold>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createTosaConstFoldPass() {
  return std::make_unique<TosaConstFoldPass>();
}

//===----------------------------------------------------------------------===//
// tosa-fold-avgpool-reduce
//===----------------------------------------------------------------------===//
//
// Replace `cast(scale * sum(x)/N + zp)` (the global avg-pool emitted by
// tflite as a `reduce_sum × 2 → mul(1/N) → requant tail`) with
// `avg_pool2d(cast(scale * x + zp))` so the average runs as a single
// Pool2D AVG kernel on u8 instead of 14 unique slice/add shaders. See
// the pass description in TIMVXPasses.td for why this is needed.

namespace {

// Read a scalar fp32 value from a `tosa.const`. Used to recognize the
// 1/N, scale, and zp constants in the requant chain.
static std::optional<double> getConstFp32Scalar(Value v) {
  auto c = v.getDefiningOp<tosa::ConstOp>();
  if (!c) return std::nullopt;
  auto attr = dyn_cast<DenseFPElementsAttr>(c.getValuesAttr());
  if (!attr) return std::nullopt;
  if (!attr.getElementType().isF32()) return std::nullopt;
  if (attr.isSplat())
    return attr.getSplatValue<APFloat>().convertToDouble();
  if (attr.getNumElements() == 1)
    return (*attr.getValues<APFloat>().begin()).convertToDouble();
  return std::nullopt;
}

// Pick out the (variable, scalar-const) operand split for a binary op.
// Returns {variable_value, scalar_double} on success.
template <typename BinOp>
static std::optional<std::pair<Value, double>>
splitBinaryConstOperand(BinOp op) {
  Value a = op.getInput1(), b = op.getInput2();
  if (auto s = getConstFp32Scalar(b)) return {{a, *s}};
  if (auto s = getConstFp32Scalar(a)) return {{b, *s}};
  return std::nullopt;
}

// Match a tosa.cast f32 → narrow_int that carries the
// `timvx.output_scale` / `timvx.output_zp` discardable attrs deposited
// by `RequantI32SkipFold` — that's the requant-tail's terminal cast.
// On match, walk back through:
//   (cast_terminal) ← add(zp_const) ← mul(scale_const) ← mul(1/N const)
//   ← reshape ← reduce_sum(W) ← reduce_sum(H) ← <fp32 spatial input>
//
// and rewrite to:
//   <fp32 spatial> → mul(scale) → add(zp) → cast f32→int → avg_pool2d
//   → reshape (back to the cast_terminal's original output shape).
struct AvgPoolReduceFold : public OpRewritePattern<tosa::CastOp> {
  using OpRewritePattern<tosa::CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::CastOp castOp,
                                 PatternRewriter &rewriter) const final {
    // (1) cast must be the post-RequantI32SkipFold terminal cast: f32 → narrow int
    //     with our discardable quant attrs.
    auto srcTy = dyn_cast<RankedTensorType>(castOp.getInput().getType());
    auto dstTy = dyn_cast<RankedTensorType>(castOp.getType());
    if (!srcTy || !dstTy) return failure();
    if (!srcTy.getElementType().isF32()) return failure();
    auto outI = dyn_cast<IntegerType>(dstTy.getElementType());
    if (!outI || outI.getWidth() >= 32) return failure();
    auto castScale = castOp->getAttrOfType<FloatAttr>("timvx.output_scale");
    auto castZp = castOp->getAttrOfType<IntegerAttr>("timvx.output_zp");
    if (!castScale || !castZp) return failure();

    // Helper: a chain link is "single-use" iff the producing op's only
    // consumer is the next link. Bail otherwise — pulling the requant
    // forward would change semantics for the other consumer.
    auto onlyUseIs = [](Value v, Operation *user) {
      return v.hasOneUse() && *v.getUsers().begin() == user;
    };

    // (2) cast input ← tosa.add %y, zp_const (splat fp32)
    auto add = castOp.getInput().getDefiningOp<tosa::AddOp>();
    if (!add) return failure();
    if (!onlyUseIs(add.getResult(), castOp)) return failure();
    auto addSplit = splitBinaryConstOperand<tosa::AddOp>(add);
    if (!addSplit) return failure();
    auto [addVar, zpVal] = *addSplit;

    // (3) addVar ← tosa.mul %z, scale_const (splat fp32)
    auto mulScale = addVar.getDefiningOp<tosa::MulOp>();
    if (!mulScale) return failure();
    if (!onlyUseIs(mulScale.getResult(), add)) return failure();
    auto mulScaleSplit = splitBinaryConstOperand<tosa::MulOp>(mulScale);
    if (!mulScaleSplit) return failure();
    auto [mulScaleVar, scaleVal] = *mulScaleSplit;

    // (4) mulScaleVar ← tosa.mul %w, inv_n_const (splat fp32 = 1/N)
    auto mulInvN = mulScaleVar.getDefiningOp<tosa::MulOp>();
    if (!mulInvN) return failure();
    if (!onlyUseIs(mulInvN.getResult(), mulScale)) return failure();
    auto mulInvNSplit = splitBinaryConstOperand<tosa::MulOp>(mulInvN);
    if (!mulInvNSplit) return failure();
    auto [mulInvNVar, invNVal] = *mulInvNSplit;
    if (invNVal <= 0.0) return failure();
    int64_t expectedN = static_cast<int64_t>(std::llround(1.0 / invNVal));
    if (expectedN <= 1) return failure();
    // Sanity-check: 1/N const should round-trip.
    if (std::abs(invNVal - 1.0 / static_cast<double>(expectedN)) > 1e-6)
      return failure();

    // (5) Optional reshape (drops singleton dims). Walk through any
    //     chain of single-use reshapes.
    Value cur = mulInvNVar;
    Operation *lastConsumer = mulInvN;
    while (auto rs = cur.getDefiningOp<tosa::ReshapeOp>()) {
      if (!onlyUseIs(rs.getResult(), lastConsumer)) return failure();
      cur = rs.getInput1();
      lastConsumer = rs;
    }

    // (6) reduce_sum chain (one or two ops, axes H/W in NHWC).
    int64_t kernelH = 1, kernelW = 1;
    SmallVector<tosa::ReduceSumOp> reduces;
    while (auto rs = cur.getDefiningOp<tosa::ReduceSumOp>()) {
      if (!onlyUseIs(rs.getResult(), lastConsumer)) return failure();
      auto inT = dyn_cast<RankedTensorType>(rs.getInput().getType());
      if (!inT || inT.getRank() != 4) return failure();
      int axis = static_cast<int>(rs.getAxis());
      // NHWC: dim 1 = H, dim 2 = W. We don't fold reductions along N or C.
      if (axis == 1) kernelH *= inT.getDimSize(1);
      else if (axis == 2) kernelW *= inT.getDimSize(2);
      else return failure();
      reduces.push_back(rs);
      cur = rs.getInput();
      lastConsumer = rs;
    }
    if (reduces.empty()) return failure();
    if (kernelH * kernelW != expectedN) return failure();

    // (7) `cur` is now the spatial fp32 tensor that the reduce_sum
    //     chain consumes. Verify shape; rebuild the chain.
    auto spatialTy = dyn_cast<RankedTensorType>(cur.getType());
    if (!spatialTy || spatialTy.getRank() != 4 ||
        !spatialTy.getElementType().isF32())
      return failure();
    auto spatialShape = spatialTy.getShape();
    int64_t batch = spatialShape[0];
    int64_t H = spatialShape[1], W = spatialShape[2], C = spatialShape[3];
    if (H != kernelH || W != kernelW) return failure();

    Location loc = castOp.getLoc();
    Type fp32 = rewriter.getF32Type();
    Type narrowInt = dstTy.getElementType();

    auto splat4D = [&](double v) {
      auto ty = RankedTensorType::get({1, 1, 1, 1}, fp32);
      auto attr = DenseElementsAttr::get(ty, static_cast<float>(v));
      return tosa::ConstOp::create(rewriter, loc, ty, attr).getResult();
    };

    // Per-pixel rescale: y = scale * x + zp (still fp32, full spatial).
    Value perPixelMul = tosa::MulOp::create(
        rewriter, loc, RankedTensorType::get(spatialShape, fp32),
        cur, splat4D(scaleVal),
        /*shift=*/mulScale.getShift());  // reuse the original mul's shift const
    Value perPixelAdd = tosa::AddOp::create(
        rewriter, loc, RankedTensorType::get(spatialShape, fp32),
        perPixelMul, splat4D(zpVal));
    Value perPixelInt = tosa::CastOp::create(
        rewriter, loc, RankedTensorType::get(spatialShape, narrowInt),
        perPixelAdd);
    // Carry the same quant tags as the original cast — buildQuantInfoMap
    // will pick them up so downstream conv/FC sees the right input quant.
    perPixelInt.getDefiningOp()->setAttr("timvx.output_scale", castScale);
    perPixelInt.getDefiningOp()->setAttr("timvx.output_zp", castZp);

    // tosa.avg_pool2d: kernel = [H, W], stride = [1,1], pad = [0,0,0,0].
    // Both input_zp and output_zp are passed as scalar tensor<1xT> consts
    // matching the per-pixel cast's storage type and (i8) zp.
    auto zpTensorTy = RankedTensorType::get({1}, narrowInt);
    auto i8ZpVal = static_cast<int8_t>(castZp.getInt());
    auto inZpAttr = DenseElementsAttr::get(zpTensorTy, ArrayRef<int8_t>{i8ZpVal});
    Value inZp = tosa::ConstOp::create(rewriter, loc, zpTensorTy, inZpAttr);
    Value outZp = tosa::ConstOp::create(rewriter, loc, zpTensorTy, inZpAttr);

    auto poolOutTy = RankedTensorType::get({batch, 1, 1, C}, narrowInt);
    auto kernelAttr = rewriter.getDenseI64ArrayAttr({kernelH, kernelW});
    auto strideAttr = rewriter.getDenseI64ArrayAttr({1, 1});
    auto padAttr = rewriter.getDenseI64ArrayAttr({0, 0, 0, 0});
    Value pooled = tosa::AvgPool2dOp::create(
        rewriter, loc, poolOutTy, perPixelInt, inZp, outZp,
        kernelAttr, strideAttr, padAttr,
        /*acc_type=*/TypeAttr::get(rewriter.getI32Type()));

    // Reshape pool result back to the original cast result's shape
    // (typically `[batch, C]` after the user's intermediate reshapes).
    auto finalTy = castOp.getType();
    auto finalShape = cast<RankedTensorType>(finalTy).getShape();
    auto shapeTy = RankedTensorType::get(
        {static_cast<int64_t>(finalShape.size())}, rewriter.getIndexType());
    SmallVector<APInt> shapeInts;
    shapeInts.reserve(finalShape.size());
    for (int64_t d : finalShape)
      shapeInts.emplace_back(64, static_cast<uint64_t>(d), /*isSigned=*/true);
    Value shapeConst = tosa::ConstShapeOp::create(
        rewriter, loc,
        tosa::shapeType::get(rewriter.getContext(),
                              static_cast<int64_t>(finalShape.size())),
        DenseIntElementsAttr::get(shapeTy, shapeInts));
    Value reshaped = tosa::ReshapeOp::create(rewriter, loc, finalTy,
                                               pooled, shapeConst);

    rewriter.replaceOp(castOp, reshaped);
    return success();
  }
};

struct TosaFoldAvgPoolReducePass
    : public impl::TosaFoldAvgPoolReducePassBase<TosaFoldAvgPoolReducePass> {
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<AvgPoolReduceFold>(&getContext());
    GreedyRewriteConfig cfg;
    // Same reasoning as TIMVXConv1x1ToFCPass: greedy folding +
    // constant-CSE re-materializes constants through the dialect's
    // `materializeConstant` hook, which can drop the discardable
    // `timvx.output_scale` / `timvx.output_zp` attrs we depend on.
    cfg.enableFolding(false);
    cfg.enableConstantCSE(false);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), cfg)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createTosaFoldAvgPoolReducePass() {
  return std::make_unique<TosaFoldAvgPoolReducePass>();
}

//===----------------------------------------------------------------------===//
// tosa-layout-tag (driver below)
//===----------------------------------------------------------------------===//
//
// Forward dataflow analysis. Anchors at TOSA spatial ops (per-spec layouts:
// conv2d input NHWC + weight OHWI; pool2d input NHWC), propagates through
// elementwise ops with broadcast-aware layout derivation, and writes the
// inferred layout to each tagged op as a `timvx.layout` StringAttr (or
// `func.func` arg attr).
//
// Doesn't propagate across `tosa.transpose` (per design — transposes are
// left for `--tosa-reduce-transposes` to coalesce later) or across rank-
// changing ops (`tosa.reshape`, `tosa.matmul`, `tosa.fully_connected`),
// which become natural boundaries for the downstream WHCN-canonicalize.
//
// Conflict (a value tagged with two different layouts) is reported as an
// error on the producing op.

namespace {

enum class AxisRole : uint8_t {
  N, H, W, C, D,    // logical NHWC / NDHWC activation dims
  Oc, Ic,           // OHWI / IcWHOc weight dims
  Broadcast         // size-1 dim that broadcasts (rendered as "1")
};

using Layout = SmallVector<AxisRole, 5>;

static StringRef axisRoleName(AxisRole r) {
  switch (r) {
  case AxisRole::N:         return "N";
  case AxisRole::H:         return "H";
  case AxisRole::W:         return "W";
  case AxisRole::C:         return "C";
  case AxisRole::D:         return "D";
  case AxisRole::Oc:        return "Oc";
  case AxisRole::Ic:        return "Ic";
  case AxisRole::Broadcast: return "1";
  }
  llvm_unreachable("unhandled AxisRole");
}

static std::string layoutToString(const Layout &l) {
  std::string s;
  llvm::raw_string_ostream os(s);
  for (size_t i = 0; i < l.size(); ++i) {
    if (i) os << ',';
    os << axisRoleName(l[i]);
  }
  return os.str();
}

// Number of named (non-Broadcast) roles in a layout — the "info content"
// used to pick the broadcast target between multiple operands.
static int infoCount(const Layout &l) {
  int n = 0;
  for (auto r : l)
    if (r != AxisRole::Broadcast)
      ++n;
  return n;
}

static constexpr StringLiteral kLayoutAttr = "timvx.layout";

// In-memory dataflow state. Live for one pass invocation.
struct LayoutInferenceCtx {
  MLIRContext *ctx;
  DenseMap<Value, Layout> layouts;
  SmallVector<Value, 64> worklist;
  bool failed = false;

  // Record `l` as the layout of `v`. Returns true on success (including the
  // already-tagged-with-same-layout case), false on conflict (and emits an
  // error on the producing op or function).
  bool tag(Value v, Layout l) {
    auto it = layouts.find(v);
    if (it == layouts.end()) {
      layouts.try_emplace(v, std::move(l));
      worklist.push_back(v);
      return true;
    }
    if (it->second == l)
      return true;
    InFlightDiagnostic diag = errorAt(v);
    diag << "tosa-layout-tag: conflicting layouts for an SSA value: "
         << "existing '" << layoutToString(it->second)
         << "' vs new '" << layoutToString(l)
         << "' (one tensor used as two layout-disagreeing roles)";
    failed = true;
    return false;
  }

  InFlightDiagnostic errorAt(Value v) {
    if (Operation *def = v.getDefiningOp())
      return def->emitError();
    if (auto bArg = dyn_cast<BlockArgument>(v))
      return bArg.getOwner()->getParentOp()->emitError();
    return mlir::emitError(UnknownLoc::get(ctx));
  }
};

// Derive the per-axis layout of a value `v` participating in a broadcast
// op whose result shape is `targetShape` and target layout is `primary`.
// Right-aligns v's shape against `targetShape`. Rule per dim:
//   - v's size matches target's size  → inherit primary's role at that pos
//                                       (covers main operands where N=1
//                                       happens to align with target's N=1)
//   - v's size is 1 (and differs)     → Broadcast (the dim actually
//                                       expands at runtime)
//   - otherwise                       → return empty (incompatible)
// Left-padded positions (when v's rank < primary's) are Broadcast.
static Layout deriveBroadcastLayout(const Layout &primary,
                                    ArrayRef<int64_t> targetShape, Value v) {
  auto rt = dyn_cast<RankedTensorType>(v.getType());
  if (!rt || rt.getRank() > (int64_t)primary.size() ||
      targetShape.size() != primary.size())
    return {};
  Layout out;
  out.reserve(primary.size());
  size_t pad = primary.size() - rt.getRank();
  for (size_t i = 0; i < pad; ++i)
    out.push_back(AxisRole::Broadcast);
  for (int64_t i = 0; i < rt.getRank(); ++i) {
    int64_t s = rt.getDimSize(i);
    int64_t t = targetShape[pad + i];
    if (s == t)
      out.push_back(primary[pad + i]);
    else if (s == 1)
      out.push_back(AxisRole::Broadcast);
    else
      return {};
  }
  return out;
}

// Pick the most-informative known layout among `values` to use as the
// broadcast target. Prefers more named roles; tiebreaker is higher rank.
static Layout pickPrimary(ArrayRef<Value> values,
                          const LayoutInferenceCtx &ctx) {
  Layout best;
  int bestInfo = -1;
  for (Value v : values) {
    auto it = ctx.layouts.find(v);
    if (it == ctx.layouts.end())
      continue;
    int info = infoCount(it->second);
    if (info > bestInfo ||
        (info == bestInfo && it->second.size() > best.size())) {
      best = it->second;
      bestInfo = info;
    }
  }
  return best;
}

// Anchor tags from TOSA's spatial ops (where the spec fixes the layouts).
// Returns true if `op` matched a known anchor (used purely for early-out
// during the initial walk; the worklist drives propagation either way).
static bool tagAnchor(Operation *op, LayoutInferenceCtx &ctx) {
  Layout nhwc{AxisRole::N, AxisRole::H, AxisRole::W, AxisRole::C};
  Layout ohwi{AxisRole::Oc, AxisRole::H, AxisRole::W, AxisRole::Ic};
  Layout obias{AxisRole::Oc};
  if (auto conv = dyn_cast<tosa::Conv2DOp>(op)) {
    ctx.tag(conv.getInput(),  nhwc);
    ctx.tag(conv.getWeight(), ohwi);
    ctx.tag(conv.getBias(),   obias);
    ctx.tag(conv.getResult(), nhwc);
    return true;
  }
  if (auto pool = dyn_cast<tosa::MaxPool2dOp>(op)) {
    ctx.tag(pool.getInput(),  nhwc);
    ctx.tag(pool.getResult(), nhwc);
    return true;
  }
  if (auto pool = dyn_cast<tosa::AvgPool2dOp>(op)) {
    ctx.tag(pool.getInput(),  nhwc);
    ctx.tag(pool.getResult(), nhwc);
    return true;
  }
  return false;
}

// Apply the layout transfer function for `op`. May tag previously-untagged
// operands or the result. No-op for ops that aren't recognized layout-
// passthroughs (which become natural boundaries).
static void propagateThrough(Operation *op, LayoutInferenceCtx &ctx) {
  // Binary elementwise with broadcast: result and both operands share
  // the broadcast target layout (operands may individually be Broadcast
  // for dims where their size differs from the result's).
  auto tagBinary = [&](Value lhs, Value rhs, Value result) {
    auto rTy = dyn_cast<RankedTensorType>(result.getType());
    if (!rTy || !rTy.hasStaticShape())
      return;
    SmallVector<Value, 3> vs{lhs, rhs, result};
    Layout primary = pickPrimary(vs, ctx);
    if (primary.empty() || primary.size() != (size_t)rTy.getRank())
      return;
    for (Value v : vs) {
      if (ctx.layouts.contains(v))
        continue;
      Layout l = deriveBroadcastLayout(primary, rTy.getShape(), v);
      if (!l.empty())
        ctx.tag(v, l);
    }
  };
  if (isa<tosa::AddOp, tosa::SubOp, tosa::PowOp>(op) &&
      op->getNumOperands() == 2 && op->getNumResults() == 1) {
    tagBinary(op->getOperand(0), op->getOperand(1), op->getResult(0));
    return;
  }
  // Mul has a third `shift` operand which is layout-irrelevant; restrict
  // primary-picking and tagging to the two data operands and the result.
  if (auto mul = dyn_cast<tosa::MulOp>(op)) {
    tagBinary(mul.getInput1(), mul.getInput2(), mul.getResult());
    return;
  }
  // Unary same-shape elementwise: input and result share layout.
  if (auto clamp = dyn_cast<tosa::ClampOp>(op)) {
    SmallVector<Value, 2> vs{clamp.getInput(), clamp.getResult()};
    Layout primary = pickPrimary(vs, ctx);
    if (primary.empty())
      return;
    for (Value v : vs)
      if (!ctx.layouts.contains(v))
        ctx.tag(v, primary);
    return;
  }
  if (auto rcp = dyn_cast<tosa::ReciprocalOp>(op)) {
    SmallVector<Value, 2> vs{rcp.getInput1(), rcp.getResult()};
    Layout primary = pickPrimary(vs, ctx);
    if (primary.empty())
      return;
    for (Value v : vs)
      if (!ctx.layouts.contains(v))
        ctx.tag(v, primary);
    return;
  }
  // tosa.rescale: per-element requantize. The data input + result share
  // layout; the multiplier / shift / zp scalar consts are layout-irrelevant.
  // Without this propagation, the layout-to-whcn pass would insert a
  // boundary transpose between a conv2d and its rescale, which then breaks
  // RescaleConvFusion's `rescale.input.defOp<tosa.conv2d>` match.
  if (auto resc = dyn_cast<tosa::RescaleOp>(op)) {
    SmallVector<Value, 2> vs{resc.getInput(), resc.getResult()};
    Layout primary = pickPrimary(vs, ctx);
    if (primary.empty())
      return;
    for (Value v : vs)
      if (!ctx.layouts.contains(v))
        ctx.tag(v, primary);
    return;
  }
  // tosa.cast: dtype-only conversion (i8 → f32 dequant cast or f32 → i32 → i8
  // requant cast in the residual path). Same-shape, single-input, so it's a
  // layout passthrough.
  if (auto cast = dyn_cast<tosa::CastOp>(op)) {
    SmallVector<Value, 2> vs{cast.getInput(), cast.getResult()};
    Layout primary = pickPrimary(vs, ctx);
    if (primary.empty())
      return;
    for (Value v : vs)
      if (!ctx.layouts.contains(v))
        ctx.tag(v, primary);
    return;
  }
  // tosa.slice: shape-changing in extent only — per-axis role is
  // preserved (W stays W, H stays H, etc.), just the dim sizes shrink.
  // Layout passes through input → result. The start/size const operands
  // are not feature maps and are tagged separately by the
  // layout-to-whcn pass. Without this, slice's input would be untagged
  // and Phase 5 would insert a boundary NHWC→WHCN transpose at the
  // slice's input — exactly the pattern we want to avoid.
  if (auto slice = dyn_cast<tosa::SliceOp>(op)) {
    SmallVector<Value, 2> vs{slice.getInput1(), slice.getResult()};
    Layout primary = pickPrimary(vs, ctx);
    if (primary.empty())
      return;
    for (Value v : vs)
      if (!ctx.layouts.contains(v))
        ctx.tag(v, primary);
    return;
  }
  // Other ops: layout transfer not defined — boundary.
}

} // namespace

struct TosaLayoutTagPass
    : public impl::TosaLayoutTagPassBase<TosaLayoutTagPass> {
  void runOnOperation() final {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    LayoutInferenceCtx infer;
    infer.ctx = ctx;

    // 1. Anchor at spatial ops.
    mod.walk([&](Operation *op) { (void)tagAnchor(op, infer); });
    if (infer.failed)
      return signalPassFailure();

    // 2. Propagate to fixed point. Both directions: a freshly-tagged value
    //    informs its defining op's siblings (forward) and its users
    //    (backward); we enqueue once per tag so each op's transfer
    //    function gets re-evaluated as more of its operands become known.
    while (!infer.worklist.empty()) {
      Value v = infer.worklist.pop_back_val();
      if (Operation *def = v.getDefiningOp())
        propagateThrough(def, infer);
      for (Operation *user : v.getUsers())
        propagateThrough(user, infer);
      if (infer.failed)
        return signalPassFailure();
    }

    // 3. Materialize tags as MLIR attributes for inspection / downstream
    //    consumption. Single-result ops only for now (covers all of TOSA's
    //    spatial / elementwise / const surface).
    for (auto &kv : infer.layouts) {
      Value v = kv.first;
      auto attr = StringAttr::get(ctx, layoutToString(kv.second));
      if (Operation *def = v.getDefiningOp()) {
        if (def->getNumResults() == 1)
          def->setAttr(kLayoutAttr, attr);
      } else if (auto bArg = dyn_cast<BlockArgument>(v)) {
        if (auto func = dyn_cast<func::FuncOp>(bArg.getOwner()->getParentOp()))
          func.setArgAttr(bArg.getArgNumber(), kLayoutAttr, attr);
      }
    }
  }
};

std::unique_ptr<Pass> createTosaLayoutTagPass() {
  return std::make_unique<TosaLayoutTagPass>();
}

//===----------------------------------------------------------------------===//
// tosa-layout-to-whcn (driver below)
//===----------------------------------------------------------------------===//
//
// Consumes the `timvx.layout` tags set by --tosa-layout-tag and physically
// rewrites the IR so each tagged tensor is in TIM-VX's WHCN convention.
//
//   * Constants     → data permuted at compile time + type updated.
//   * Op results    → type updated in place (the ops themselves are
//                     layout-agnostic; broadcast / pointwise semantics carry
//                     through any consistent shape).
//   * Spatial ops   → also swap kernel / stride / pad / dilation arrays
//                     from TOSA's [H, W] / [top, bot, left, right] to
//                     WHCN's [W, H] / [left, right, top, bot].
//   * Boundaries    → insert a tosa.transpose to bridge the layouts.
//                     `--canonicalize` then folds the entry-side transpose
//                     against the front-end's NCHW→NHWC transpose.
//
// Caveat: tosa.conv2d / max_pool2d / avg_pool2d's verifiers expect NHWC-
// flavored shape relationships, which fail after this pass. Run mlir-opt
// with `--no-verify-each` (lower_sample.py threads this through) so the
// failure can't fire between this pass and --tosa-to-timvx, which lowers
// the now-WHCN tosa ops to timvx.* (no NHWC requirement).

namespace {

// Where each named axis role lives in the WHCN ordering.
//   W=0, H=1, C/Ic=2, N/Oc=3.  Broadcasts fill any unfilled slot.
//   D (3D depth) is intentionally rejected; 5D layouts aren't handled yet.
static std::optional<int64_t> whcnPosition(AxisRole r) {
  switch (r) {
  case AxisRole::W:                    return 0;
  case AxisRole::H:                    return 1;
  case AxisRole::C: case AxisRole::Ic: return 2;
  case AxisRole::N: case AxisRole::Oc: return 3;
  case AxisRole::D:                    return -1;
  case AxisRole::Broadcast:            return std::nullopt;
  }
  llvm_unreachable("axis role");
}

// Compute perm such that `result_dim[i] = source_dim[perm[i]]` to reorder
// a tensor with `layout` into WHCN. Named roles take their canonical
// positions; broadcasts fill remaining slots in source order. Returns
// empty if the layout is malformed (collisions or D in rank-4).
static SmallVector<int64_t> computeWhcnPerm(const Layout &layout) {
  size_t rank = layout.size();
  SmallVector<int64_t> perm(rank, -1);
  for (size_t i = 0; i < rank; ++i) {
    auto pos = whcnPosition(layout[i]);
    if (!pos)
      continue;
    if (*pos < 0 || *pos >= (int64_t)rank || perm[*pos] != -1)
      return {};
    perm[*pos] = (int64_t)i;
  }
  size_t cursor = 0;
  for (size_t i = 0; i < rank; ++i) {
    if (layout[i] != AxisRole::Broadcast)
      continue;
    while (cursor < rank && perm[cursor] != -1)
      ++cursor;
    if (cursor >= rank)
      return {};
    perm[cursor++] = (int64_t)i;
  }
  return perm;
}

// inv[perm[i]] = i.
static SmallVector<int64_t> invertPerm(ArrayRef<int64_t> perm) {
  SmallVector<int64_t> inv(perm.size(), -1);
  for (size_t i = 0; i < perm.size(); ++i)
    inv[perm[i]] = (int64_t)i;
  return inv;
}

static SmallVector<int64_t> applyPermToShape(ArrayRef<int64_t> shape,
                                              ArrayRef<int64_t> perm) {
  SmallVector<int64_t> out;
  out.reserve(perm.size());
  for (int64_t p : perm)
    out.push_back(shape[p]);
  return out;
}

// Parse "N,H,W,C" → Layout. Inverse of layoutToString.
static std::optional<Layout> parseLayoutStr(StringRef s) {
  Layout out;
  while (!s.empty()) {
    auto comma = s.find(',');
    StringRef tok = comma == StringRef::npos ? s : s.take_front(comma);
    if      (tok == "N")  out.push_back(AxisRole::N);
    else if (tok == "H")  out.push_back(AxisRole::H);
    else if (tok == "W")  out.push_back(AxisRole::W);
    else if (tok == "C")  out.push_back(AxisRole::C);
    else if (tok == "D")  out.push_back(AxisRole::D);
    else if (tok == "Oc") out.push_back(AxisRole::Oc);
    else if (tok == "Ic") out.push_back(AxisRole::Ic);
    else if (tok == "1")  out.push_back(AxisRole::Broadcast);
    else                  return std::nullopt;
    s = comma == StringRef::npos ? StringRef{} : s.drop_front(comma + 1);
  }
  return out;
}

// Compute the permutation `outIndex -> inIndex` for a row-major buffer
// reshape from `inShape` to `outShape = applyPerm(inShape, perm)`. Returns
// a vector of size product(inShape).
static SmallVector<int64_t>
computePermutationIndices(ArrayRef<int64_t> inShape, ArrayRef<int64_t> perm,
                          ArrayRef<int64_t> outShape) {
  int64_t rank = inShape.size();
  int64_t numel = 1;
  for (int64_t d : inShape) numel *= d;
  SmallVector<int64_t> inStrides(rank), outStrides(rank);
  inStrides[rank - 1] = 1;
  outStrides[rank - 1] = 1;
  for (int64_t i = rank - 2; i >= 0; --i) {
    inStrides[i] = inStrides[i + 1] * inShape[i + 1];
    outStrides[i] = outStrides[i + 1] * outShape[i + 1];
  }
  SmallVector<int64_t> indices(numel);
  for (int64_t outLin = 0; outLin < numel; ++outLin) {
    int64_t rem = outLin;
    int64_t inLin = 0;
    for (int64_t i = 0; i < rank; ++i) {
      int64_t coord = rem / outStrides[i];
      rem -= coord * outStrides[i];
      inLin += coord * inStrides[perm[i]];
    }
    indices[outLin] = inLin;
  }
  return indices;
}

// Permute the values of a dense / dense-resource attribute according to
// `perm`. Handles f32, i8, i16, i32 storage; returns nullptr otherwise.
//
// Used by `--tosa-layout-to-whcn` to permute compile-time constants
// (conv weights from OHWI→WHIcOc, FC weights, etc.). Quantized models add
// i8 weights and i32 biases to the dtype set we need to support.
static DenseElementsAttr permuteDenseElements(ElementsAttr in,
                                              ArrayRef<int64_t> perm) {
  auto inTy = dyn_cast<RankedTensorType>(in.getType());
  if (!inTy || !inTy.hasStaticShape())
    return nullptr;
  ArrayRef<int64_t> inShape = inTy.getShape();
  SmallVector<int64_t> outShape = applyPermToShape(inShape, perm);
  auto outTy = RankedTensorType::get(outShape, inTy.getElementType());

  Type elem = inTy.getElementType();
  int64_t numel = inTy.getNumElements();

  // Precompute outLin -> inLin once; reused across the typed branches.
  auto idx = computePermutationIndices(inShape, perm, outShape);

  // Float path (covers BatchNorm const-fold + FP32 weights).
  if (elem.isF32()) {
    SmallVector<float> src;
    src.reserve(numel);
    if (auto dense = dyn_cast<DenseElementsAttr>(in)) {
      for (APFloat v : dense.getValues<APFloat>())
        src.push_back(v.convertToFloat());
    } else if (auto r = dyn_cast<DenseF32ResourceElementsAttr>(in)) {
      auto arr = r.tryGetAsArrayRef();
      if (!arr) return nullptr;
      src.assign(arr->begin(), arr->end());
    } else {
      return nullptr;
    }
    SmallVector<float> out(numel);
    for (int64_t i = 0; i < numel; ++i) out[i] = src[idx[i]];
    return DenseElementsAttr::get(outTy, ArrayRef<float>(out));
  }

  // Integer path (i8 weights, i16 / i32 biases, ...). The DenseElementsAttr
  // builder accepts ArrayRef<APInt> for any integer width, so we materialize
  // a single APInt vector regardless of width.
  if (auto it = dyn_cast<IntegerType>(elem)) {
    unsigned bits = it.getWidth();
    SmallVector<APInt> src;
    src.reserve(numel);

    if (auto dense = dyn_cast<DenseElementsAttr>(in)) {
      for (APInt v : dense.getValues<APInt>()) src.push_back(v);
    } else if (auto r = dyn_cast<DenseI8ResourceElementsAttr>(in)) {
      auto arr = r.tryGetAsArrayRef();
      if (!arr) return nullptr;
      for (int8_t v : *arr) src.emplace_back(bits, uint64_t(v), /*signed=*/true);
    } else if (auto r = dyn_cast<DenseI16ResourceElementsAttr>(in)) {
      auto arr = r.tryGetAsArrayRef();
      if (!arr) return nullptr;
      for (int16_t v : *arr) src.emplace_back(bits, uint64_t(v), true);
    } else if (auto r = dyn_cast<DenseI32ResourceElementsAttr>(in)) {
      auto arr = r.tryGetAsArrayRef();
      if (!arr) return nullptr;
      for (int32_t v : *arr) src.emplace_back(bits, uint64_t(v), true);
    } else if (auto r = dyn_cast<DenseUI8ResourceElementsAttr>(in)) {
      auto arr = r.tryGetAsArrayRef();
      if (!arr) return nullptr;
      for (uint8_t v : *arr) src.emplace_back(bits, uint64_t(v), false);
    } else {
      return nullptr;
    }
    SmallVector<APInt> out(numel, APInt(bits, 0));
    for (int64_t i = 0; i < numel; ++i) out[i] = src[idx[i]];
    return DenseElementsAttr::get(outTy, out);
  }

  return nullptr;
}

// Swap dim 0 and dim 1 of a 2-element attr ([H, W] → [W, H]).
static DenseI64ArrayAttr swapHW(DenseI64ArrayAttr a, OpBuilder &b) {
  auto v = a.asArrayRef();
  if (v.size() != 2) return a;
  return b.getDenseI64ArrayAttr({v[1], v[0]});
}
// Swap pair (0,1) with (2,3): [top, bot, left, right] → [left, right, top, bot].
static DenseI64ArrayAttr swapPadPairs(DenseI64ArrayAttr a, OpBuilder &b) {
  auto v = a.asArrayRef();
  if (v.size() != 4) return a;
  return b.getDenseI64ArrayAttr({v[2], v[3], v[0], v[1]});
}

// True iff `op` is a TOSA spatial op whose attrs need swapping.
static bool isSpatialOp(Operation *op) {
  return isa<tosa::Conv2DOp, tosa::MaxPool2dOp, tosa::AvgPool2dOp>(op);
}

// True iff `op` is one whose result type is safe to permute in place to
// the WHCN-permuted shape. Mirrors the set of ops whose layout transfer
// function is defined in `--tosa-layout-tag` (anchors + binary/unary
// elementwise passthroughs). Non-passthrough ops (slice, reshape, ...)
// may carry a tagged result purely because a downstream anchor-input
// pointed at them; their op semantics are NHWC-shape-bound, so retyping
// would silently break them. Phase 5 handles those via boundary
// transposes at the anchor operand instead.
static bool isLayoutPermutableOp(Operation *op) {
  return isa<tosa::Conv2DOp, tosa::MaxPool2dOp, tosa::AvgPool2dOp,
             tosa::AddOp, tosa::SubOp, tosa::MulOp, tosa::PowOp,
             tosa::ClampOp, tosa::ReciprocalOp,
             // Quant graph: rescale (i32→i8 requantize) and cast (dtype
             // conversion) are per-element; their result types can be
             // safely re-shaped via the same WHCN permutation as their
             // input.
             tosa::RescaleOp, tosa::CastOp,
             // Slice with WHCN-permuted start/size operands. The
             // permutation of the start/size const values happens in
             // the layout-to-whcn pass's Phase 2.5 (slice-specific
             // operand permutation). Without making slice permutable
             // here, the layout pass would insert NHWC↔WHCN boundary
             // transposes around the slice — and that transpose-slice-
             // transpose chain on u8 tensors stalls TIM-VX's runtime
             // command queue under graph-execution pressure (10s NPU
             // watchdog → vxProcessGraph returns failure with no log).
             // Specifically observed on ResNet18 stages 2/3/4 where
             // the projection-shortcut downsample's TOSA expression
             // routes through `transpose → slice (1px crop) →
             // transpose → 1x1 stride-2 conv`. Stage 1 has identity
             // shortcuts so the chain doesn't appear.
             tosa::SliceOp>(op);
}

// Pretty bridging inserter: insert a tosa.transpose just before `useOp`,
// reading from `value` with the given `perm`, and rewire `*operand` to
// the new transpose's result.
static void insertTransposeBefore(OpOperand *operand, Value value,
                                  ArrayRef<int64_t> perm) {
  Operation *useOp = operand->getOwner();
  OpBuilder b(useOp);
  auto rt = cast<RankedTensorType>(value.getType());
  auto outShape = applyPermToShape(rt.getShape(), perm);
  auto outTy = RankedTensorType::get(outShape, rt.getElementType());
  SmallVector<int32_t> permI32(perm.begin(), perm.end());
  auto t = tosa::TransposeOp::create(b, useOp->getLoc(), outTy, value,
                                      b.getDenseI32ArrayAttr(permI32));
  operand->set(t.getResult());
}

} // namespace

struct TosaLayoutToWhcnPass
    : public impl::TosaLayoutToWhcnPassBase<TosaLayoutToWhcnPass> {
  void runOnOperation() final {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();

    // Phase 1: collect tags (as Layouts) from MLIR attrs.
    DenseMap<Value, Layout> layouts;
    mod.walk([&](Operation *op) {
      auto attr = op->getAttrOfType<StringAttr>(kLayoutAttr);
      if (attr && op->getNumResults() == 1)
        if (auto l = parseLayoutStr(attr.getValue()))
          layouts[op->getResult(0)] = std::move(*l);
    });
    mod.walk([&](func::FuncOp f) {
      for (unsigned i = 0; i < f.getNumArguments(); ++i) {
        auto attr = f.getArgAttrOfType<StringAttr>(i, kLayoutAttr);
        if (attr)
          if (auto l = parseLayoutStr(attr.getValue()))
            layouts[f.getArgument(i)] = std::move(*l);
      }
    });

    // Phase 2: permute tagged constants (in place). Note we walk-and-mutate;
    // tosa::ConstOp setters accept the new attr/type cleanly.
    SmallVector<tosa::ConstOp> failedConsts;
    mod.walk([&](tosa::ConstOp constOp) {
      auto it = layouts.find(constOp.getResult());
      if (it == layouts.end()) return;
      auto perm = computeWhcnPerm(it->second);
      if (perm.empty() || perm.size() <= 1) return;
      auto values = dyn_cast<ElementsAttr>(constOp.getValuesAttr());
      if (!values) return;
      auto newAttr = permuteDenseElements(values, perm);
      if (!newAttr) {
        failedConsts.push_back(constOp);
        return;
      }
      constOp.setValuesAttr(newAttr);
      constOp.getResult().setType(cast<TensorType>(newAttr.getType()));
    });
    if (!failedConsts.empty()) {
      for (auto c : failedConsts)
        c.emitError("tosa-layout-to-whcn: cannot permute (non-FP32 / non-static)");
      return signalPassFailure();
    }

    // Phase 2.5: permute tosa.slice's start/size const operands.
    //
    // tosa.slice has two operand-form 1-D index vectors (`start` /
    // `size`) whose i-th entry indexes the i-th axis of the slice's
    // input. When the input is being permuted from NHWC to WHCN, those
    // operand vectors must be permuted by the SAME index permutation
    // (i.e. WHCN_start[i] = NHWC_start[perm[i]] where perm is the
    // WHCN-from-NHWC perm) — otherwise the slice would crop the wrong
    // axes. The const_shape ops backing these operands are tiny 1-D
    // index tensors and not "feature maps", so Phase 2's layout-tag-
    // driven const permutation skips them.
    //
    // Without this phase the start/size would still address NHWC dims
    // while the input is now in WHCN — output values would be garbage
    // and the runtime composition would mis-route memory reads.
    SmallVector<tosa::SliceOp> failedSlices;
    mod.walk([&](tosa::SliceOp slice) {
      auto it = layouts.find(slice.getResult());
      if (it == layouts.end()) return;
      auto perm = computeWhcnPerm(it->second);
      if (perm.empty() || perm.size() <= 1) return;

      auto permuteShapeOperand = [&](Value operand) -> bool {
        auto cs = operand.getDefiningOp<tosa::ConstShapeOp>();
        if (!cs) return false;
        auto attr = dyn_cast<DenseIntElementsAttr>(cs.getValuesAttr());
        if (!attr) return false;
        if (attr.getNumElements() != static_cast<int64_t>(perm.size()))
          return false;

        // Reuse the original APInts (they carry their own bitwidth, so
        // we don't need to ask the element type — it's `index`, which
        // isn't IntOrFloat and rejects getIntOrFloatBitWidth()).
        SmallVector<APInt> orig;
        orig.reserve(attr.getNumElements());
        for (APInt v : attr.getValues<APInt>())
          orig.push_back(v);
        SmallVector<APInt> permuted(perm.size());
        for (size_t i = 0; i < perm.size(); ++i)
          permuted[i] = orig[perm[i]];

        auto newAttr = DenseIntElementsAttr::get(
            cast<RankedTensorType>(attr.getType()), permuted);
        cs.setValuesAttr(newAttr);
        return true;
      };

      bool ok = permuteShapeOperand(slice.getStart()) &&
                permuteShapeOperand(slice.getSize());
      if (!ok) failedSlices.push_back(slice);
    });
    if (!failedSlices.empty()) {
      for (auto s : failedSlices)
        s.emitError("tosa-layout-to-whcn: slice start/size operands are "
                    "not const_shape; layout-permutable slice requires "
                    "compile-time-known indices");
      return signalPassFailure();
    }

    // Phase 3: update tagged op-result types -- but only for ops whose
    // semantics are layout-agnostic (anchors + elementwise passthroughs).
    // For non-passthrough ops (slice, reshape, ...) whose result was tagged
    // only because a downstream anchor consumed it, leave the type alone;
    // Phase 5 will insert a boundary transpose at the anchor's operand.
    mod.walk([&](Operation *op) {
      if (op->getNumResults() != 1) return;
      if (!isLayoutPermutableOp(op)) return;
      Value r = op->getResult(0);
      auto it = layouts.find(r);
      if (it == layouts.end()) return;
      auto perm = computeWhcnPerm(it->second);
      if (perm.empty() || perm.size() <= 1) return;
      auto rt = dyn_cast<RankedTensorType>(r.getType());
      if (!rt || !rt.hasStaticShape()) return;
      r.setType(RankedTensorType::get(applyPermToShape(rt.getShape(), perm),
                                      rt.getElementType()));
    });

    // Phase 4: swap kernel/stride/pad/dilation on tagged spatial ops.
    mod.walk([&](Operation *op) {
      if (!isSpatialOp(op)) return;
      if (!layouts.contains(op->getResult(0))) return;
      OpBuilder b(op);
      if (auto conv = dyn_cast<tosa::Conv2DOp>(op)) {
        conv.setStrideAttr(swapHW(conv.getStrideAttr(), b));
        conv.setDilationAttr(swapHW(conv.getDilationAttr(), b));
        conv.setPadAttr(swapPadPairs(conv.getPadAttr(), b));
      } else if (auto pool = dyn_cast<tosa::MaxPool2dOp>(op)) {
        pool.setKernelAttr(swapHW(pool.getKernelAttr(), b));
        pool.setStrideAttr(swapHW(pool.getStrideAttr(), b));
        pool.setPadAttr(swapPadPairs(pool.getPadAttr(), b));
      } else if (auto pool = dyn_cast<tosa::AvgPool2dOp>(op)) {
        pool.setKernelAttr(swapHW(pool.getKernelAttr(), b));
        pool.setStrideAttr(swapHW(pool.getStrideAttr(), b));
        pool.setPadAttr(swapPadPairs(pool.getPadAttr(), b));
      }
    });

    // Phase 5: boundary transposes.
    //
    // For each operand (consumed_value, consumer):
    //   * If consumed_value is tagged and its type *matches* the WHCN
    //     perm-shape: it's a "real WHCN" value.
    //       - If consumer is untagged → exit boundary; insert WHCN→original
    //         transpose so the consumer sees its expected NHWC shape.
    //   * If consumed_value is tagged but type *doesn't* match WHCN (e.g.,
    //     a tosa.transpose result we deliberately left alone): treat type
    //     as "still NHWC."
    //       - If consumer is tagged → entry boundary; insert NHWC→WHCN.
    //
    // We snapshot operand pointers first (mutating the IR mid-walk would
    // invalidate the iteration).
    SmallVector<OpOperand *> exitOperands; // WHCN value → untagged consumer
    SmallVector<OpOperand *> entryOperands; // non-WHCN tagged value → tagged consumer
    auto isWhcnTyped = [&](Value v, const Layout &l) -> bool {
      auto rt = dyn_cast<RankedTensorType>(v.getType());
      if (!rt) return false;
      auto perm = computeWhcnPerm(l);
      if (perm.empty()) return false;
      // The value's type was permuted to WHCN by Phase 2 (consts) or
      // Phase 3 (layout-permutable ops). Anything else (transpose, slice,
      // reshape, func arg, ...) still carries an NHWC-shape type, so an
      // entry boundary transpose is needed at the anchor's operand.
      Operation *def = v.getDefiningOp();
      if (!def) return false; // function arg
      return isa<tosa::ConstOp>(def) || isLayoutPermutableOp(def);
    };
    mod.walk([&](Operation *op) {
      // Skip the freshly-inserted boundary transposes themselves.
      if (isa<tosa::TransposeOp>(op) && !layouts.contains(op->getResult(0)))
        return;
      for (OpOperand &operand : op->getOpOperands()) {
        Value v = operand.get();
        auto it = layouts.find(v);
        if (it == layouts.end())
          continue;
        bool valueIsWhcn = isWhcnTyped(v, it->second);
        // "In-region" consumer = both layout-permutable *and* its result
        // is tagged. A merely-tagged result on a non-permutable op (e.g.
        // tosa.pad whose result was tagged by a downstream conv anchor)
        // doesn't make the consumer in-region — pad's op semantics need
        // an NHWC-shaped input and a boundary transpose, even though the
        // tag chain reaches its result.
        bool consumerInRegion = isLayoutPermutableOp(op) &&
                                 op->getNumResults() == 1 &&
                                 layouts.contains(op->getResult(0));
        if (valueIsWhcn && !consumerInRegion)
          exitOperands.push_back(&operand);
        else if (!valueIsWhcn && consumerInRegion)
          entryOperands.push_back(&operand);
        // (WHCN value → tagged consumer: in-region, no transpose.
        //  Non-WHCN value → untagged consumer: also no transpose.)
      }
    });
    // Insert the bridging transposes.
    for (OpOperand *o : exitOperands) {
      Value v = o->get();
      Layout l = layouts[v];
      auto perm = computeWhcnPerm(l);
      if (perm.empty()) continue;
      // WHCN → original (inverse of WHCN perm).
      insertTransposeBefore(o, v, invertPerm(perm));
    }
    for (OpOperand *o : entryOperands) {
      Value v = o->get();
      Layout l = layouts[v];
      auto perm = computeWhcnPerm(l);
      if (perm.empty()) continue;
      // Original → WHCN.
      insertTransposeBefore(o, v, perm);
    }

    // Phase 6: strip the layout attrs (intent is now embedded in shapes).
    mod.walk([&](Operation *op) {
      if (op->hasAttr(kLayoutAttr))
        op->removeAttr(kLayoutAttr);
    });
    mod.walk([&](func::FuncOp f) {
      for (unsigned i = 0; i < f.getNumArguments(); ++i)
        f.removeArgAttr(i, kLayoutAttr);
    });
    (void)ctx;
  }
};

std::unique_ptr<Pass> createTosaLayoutToWhcnPass() {
  return std::make_unique<TosaLayoutToWhcnPass>();
}

//===----------------------------------------------------------------------===//
// Pass driver: tosa-to-timvx
//===----------------------------------------------------------------------===//

struct TosaToTIMVXPass : public impl::TosaToTIMVXPassBase<TosaToTIMVXPass> {
  void runOnOperation() final {
    ConversionTarget target(getContext());
    target.addLegalDialect<TIMVXDialect>();

    target.addIllegalDialect<tosa::TosaDialect>();

    // A `tosa.conv2d` that's the producer of a `tosa.rescale` will be
    // subsumed by RescaleConvFusion (which erases the conv after fusing).
    // Marking it dynamically legal here lets the driver defer it to the
    // rescale match instead of failing on it via the FP Conv2DOpConversion
    // (which match-fails on non-zero input_zp).
    target.addDynamicallyLegalOp<tosa::Conv2DOp>([](tosa::Conv2DOp op) {
      return op->hasOneUse() &&
             isa<tosa::RescaleOp>(*op->getUsers().begin());
    });

    // target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    // Pre-walk to derive (scale, zp) for every quantized SSA value. The
    // rescale-conv fusion pattern queries this so it doesn't have to
    // re-derive scales for each instance.
    QuantInfoMap qmap;
    getOperation().walk(
        [&](func::FuncOp f) { buildQuantInfoMap(f, qmap); });

    RewritePatternSet patterns(&getContext());
    patterns.add<
        // Tensor-only passthroughs (no attrs to translate).
        TensorOnlyOpConversion<tosa::PowOp, PowOp>,
        TensorOnlyOpConversion<tosa::ReciprocalOp, RcpOp>,
        TensorOnlyOpConversion<tosa::AddOp, AddOp>,
        TensorOnlyOpConversion<tosa::SubOp, SubOp>,
        // Bespoke (attribute / operand translation).
        ClampOpConversion, MulOpConversion, MatMulOpConversion,
        ConstOpConversion, ConstShapeOpConversion,
        Conv2DOpConversion,
        ReduceSumConversion>(&getContext());
    patterns.add<RescaleConvFusion, CastConversion, PadConversion,
                 MaxPool2DConversion, AvgPool2DConversion,
                 ReshapeOpConversion, SliceOpConversion,
                 TransposeConversion>(&getContext(), qmap);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createTosaToTIMVXPass() {
  return std::make_unique<TosaToTIMVXPass>();
}

//===----------------------------------------------------------------------===//
// timvx-conv1x1-to-fc
//===----------------------------------------------------------------------===//
//
// Pattern: when a `timvx.conv2d` has W=H=1 input, W=H=1 weight, and unit
// pad/stride/dilation, it's mathematically a fully-connected layer. Routes
// through `timvx.fully_connected` (which has a tighter NN-engine path on
// Vivante NPUs — the weight is pre-tiled at bind vs. Conv2D's per-call
// spatial setup). Triggered by tflite's quantized exporter, which emits
// the trailing classifier as a 1x1 Conv2D + reshape pair.

namespace {

// Match a 1x1 timvx.conv2d and rewrite to reshape→fully_connected→reshape.
struct Conv1x1ToFC : public OpRewritePattern<Conv2DOp> {
  using OpRewritePattern<Conv2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Conv2DOp op,
                                 PatternRewriter &rewriter) const final {
    auto inTy = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto wTy  = dyn_cast<RankedTensorType>(op.getWeight().getType());
    auto outTy = dyn_cast<RankedTensorType>(op.getType());
    if (!inTy || !wTy || !outTy) return failure();
    if (inTy.getRank() != 4 || wTy.getRank() != 4 || outTy.getRank() != 4)
      return failure();

    // TIM-VX inner-first convention. Input: WHCN. Weight: WHIcOc. Output: WHCN.
    auto inS = inTy.getShape();
    auto wS  = wTy.getShape();
    auto oS  = outTy.getShape();
    if (inS[0] != 1 || inS[1] != 1) return failure();      // input W=H=1
    if (wS[0]  != 1 || wS[1]  != 1) return failure();      // weight W=H=1
    if (oS[0]  != 1 || oS[1]  != 1) return failure();      // output W=H=1
    int64_t K = inS[2], batch = inS[3];
    int64_t M = wS[3];
    if (wS[2] != K || oS[2] != M || oS[3] != batch) return failure();

    // Unit pad/stride/dilation only — anything else has no FC equivalent.
    auto pad = op.getPad();
    auto stride = op.getStride();
    auto dilation = op.getDilation();
    if (pad.size() != 4 || pad[0] != 0 || pad[1] != 0 ||
        pad[2] != 0 || pad[3] != 0)
      return failure();
    if (stride.size() != 2 || stride[0] != 1 || stride[1] != 1)
      return failure();
    if (dilation.size() != 2 || dilation[0] != 1 || dilation[1] != 1)
      return failure();

    // Weight must be a `timvx.const` so we can re-attach the dense data
    // to a rank-2 type (`reshape` is shape-only at the elements-attr level).
    auto wConst = op.getWeight().getDefiningOp<ConstOp>();
    if (!wConst) return failure();
    auto wAttr = dyn_cast<DenseElementsAttr>(wConst.getValuesAttr());
    if (!wAttr) return failure();

    Location loc = op.getLoc();
    Type elemTy = wTy.getElementType();

    // [1,1,K,M] → [K,M] — same memory order, just dropping the size-1
    // outer dims.
    auto newWTy = RankedTensorType::get({K, M}, elemTy);
    auto newWAttr = wAttr.reshape(newWTy);
    Value newWeight = ConstOp::create(rewriter, loc, newWTy, newWAttr,
                                       wConst.getQuantScaleAttr(),
                                       wConst.getQuantZpAttr());

    // Helper: build a `timvx.reshape` carrying explicit `output_scale`/
    // `output_zp` attrs so fmtTensorSpec can emit a fully-quantized spec
    // for the new transient. Without these the emitted runtime tensor
    // would land as a plain INT8 (no Quantization()), creating a
    // dtype-vs-quant mismatch with the surrounding u8 ops once the
    // i8|asym→u8 promotion fires.
    auto reshapeWithQuant = [&](Value src, ArrayRef<int64_t> newShape,
                                 FloatAttr s, IntegerAttr z) -> Value {
      auto srcTy = cast<RankedTensorType>(src.getType());
      auto shapeTy = RankedTensorType::get(
          {static_cast<int64_t>(newShape.size())}, rewriter.getIndexType());
      SmallVector<APInt> shapeInts;
      shapeInts.reserve(newShape.size());
      for (int64_t d : newShape)
        shapeInts.emplace_back(/*numBits=*/64,
                                /*val=*/static_cast<uint64_t>(d),
                                /*isSigned=*/true);
      auto shapeAttr = DenseIntElementsAttr::get(shapeTy, shapeInts);
      Value shapeConst = ConstShapeOp::create(rewriter, loc, shapeTy,
                                                shapeAttr);
      auto outRTy = RankedTensorType::get(newShape, srcTy.getElementType());
      return ReshapeOp::create(rewriter, loc, outRTy, src, shapeConst, s, z);
    };

    // Recover the input's (scale, zp) from its defining op's
    // `output_scale` / `output_zp` attrs (every quantized timvx op
    // carries them in this convention). If the input traces back to
    // something without them (e.g. a func arg, or a non-quant op),
    // fall through with null attrs — the emitted spec then has no
    // Quantization() and TIM-VX's auto-rewriter takes over.
    FloatAttr inScale;
    IntegerAttr inZp;
    if (auto *defOp = op.getInput().getDefiningOp()) {
      inScale = defOp->getAttrOfType<FloatAttr>("output_scale");
      inZp = defOp->getAttrOfType<IntegerAttr>("output_zp");
    }

    // Reshape input [1,1,K,batch] -> [batch,K]. The runtime helper for
    // FC (timvx_runtime::fully_connected) reads `out_features = output
    // shape.back()` and constructs the op with `axis=1` — i.e. it
    // expects the MLIR FC contract "input is rank-2 [batch, K], output
    // is [batch, N]". Match that convention here so the same helper
    // works for both the matmul→FC fast path and our Conv1x1ToFC.
    Value xR = reshapeWithQuant(op.getInput(), {batch, K}, inScale, inZp);

    // FC: [batch,K] x [K,M] + [M] -> [batch,M]. Output quant matches
    // the conv's (rescale-derived) output.
    auto fcOutTy = RankedTensorType::get({batch, M}, outTy.getElementType());
    Value fc = FullyConnectedOp::create(rewriter, loc, fcOutTy, xR, newWeight,
                                         op.getBias(),
                                         op.getOutputScaleAttr(),
                                         op.getOutputZpAttr());

    // Reshape FC result back to [1,1,M,batch] so downstream consumers
    // see the same type as the original Conv2D output. Quant matches FC's.
    Value restored = reshapeWithQuant(fc, oS, op.getOutputScaleAttr(),
                                       op.getOutputZpAttr());
    rewriter.replaceOp(op, restored);
    return success();
  }
};

struct TIMVXConv1x1ToFCPass
    : public impl::TIMVXConv1x1ToFCPassBase<TIMVXConv1x1ToFCPass> {
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<Conv1x1ToFC>(&getContext());
    // The greedy driver folds + CSE-merges ConstantLike ops by default.
    // For `timvx.const`, fold() returns the `values` attr only — the
    // default materializer then rebuilds `timvx.const` without the
    // optional `quant_scale` / `quant_zp` attrs, stripping every
    // weight/bias const's quant info and breaking downstream codegen.
    // Disable folding and constant-CSE for this pass: we don't need
    // them (Conv1x1ToFC matches op shapes, not values).
    GreedyRewriteConfig cfg;
    cfg.enableFolding(false);
    cfg.enableConstantCSE(false);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), cfg)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createTIMVXConv1x1ToFCPass() {
  return std::make_unique<TIMVXConv1x1ToFCPass>();
}

//===----------------------------------------------------------------------===//
//                            timvx -> emitc
//===----------------------------------------------------------------------===//
//
// Per-op lowerings emit one `emitc.call_opaque` against a runtime helper in
// `timvx_runtime::`. The helper header is supplied out-of-band; signatures
// follow the pattern
//
//   TensorPtr <op>(GraphPtr graph,
//                  <ssa input tensors...>,
//                  <constant attrs...>,
//                  tim::vx::TensorSpec output_spec);
//
// where `TensorPtr = std::shared_ptr<tim::vx::Tensor>` and
//       `GraphPtr  = std::shared_ptr<tim::vx::Graph>`.
//
// The graph itself is referenced as a free identifier `graph`; the surrounding
// function is expected to declare it (typically as the first parameter).
// Output spec / per-op constants are serialized as opaque C++ text and
// inlined at call sites. Constant data for `timvx.const` is left as a
// `nullptr` placeholder — wiring real model weights is a separate concern.

namespace {

constexpr StringRef kTensorCxx = "std::shared_ptr<tim::vx::Tensor>";
constexpr StringRef kShapeCxx = "std::vector<uint32_t>";
constexpr StringRef kGraphCxx = "std::shared_ptr<tim::vx::Graph>";
constexpr StringRef kRuntimeNs = "timvx_runtime::";

emitc::OpaqueType tensorOpaqueTy(MLIRContext *c) {
  return emitc::OpaqueType::get(c, kTensorCxx);
}
emitc::OpaqueType shapeOpaqueTy(MLIRContext *c) {
  return emitc::OpaqueType::get(c, kShapeCxx);
}
emitc::OpaqueType graphOpaqueTy(MLIRContext *c) {
  return emitc::OpaqueType::get(c, kGraphCxx);
}

// Decide whether an asymmetric int8 tensor should be emitted as u8 with
// zp shifted +128. On VIP9000Nano-DI the runtime's
// `_add_graph_dataconvert_for_int8` (vsi_nn_graph_optimization.c) auto-
// promotes every asym int8 IO tensor to u8|asym(zp+128) and inserts a
// DataConvert at the boundary; that auto-DataConvert COMPILE_FAILs at
// graph IO for non-trivial shapes (the `vivante.nn.tensorcopy` kernel
// rejects e.g. rank-2 1×1000 outputs, even though same-shape `u8↔i8`
// passes the per-pair probe). Pre-emitting all asym int8 specs as u8
// matches the form TIM-VX wants internally — internally consistent
// throughout, so the auto-rewriter sees no candidates and skips
// inserting the boundary DataConvert. Constants need their bytes
// XOR'd with 0x80 to compensate for the storage relabel.
//
// Cases we promote:
//   - Plain `i8` element type with explicit asym scale/zp on the
//     producing op (our convention: `output_scale`/`output_zp` on the
//     `timvx.*` op, or `quant_scale`/`quant_zp` on `timvx.const`).
//   - `quant.uniform<i8:f32, S:Z>` element type with signed i8 storage
//     (TOSA-encoded asym int8 — surfaces on tosa.const after import).
//
// Cases we leave alone:
//   - `i8` without asym quant (raw int8, no zp transform applies).
//   - `u8` (already u8).
//   - Per-channel int8 (the chip rejects per-channel weights anyway;
//     we don't have a clean per-channel→per-tensor remap here).
static bool shouldPromoteI8AsymToU8(Type elem, FloatAttr scaleOverride,
                                     IntegerAttr zpOverride) {
  if (auto i = dyn_cast<IntegerType>(elem))
    if (i.getWidth() == 8 && i.isSignless())
      return scaleOverride && zpOverride;
  if (auto quni = dyn_cast<quant::UniformQuantizedType>(elem))
    if (auto sty = dyn_cast<IntegerType>(quni.getStorageType()))
      return sty.getWidth() == 8 && quni.isSigned();
  return false;
}

// Render an MLIR element type as the matching tim::vx::DataType enum literal.
std::string tvxDataType(Type t) {
  // `!quant.uniform<i8:f32, S:Z>` and friends carry the storage type (i8/u8/...)
  // as their underlying integer; TIM-VX's DataType enum is keyed on storage.
  if (auto qt = dyn_cast<quant::QuantizedType>(t))
    return tvxDataType(qt.getStorageType());
  if (t.isF32())
    return "tim::vx::DataType::FLOAT32";
  if (t.isF16())
    return "tim::vx::DataType::FLOAT16";
  if (t.isInteger(1))
    return "tim::vx::DataType::BOOL8";
  if (auto it = dyn_cast<IntegerType>(t)) {
    bool u = it.isUnsigned();
    switch (it.getWidth()) {
    case 8:
      return u ? "tim::vx::DataType::UINT8" : "tim::vx::DataType::INT8";
    case 16:
      return u ? "tim::vx::DataType::UINT16" : "tim::vx::DataType::INT16";
    case 32:
      return u ? "tim::vx::DataType::UINT32" : "tim::vx::DataType::INT32";
    case 64:
      return "tim::vx::DataType::INT64";
    }
  }
  return "tim::vx::DataType::UNKNOWN";
}

// Format an int range as a C++ brace-init list "{a, b, c}".
template <typename Range> std::string fmtBraceList(Range &&r) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "{";
  llvm::interleaveComma(r, os);
  os << "}";
  return os.str();
}

// `std::array<uint32_t, N>{...}` literal for a small int array.
std::string fmtArray(ArrayRef<int64_t> v) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "std::array<uint32_t, " << v.size() << ">"
     << fmtBraceList(llvm::map_range(v, [](int64_t x) { return uint64_t(x); }));
  return os.str();
}

// True iff any of `op`'s results are consumed by a func.return. Used to
// decide whether the produced tensor should be a TIM-VX OUTPUT (so the
// host can CopyDataFromTensor on it) versus an internal TRANSIENT.
bool isFuncReturnedOp(Operation *op) {
  for (Value r : op->getResults())
    for (Operation *u : r.getUsers())
      if (isa<func::ReturnOp>(u))
        return true;
  return false;
}

// Map an MLIR scalar element type to its C++ name (used for the static
// array decl that backs each timvx.const).
StringRef cxxScalarType(Type t) {
  if (t.isF32())       return "float";
  if (t.isF64())       return "double";
  if (t.isF16())       return "uint16_t"; // raw FP16 storage
  if (t.isInteger(1))  return "uint8_t";
  if (auto it = dyn_cast<IntegerType>(t)) {
    bool u = it.isUnsigned();
    switch (it.getWidth()) {
    case 8:  return u ? "uint8_t"  : "int8_t";
    case 16: return u ? "uint16_t" : "int16_t";
    case 32: return u ? "uint32_t" : "int32_t";
    case 64: return "int64_t";
    }
  }
  return "char";
}

// Emit a static-array decl from a packed POD buffer. Used for DenseResource
// attrs where the data is stored as a raw blob.
template <typename T>
void writePodArray(llvm::raw_string_ostream &os, ArrayRef<T> values,
                   StringRef floatSuffix) {
  bool first = true;
  if constexpr (std::is_floating_point_v<T>) {
    char buf[32];
    for (T v : values) {
      os << (first ? "" : ", ");
      first = false;
      std::snprintf(buf, sizeof(buf), "%a", static_cast<double>(v));
      os << buf << floatSuffix;
    }
  } else {
    for (T v : values) {
      os << (first ? "" : ", ");
      first = false;
      os << static_cast<int64_t>(v);
    }
  }
}

// Render an ElementsAttr — inline `dense<...>` or `dense_resource<...>` —
// as `static const T name[N] = { v0, v1, ... };`. Returns "" if the storage
// or element type isn't one we handle.
//
// When `promoteI8ToU8` is true (caller's choice — driven by
// `shouldPromoteI8AsymToU8`), the array is emitted as `uint8_t[]` with
// each byte XOR'd 0x80, so reading the bytes as u8 with zp shifted +128
// gives the same real values as reading the original bytes as i8 with
// the original zp. The values must be backed by a signless i8 storage
// (inline `DenseElementsAttr` or `DenseI8ResourceElementsAttr`) — other
// storage shapes return "".
std::string fmtStaticArrayDecl(ElementsAttr values, StringRef name,
                                bool promoteI8ToU8 = false) {
  auto rt = cast<RankedTensorType>(values.getType());
  Type elem = rt.getElementType();
  // Rank-0 (scalar) tensors hold one value; the C++ array still needs a
  // size, so clamp to >= 1.
  uint64_t numel = std::max<uint64_t>(rt.getNumElements(), 1);

  std::string out;
  llvm::raw_string_ostream os(out);

  if (promoteI8ToU8) {
    os << "static const uint8_t " << name << "[" << numel << "] = {";
    bool first = true;
    auto emit = [&](uint8_t b) {
      os << (first ? "" : ", ") << static_cast<int>(b);
      first = false;
    };
    if (auto ints = values.tryGetValues<APInt>()) {
      for (APInt v : *ints) {
        // i8 byte → equivalent u8 byte (x XOR 0x80). The 8-bit truncation
        // is implicit in the cast-to-uint8 below.
        emit(static_cast<uint8_t>(v.getZExtValue()) ^ 0x80);
      }
    } else if (auto r = dyn_cast<DenseI8ResourceElementsAttr>(values)) {
      auto data = r.tryGetAsArrayRef();
      if (!data) return "";
      for (int8_t v : *data)
        emit(static_cast<uint8_t>(v) ^ 0x80);
    } else {
      return "";
    }
    os << "};";
    return out;
  }

  os << "static const " << cxxScalarType(elem) << " " << name << "[" << numel
     << "] = {";

  StringRef floatSuffix = elem.isF32() ? "f" : "";

  // Inline dense<...> attrs respond to the unified getValues API.
  if (auto floats = values.tryGetValues<APFloat>()) {
    char buf[32];
    bool first = true;
    for (APFloat v : *floats) {
      os << (first ? "" : ", ");
      first = false;
      // %a emits hex-float literal — bit-exact and always parses as a
      // valid C++ floating-point constant (avoids "0f" octal trap).
      std::snprintf(buf, sizeof(buf), "%a", v.convertToDouble());
      os << buf << floatSuffix;
    }
  } else if (auto ints = values.tryGetValues<APInt>()) {
    SmallString<32> s;
    bool first = true;
    bool isSigned = !cxxScalarType(elem).starts_with("u");
    for (APInt v : *ints) {
      os << (first ? "" : ", ");
      first = false;
      s.clear();
      v.toString(s, /*radix=*/10, isSigned);
      os << s;
    }
  }
  // dense_resource<...> attrs need typed access through their Base subclass.
  else if (auto r = dyn_cast<DenseF32ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return "";
    writePodArray<float>(os, *data, floatSuffix);
  } else if (auto r = dyn_cast<DenseF64ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return "";
    writePodArray<double>(os, *data, "");
  } else if (auto r = dyn_cast<DenseI8ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return "";
    writePodArray<int8_t>(os, *data, "");
  } else if (auto r = dyn_cast<DenseI16ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return "";
    writePodArray<int16_t>(os, *data, "");
  } else if (auto r = dyn_cast<DenseI32ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return "";
    writePodArray<int32_t>(os, *data, "");
  } else if (auto r = dyn_cast<DenseI64ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return "";
    writePodArray<int64_t>(os, *data, "");
  } else if (auto r = dyn_cast<DenseUI8ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return "";
    writePodArray<uint8_t>(os, *data, "");
  } else if (auto r = dyn_cast<DenseUI16ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return "";
    writePodArray<uint16_t>(os, *data, "");
  } else if (auto r = dyn_cast<DenseUI32ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return "";
    writePodArray<uint32_t>(os, *data, "");
  } else if (auto r = dyn_cast<DenseUI64ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return "";
    writePodArray<uint64_t>(os, *data, "");
  } else {
    return ""; // unsupported attr storage
  }

  os << "};";
  return out;
}

// Format a tensor's MLIR type as a `tim::vx::TensorSpec(...)` constructor
// expression. Attribute defaults to TRANSIENT (intermediate); callers can
// Format a double as a float literal that's always parseable by C++:
// `%.10g` alone can drop the decimal point (e.g. `1` for 1.0), which
// chains with the `f` suffix to an invalid token `1f`.
static std::string fmtFloatLiteral(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.10g", v);
  std::string s(buf);
  // If `%g` skipped the radix point and didn't emit an exponent, append
  // `.0` so the appended `f` suffix forms a well-formed float literal.
  bool hasDot = s.find_first_of(".eEpP") != std::string::npos;
  if (!hasDot) s += ".0";
  return s + "f";
}

// override (e.g. for constants).
//
// `scaleOverride` / `zpOverride`: when both set (e.g. populated from a
// timvx op's `quant_scale` / `quant_zp` attrs), the emitted spec carries
// `Quantization(ASYMMETRIC, scale, zp)`. Otherwise we fall back to the
// element type — `!quant.uniform<...>` types still emit Quantization()
// automatically. Plain integer/float types emit no quantization arg.
std::string fmtTensorSpec(Type ty, StringRef tensorAttr = "TRANSIENT",
                           FloatAttr scaleOverride = {},
                           IntegerAttr zpOverride = {}) {
  auto rt = cast<RankedTensorType>(ty);
  Type elem = rt.getElementType();
  std::string s;
  llvm::raw_string_ostream os(s);

  if (shouldPromoteI8AsymToU8(elem, scaleOverride, zpOverride)) {
    double scale;
    int64_t zp;
    if (scaleOverride && zpOverride) {
      scale = scaleOverride.getValueAsDouble();
      zp = zpOverride.getInt();
    } else {
      auto quni = cast<quant::UniformQuantizedType>(elem);
      scale = quni.getScale();
      zp = quni.getZeroPoint();
    }
    os << "tim::vx::TensorSpec(tim::vx::DataType::UINT8, " << kShapeCxx
       << fmtBraceList(rt.getShape())
       << ", tim::vx::TensorAttribute::" << tensorAttr
       << ", tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC, "
       << fmtFloatLiteral(scale) << ", " << (zp + 128) << "))";
    return os.str();
  }

  os << "tim::vx::TensorSpec(" << tvxDataType(elem) << ", "
     << kShapeCxx << fmtBraceList(rt.getShape())
     << ", tim::vx::TensorAttribute::" << tensorAttr;

  if (scaleOverride && zpOverride) {
    os << ", tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC, "
       << fmtFloatLiteral(scaleOverride.getValueAsDouble()) << ", "
       << zpOverride.getInt() << ")";
  } else if (auto quni = dyn_cast<quant::UniformQuantizedType>(elem)) {
    os << ", tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC, "
       << fmtFloatLiteral(quni.getScale()) << ", "
       << quni.getZeroPoint() << ")";
  } else if (auto qpa = dyn_cast<quant::UniformQuantizedPerAxisType>(elem)) {
    // Per-axis: TIM-VX expects scales / zps as std::vector + channel_dim.
    os << ", tim::vx::Quantization(tim::vx::QuantType::"
       << (llvm::all_of(qpa.getZeroPoints(), [](int64_t z) { return z == 0; })
               ? "SYMMETRIC_PER_CHANNEL"
               : "ASYMMETRIC_PER_CHANNEL")
       << ", " << qpa.getQuantizedDimension() << ", std::vector<float>{";
    llvm::interleaveComma(qpa.getScales(), os, [&](double s) {
      os << fmtFloatLiteral(s);
    });
    os << "}, std::vector<int32_t>{";
    llvm::interleaveComma(qpa.getZeroPoints(), os);
    os << "})";
  }

  os << ")";
  return os.str();
}

emitc::OpaqueAttr opq(MLIRContext *c, StringRef s) {
  return emitc::OpaqueAttr::get(c, s);
}

// Look up the `graph` SSA value — by convention the first argument of the
// enclosing func.func, prepended in the pre-pass below.
Value getGraphArg(Operation *op) {
  auto func = op->getParentOfType<func::FuncOp>();
  if (!func || func.getNumArguments() == 0)
    return {};
  Value first = func.getArgument(0);
  auto opaque = dyn_cast<emitc::OpaqueType>(first.getType());
  if (!opaque || opaque.getValue() != kGraphCxx)
    return {};
  return first;
}

// Replace `op` with one `emitc.call_opaque` to `timvx_runtime::<helperName>`,
// threading the implicit graph argument through as a real SSA operand. The
// emitted call is `helperName(graph, operands..., trailing_attrs...)`.
LogicalResult emitRuntimeCall(ConversionPatternRewriter &rewriter,
                              Operation *op, StringRef helperName,
                              ValueRange operands,
                              ArrayRef<Attribute> trailing,
                              Type resultType) {
  Value graph = getGraphArg(op);
  if (!graph)
    return rewriter.notifyMatchFailure(
        op, "enclosing func has no graph argument; pre-pass missed it");

  SmallVector<Value, 8> allOperands{graph};
  allOperands.append(operands.begin(), operands.end());

  SmallVector<Attribute, 8> args;
  for (size_t i = 0; i < allOperands.size(); ++i)
    args.push_back(rewriter.getIndexAttr(i));
  args.append(trailing.begin(), trailing.end());

  std::string callee = (kRuntimeNs + helperName).str();
  rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
      op, TypeRange{resultType}, callee, allOperands,
      rewriter.getArrayAttr(args), ArrayAttr{});
  return success();
}

//===----------------------------------------------------------------------===//
// Type converter
//===----------------------------------------------------------------------===//

// Maps:
//   tensor<Nxindex>      -> opaque<"std::vector<uint32_t>">  (shape vectors)
//   tensor<...>          -> opaque<"std::shared_ptr<tim::vx::Tensor>">
//   !timvx.graph         -> opaque<"std::shared_ptr<tim::vx::Graph>">
class TIMVXToEmitCTypeConverter : public TypeConverter {
public:
  TIMVXToEmitCTypeConverter(MLIRContext *ctx) {
    addConversion([](Type t) { return t; }); // identity fallback
    addConversion([ctx](RankedTensorType t) -> Type {
      if (isa<IndexType>(t.getElementType()))
        return shapeOpaqueTy(ctx);
      return tensorOpaqueTy(ctx);
    });
    addConversion(
        [ctx](GraphType) -> Type { return graphOpaqueTy(ctx); });

    // unrealized_conversion_cast for boundary mismatches.
    auto addCast = [](OpBuilder &b, Type t, ValueRange vs, Location loc) {
      return UnrealizedConversionCastOp::create(b, loc, t, vs).getResult(0);
    };
    addSourceMaterialization(addCast);
    addTargetMaterialization(addCast);
  }
};

//===----------------------------------------------------------------------===//
// Per-op patterns
//===----------------------------------------------------------------------===//

// timvx.const_shape -> emitc.constant {value = #emitc.opaque<"{...}">}
// We don't go through a runtime helper — a brace-init literal is enough.
struct ConstShapeToEmitC : public OpConversionPattern<ConstShapeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ConstShapeOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto attr = cast<DenseIntElementsAttr>(op.getValuesAttr());
    SmallVector<int64_t> vals;
    for (APInt v : attr.getValues<APInt>())
      vals.push_back(static_cast<int64_t>(v.getZExtValue()));
    std::string lit = (kShapeCxx + fmtBraceList(vals)).str();
    rewriter.replaceOpWithNewOp<emitc::ConstantOp>(
        op, shapeOpaqueTy(rewriter.getContext()),
        opq(rewriter.getContext(), lit));
    return success();
  }
};

// timvx.const -> timvx_runtime::const_tensor(graph, spec, /*data=*/nullptr).
// Real model weights are wired up in a follow-on step.
// timvx.const -> timvx_runtime::const_tensor(graph, spec, &<data>[0])
//
// The const op's data (Dense or DenseResource ElementsAttr) is reified
// into a function-local `static const T <name>[N] = { … };` declaration
// emitted before the call, and the call passes the array name (which
// decays to a const T*). This means weights are baked into the compiled
// binary — no separate weights file, no runtime parsing of the .mlir.
struct ConstToEmitC : public OpConversionPattern<ConstOp> {
  unsigned *counter;
  ConstToEmitC(const TypeConverter &tc, MLIRContext *ctx, unsigned *c)
      : OpConversionPattern<ConstOp>(tc, ctx), counter(c) {}
  LogicalResult
  matchAndRewrite(ConstOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto values = cast<ElementsAttr>(op.getValuesAttr());

    // i8|asym consts get bit-flipped (XOR 0x80) and emitted as uint8_t,
    // matching the u8|asym(zp+128) spec the matching fmtTensorSpec path
    // emits. See shouldPromoteI8AsymToU8 for the rationale.
    bool promote = shouldPromoteI8AsymToU8(
        cast<RankedTensorType>(op.getType()).getElementType(),
        op.getQuantScaleAttr(), op.getQuantZpAttr());

    std::string name = "_timvx_const_" + std::to_string((*counter)++);
    std::string decl = fmtStaticArrayDecl(values, name, promote);
    if (decl.empty())
      return rewriter.notifyMatchFailure(
          op, "unsupported element type / storage for constant data");

    // The static array decl precedes the helper call in the emitted source.
    emitc::VerbatimOp::create(rewriter, op.getLoc(),
                              rewriter.getStringAttr(decl));

    MLIRContext *ctx = rewriter.getContext();
    SmallVector<Attribute, 2> trailing{
        opq(ctx, fmtTensorSpec(op.getType(), "CONSTANT",
                                op.getQuantScaleAttr(),
                                op.getQuantZpAttr())),
        opq(ctx, name),
    };
    return emitRuntimeCall(rewriter, op, "const_tensor", /*operands=*/{},
                           trailing, tensorOpaqueTy(ctx));
  }
};

// Helper: pick OUTPUT vs TRANSIENT for the op's output_spec text.
inline StringRef outAttr(Operation *op) {
  return isFuncReturnedOp(op) ? "OUTPUT" : "TRANSIENT";
}

// Helper for ops whose only trailing arg is the output TensorSpec.
template <typename TIMVXOp>
struct SimpleRuntimeCall : public OpConversionPattern<TIMVXOp> {
  using OpConversionPattern<TIMVXOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<TIMVXOp>::OpAdaptor;
  StringRef helperName;
  SimpleRuntimeCall(const TypeConverter &tc, MLIRContext *ctx, StringRef name)
      : OpConversionPattern<TIMVXOp>(tc, ctx), helperName(name) {}
  LogicalResult
  matchAndRewrite(TIMVXOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *c = rewriter.getContext();
    SmallVector<Attribute, 1> trailing{
        opq(c, fmtTensorSpec(op.getType(), outAttr(op)))};
    return emitRuntimeCall(rewriter, op, helperName, adaptor.getOperands(),
                           trailing, tensorOpaqueTy(c));
  }
};

struct ClipToEmitC : public OpConversionPattern<ClipOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ClipOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *c = rewriter.getContext();
    auto fmtFloat = [](float v) {
      std::string s;
      llvm::raw_string_ostream(s) << v << "f";
      return s;
    };
    SmallVector<Attribute, 3> trailing{
        opq(c, fmtFloat(op.getMinVal().convertToFloat())),
        opq(c, fmtFloat(op.getMaxVal().convertToFloat())),
        opq(c, fmtTensorSpec(op.getType(), outAttr(op))),
    };
    return emitRuntimeCall(rewriter, op, "clip",
                           ValueRange{adaptor.getInput()}, trailing,
                           tensorOpaqueTy(c));
  }
};

struct Conv2DToEmitC : public OpConversionPattern<Conv2DOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(Conv2DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *c = rewriter.getContext();
    SmallVector<Attribute, 4> trailing{
        opq(c, fmtArray(op.getPad())),
        opq(c, fmtArray(op.getStride())),
        opq(c, fmtArray(op.getDilation())),
        opq(c, fmtTensorSpec(op.getType(), outAttr(op),
                              op.getOutputScaleAttr(),
                              op.getOutputZpAttr())),
    };
    return emitRuntimeCall(rewriter, op, "conv2d",
                           ValueRange{adaptor.getInput(), adaptor.getWeight(),
                                      adaptor.getBias()},
                           trailing, tensorOpaqueTy(c));
  }
};

struct Pool2DToEmitC : public OpConversionPattern<Pool2DOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(Pool2DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *c = rewriter.getContext();
    std::string poolEnum =
        ("tim::vx::PoolType::" + stringifyPoolType(op.getPoolType())).str();
    SmallVector<Attribute, 5> trailing{
        opq(c, poolEnum),
        opq(c, fmtArray(op.getKernel())),
        opq(c, fmtArray(op.getStride())),
        opq(c, fmtArray(op.getPad())),
        opq(c, fmtTensorSpec(op.getType(), outAttr(op),
                              op.getOutputScaleAttr(),
                              op.getOutputZpAttr())),
    };
    return emitRuntimeCall(rewriter, op, "pool2d",
                           ValueRange{adaptor.getInput()}, trailing,
                           tensorOpaqueTy(c));
  }
};

struct ReshapeToEmitC : public OpConversionPattern<ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *c = rewriter.getContext();
    SmallVector<Attribute, 1> trailing{
        opq(c, fmtTensorSpec(op.getType(), outAttr(op),
                              op.getOutputScaleAttr(),
                              op.getOutputZpAttr())),
    };
    return emitRuntimeCall(rewriter, op, "reshape",
                           ValueRange{adaptor.getInput1(), adaptor.getShape()},
                           trailing, tensorOpaqueTy(c));
  }
};

struct SliceToEmitC : public OpConversionPattern<SliceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *c = rewriter.getContext();
    SmallVector<Attribute, 1> trailing{
        opq(c, fmtTensorSpec(op.getType(), outAttr(op),
                              op.getOutputScaleAttr(),
                              op.getOutputZpAttr())),
    };
    return emitRuntimeCall(rewriter, op, "slice",
                           ValueRange{adaptor.getInput1(),
                                      adaptor.getStart(), adaptor.getSize()},
                           trailing, tensorOpaqueTy(c));
  }
};

struct TransposeToEmitC : public OpConversionPattern<TransposeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *c = rewriter.getContext();
    SmallVector<int64_t> perms(op.getPerms().begin(), op.getPerms().end());
    SmallVector<Attribute, 2> trailing{
        opq(c, kShapeCxx.str() + fmtBraceList(perms)),
        opq(c, fmtTensorSpec(op.getType(), outAttr(op),
                              op.getOutputScaleAttr(),
                              op.getOutputZpAttr())),
    };
    return emitRuntimeCall(rewriter, op, "transpose",
                           ValueRange{adaptor.getInput1()}, trailing,
                           tensorOpaqueTy(c));
  }
};

struct ReduceSumToEmitC : public OpConversionPattern<ReduceSumOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReduceSumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *c = rewriter.getContext();
    SmallVector<int64_t> axes(op.getAxes().begin(), op.getAxes().end());
    SmallVector<Attribute, 3> trailing{
        opq(c, "std::vector<int32_t>" + fmtBraceList(axes)),
        opq(c, op.getKeepDims() ? "true" : "false"),
        opq(c, fmtTensorSpec(op.getType(), outAttr(op))),
    };
    return emitRuntimeCall(rewriter, op, "reduce_sum",
                           ValueRange{adaptor.getInput()}, trailing,
                           tensorOpaqueTy(c));
  }
};

struct FullyConnectedToEmitC : public OpConversionPattern<FullyConnectedOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(FullyConnectedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *c = rewriter.getContext();
    SmallVector<Attribute, 1> trailing{
        opq(c, fmtTensorSpec(op.getType(), outAttr(op),
                              op.getOutputScaleAttr(),
                              op.getOutputZpAttr())),
    };
    return emitRuntimeCall(rewriter, op, "fully_connected",
                           ValueRange{adaptor.getInput(), adaptor.getWeight(),
                                      adaptor.getBias()},
                           trailing, tensorOpaqueTy(c));
  }
};

struct CastToEmitC : public OpConversionPattern<CastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *c = rewriter.getContext();
    SmallVector<Attribute, 1> trailing{
        opq(c, fmtTensorSpec(op.getType(), outAttr(op),
                              op.getOutputScaleAttr(),
                              op.getOutputZpAttr())),
    };
    return emitRuntimeCall(rewriter, op, "cast",
                           ValueRange{adaptor.getInput()}, trailing,
                           tensorOpaqueTy(c));
  }
};

struct PadToEmitC : public OpConversionPattern<PadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *c = rewriter.getContext();
    auto fmtFloat = [](float v) {
      std::string s;
      llvm::raw_string_ostream(s) << v << "f";
      return s;
    };
    SmallVector<Attribute, 2> trailing{
        opq(c, fmtFloat(op.getPadConst().convertToFloat())),
        opq(c, fmtTensorSpec(op.getType(), outAttr(op),
                              op.getOutputScaleAttr(),
                              op.getOutputZpAttr())),
    };
    return emitRuntimeCall(rewriter, op, "pad",
                           ValueRange{adaptor.getInput(), adaptor.getPadding()},
                           trailing, tensorOpaqueTy(c));
  }
};

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

// Pre-pass: prepend a `graph` argument to every func.func, and rename
// any function called `main` (would clash with C++'s `int main()` once the
// emitter inlines it into the runner translation unit).
//
// Has to run as plain IR rewriting before applyPartialConversion, because
// the per-op patterns reference the parent func's argument-0 to source the
// graph SSA value. Doing this inside dialect-conversion would create a
// chicken-and-egg with pattern ordering.
static void prependGraphParam(func::FuncOp func) {
  if (func.isExternal())
    return;
  MLIRContext *ctx = func.getContext();
  auto graphTy = emitc::OpaqueType::get(ctx, kGraphCxx);

  SmallVector<Type, 8> inputs{graphTy};
  for (Type t : func.getFunctionType().getInputs())
    inputs.push_back(t);
  func.setType(FunctionType::get(ctx, inputs,
                                 func.getFunctionType().getResults()));
  func.getBody().front().insertArgument(/*index=*/0u, graphTy, func.getLoc());

  if (func.getName() == "main")
    func.setName("timvx_main");
}

struct TIMVXToEmitCPass
    : public impl::TIMVXToEmitCPassBase<TIMVXToEmitCPass> {
  void runOnOperation() final {
    MLIRContext *ctx = &getContext();

    // Pre-pass — see prependGraphParam comment for why this is plain IR
    // rewriting rather than a conversion pattern.
    for (auto func : llvm::to_vector(getOperation().getOps<func::FuncOp>()))
      prependGraphParam(func);

    TIMVXToEmitCTypeConverter converter(ctx);

    ConversionTarget target(*ctx);
    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addIllegalDialect<TIMVXDialect>();

    // func.func / func.return are legal iff all signatures use converted
    // types — the populate* helpers handle the actual rewriting.
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp f) {
      return converter.isSignatureLegal(f.getFunctionType()) &&
             converter.isLegal(&f.getBody());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp r) { return converter.isLegal(r.getOperandTypes()); });
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp c) { return converter.isLegal(c); });

    RewritePatternSet patterns(ctx);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    populateReturnOpTypeConversionPattern(patterns, converter);
    populateCallOpTypeConversionPattern(patterns, converter);

    // Per-pass counter for generating unique static-array names backing
    // each timvx.const lowering — see ConstToEmitC.
    unsigned constCounter = 0;
    patterns.add<ConstToEmitC>(converter, ctx, &constCounter);

    patterns.add<ConstShapeToEmitC, ClipToEmitC, Conv2DToEmitC,
                 Pool2DToEmitC, TransposeToEmitC, CastToEmitC,
                 PadToEmitC, FullyConnectedToEmitC, ReduceSumToEmitC,
                 ReshapeToEmitC, SliceToEmitC>(converter, ctx);

    // 1-to-1 helper-call patterns whose only trailing arg is output_spec.
    auto addSimple = [&](StringRef name, auto opTag) {
      using T = decltype(opTag);
      patterns.add<SimpleRuntimeCall<T>>(converter, ctx, name);
    };
    addSimple("matmul", MatMulOp{});
    addSimple("multiply", MultiplyOp{});
    addSimple("add", AddOp{});
    addSimple("sub", SubOp{});
    addSimple("pow_op", PowOp{});
    addSimple("rcp", RcpOp{});

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createTIMVXToEmitCPass() {
  return std::make_unique<TIMVXToEmitCPass>();
}

} // namespace timvx
} // namespace mlir
