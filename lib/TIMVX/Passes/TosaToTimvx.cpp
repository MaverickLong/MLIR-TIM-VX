//===- TosaToTimvx.cpp - tosa-to-timvx pass ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Mapping table (TOSA op  ->  TIM-VX op  ->  tim::vx C++ class):
//
//   tosa.clamp -> timvx.clip -> tim::vx::ops::Clip
//   tosa.const -> timvx.const -> tensor w/ TensorAttribute::CONSTANT
//   tosa.const_shape -> timvx.const_shape -> (compile-time shape)
//   tosa.conv2d -> timvx.conv2d -> tim::vx::ops::Conv2d
//   tosa.matmul -> timvx.fully_connected -> tim::vx::ops::FullyConnected
//                  (when B is a tosa.const; weight is transposed at
//                   compile time, zero bias is synthesized)
//   tosa.matmul -> timvx.matmul -> tim::vx::ops::Matmul
//                  (fallback when B is a runtime activation)
//   tosa.max_pool2d -> timvx.pool2d -> tim::vx::ops::Pool2d (PoolType::MAX)
//   tosa.avg_pool2d -> timvx.pool2d -> tim::vx::ops::Pool2d (PoolType::AVG)
//   tosa.mul -> timvx.multiply -> tim::vx::ops::Multiply
//   tosa.pow -> timvx.pow -> tim::vx::ops::Pow
//   tosa.reciprocal -> timvx.rcp -> tim::vx::ops::Rcp
//   tosa.reshape -> timvx.reshape -> tim::vx::ops::Reshape
//   tosa.slice -> timvx.slice -> tim::vx::ops::Slice
//   tosa.sub -> timvx.sub -> tim::vx::ops::Sub
//   tosa.add -> timvx.add -> tim::vx::ops::Add
//   tosa.transpose -> timvx.transpose -> tim::vx::ops::Transpose
//
// Spatial ops (`Conv2DOpConversion`, `MaxPool2DConversion`,
// `AvgPool2DConversion`, plus the fused-conv inside `RescaleConvFusion`)
// wrap their TIM-VX-side op in explicit `timvx.transpose` ops:
//
//      tosa.conv2d (NHWC)
//   --becomes-->
//      %in_whcn  = timvx.transpose %in_nhwc  perms = [2, 1, 3, 0]
//      %w_whcn   = timvx.transpose %w_ohwi   perms = [2, 1, 3, 0]
//      %out_whcn = timvx.conv2d   %in_whcn, %w_whcn, %bias
//                                  pad = [l, r, t, b]   (was [t,b,l,r])
//                                  stride = [W, H]      (was [H, W])
//                                  dilation = [W, H]    (was [H, W])
//      %out_nhwc = timvx.transpose %out_whcn perms = [3, 1, 0, 2]
//
// Each transpose is a real `tim::vx::ops::Transpose` data-movement op,
// so the byte layout that `Conv2d`/`Pool2d` actually sees is genuine
// WHCN regardless of the surrounding graph.  The rest of the IR keeps
// NHWC semantics so TOSA verifiers are happy at every pipeline boundary.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

namespace mlir {
namespace timvx {

#define GEN_PASS_DEF_TOSATOTIMVXPASS
#include "TIMVX/TIMVXPasses.h.inc"

namespace {
using namespace ::mlir::timvx::detail;

//===----------------------------------------------------------------------===//
// Bespoke patterns for ops whose attributes need translation
//===----------------------------------------------------------------------===//

/// tosa.clamp -> timvx.clip (for relu-shaped ranges)
///                or
///                timvx.maximum + timvx.minimum (everything else)
///
/// `tim::vx::ops::Clip`'s kernel-selector pattern-matches certain ranges
/// to specialised activation kernels at compile time. Some of those fast
/// paths are correct on VIP9000Nano-DI:
///
///   * range (0, +inf) -> `relu`     (== max(x, 0))           — OK
///   * range (0, 1)    -> `relu1`    (== min(max(x, 0), 1))   — OK
///   * range (0, 6)    -> `relu6`    (== min(max(x, 0), 6))   — OK
///
/// But the same kernel selector folds finite-symmetric ranges to the
/// same fast paths and gives wrong results:
///
///   * range (-1, 1)   -> `relu1`     (incorrect; drops negative half)
///
/// And the Clip fast-paths are also rank-restricted (rank-4 fp32 only
/// per the op probe matrix), so even non-overlapping ranges can pick a
/// degenerate kernel for low-rank inputs.
///
/// We pick the lowering form per-clamp:
///
///   * If `min_val == 0` and `max_val == +inf`: emit `timvx.clip(0,
///     +inf)`. The kernel selector routes this to `relu`, which is
///     correct. Keeping it as a Clip preserves the idiomatic form
///     `QuantResidualFuse` looks for when collapsing residual chains.
///
///   * Otherwise (any finite range, asymmetric or symmetric), decompose
///     to `timvx.maximum(x, lo) -> timvx.minimum(_, hi)`. Both ops
///     route through plain elementwise kernels which are FP32-PASS at
///     every rank with no range-based fast-path matching.
///
/// Drops `nan_mode` (TIM-VX has no per-op NaN policy).
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

    auto resTy = cast<RankedTensorType>(op.getType());
    auto elemTy = resTy.getElementType();
    if (!isa<FloatType>(elemTy))
      return rewriter.notifyMatchFailure(
          op, "non-float clamp not yet supported");

    double lo = minVal.getValueAsDouble();
    double hi = maxVal.getValueAsDouble();
    bool isReluShape = (lo == 0.0) &&
                       (hi >= std::numeric_limits<float>::max() / 2.0);
    if (isReluShape) {
      rewriter.replaceOpWithNewOp<ClipOp>(op, op.getType(), adaptor.getInput(),
                                            minVal, maxVal);
      return success();
    }

    Location loc = op.getLoc();
    auto scalarTy = RankedTensorType::get({1}, elemTy);
    auto loSplat = DenseElementsAttr::get(scalarTy, minVal.getValue());
    auto hiSplat = DenseElementsAttr::get(scalarTy, maxVal.getValue());
    Value loConst = ConstOp::create(rewriter, loc, scalarTy, loSplat,
                                     /*quant_scale=*/FloatAttr{},
                                     /*quant_zp=*/IntegerAttr{});
    Value hiConst = ConstOp::create(rewriter, loc, scalarTy, hiSplat,
                                     /*quant_scale=*/FloatAttr{},
                                     /*quant_zp=*/IntegerAttr{});

    Value afterLo = MaximumOp::create(rewriter, loc, resTy,
                                       adaptor.getInput(), loConst);
    rewriter.replaceOpWithNewOp<MinimumOp>(op, resTy, afterLo, hiConst);
    return success();
  }
};

/// tosa.mul -> timvx.multiply. We reject the rewrite at non-zero shift
/// since TOSA computes `(input1 * input2) >> shift` and tim::vx::Multiply
/// has no shift.
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

/// Const passthrough.
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

/// const_shape passthrough — converts `!tosa.shape<N>` to
/// `tensor<Nxindex>` for simplicity.
struct ConstShapeOpConversion : public OpConversionPattern<tosa::ConstShapeOp> {
  using OpConversionPattern<tosa::ConstShapeOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::ConstShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto values = dyn_cast<DenseIntElementsAttr>(op.getValuesAttr());
    if (!values)
      return rewriter.notifyMatchFailure(op, "non-dense const_shape values");

    auto resultType =
        RankedTensorType::get({static_cast<int64_t>(values.getNumElements())},
                              rewriter.getIndexType());
    auto reTyped = DenseIntElementsAttr::get(
        resultType, llvm::to_vector(values.getValues<APInt>()));
    rewriter.replaceOpWithNewOp<ConstShapeOp>(op, resultType, reTyped);
    return success();
  }
};

// Forward declaration; definition below the matmul helpers. Wraps a
// `timvx.reshape` with reverse-perm transposes so the inner reshape sees
// MLIR-ordered bytes (the harness flips byte layout at the function boundary
// per the TIM-VX innermost-first convention; `Reshape` is the one op whose
// "preserve raw bytes" semantics differ between the two layouts).
static Value emitReshape(OpBuilder &b, Location loc, Value src,
                         ArrayRef<int64_t> newShape,
                         FloatAttr outScale,
                         IntegerAttr outZp);

/// tosa.reshape / tosa.slice carry shape operands typed as
/// `!tosa.shape<N>`, while their timvx counterparts take
/// `tensor<Nxindex>`. We rely on `ConstShapeOpConversion` having already
/// rewritten the producer; the adaptor then exposes the shape operand
/// with its new tensor<Nxindex> type.
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
    Value result = emitReshape(rewriter, op.getLoc(), adaptor.getInput1(),
                               op.getType().getShape(), s, z);
    rewriter.replaceOp(op, result);
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

//===----------------------------------------------------------------------===//
// Matmul -> FullyConnected (preferred) / Matmul (fallback)
//===----------------------------------------------------------------------===//

/// Transpose a `[1, K, N]`-shaped dense weight constant down to `[N, K]`,
/// dropping the leading batch dim and swapping the inner two. Used by the
/// matmul->fully_connected rewrite. Only handles inline DenseElementsAttr
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

/// Build a layout-correct reshape: `transpose(reverse) -> reshape(reverse) ->
/// transpose(reverse)`. See ReshapeOpConversion for the byte-order rationale;
/// any `timvx.reshape` we emit must go through this wrapping or it will
/// scramble bytes whenever input or output has multiple non-trivial dims.
/// Reverse-perm transposes degenerate to no-ops for rank <= 1.
static Value emitReshape(OpBuilder &b, Location loc, Value src,
                         ArrayRef<int64_t> newShape,
                         FloatAttr outScale = FloatAttr{},
                         IntegerAttr outZp = IntegerAttr{}) {
  auto srcTy = cast<RankedTensorType>(src.getType());
  Type elemTy = srcTy.getElementType();

  auto reversePerm = [&](int64_t r) {
    SmallVector<int32_t> p(r);
    for (int64_t i = 0; i < r; ++i)
      p[i] = static_cast<int32_t>(r - 1 - i);
    return b.getDenseI32ArrayAttr(p);
  };

  Value preInner = src;
  if (srcTy.getRank() >= 2) {
    SmallVector<int64_t> preShape(srcTy.getShape().rbegin(),
                                  srcTy.getShape().rend());
    auto preTy = RankedTensorType::get(preShape, elemTy);
    preInner = TransposeOp::create(b, loc, preTy, src,
                                   reversePerm(srcTy.getRank()),
                                   outScale, outZp);
  }

  SmallVector<int64_t> revOutShape(newShape.rbegin(), newShape.rend());
  auto revOutTy = RankedTensorType::get(revOutShape, elemTy);

  auto shapeTy = RankedTensorType::get(
      {static_cast<int64_t>(revOutShape.size())}, b.getIndexType());
  SmallVector<APInt> shapeInts;
  shapeInts.reserve(revOutShape.size());
  for (int64_t v : revOutShape)
    shapeInts.emplace_back(/*numBits=*/64, /*val=*/static_cast<uint64_t>(v),
                           /*isSigned=*/true);
  auto shapeAttr = DenseIntElementsAttr::get(shapeTy, shapeInts);
  Value revShapeConst = ConstShapeOp::create(b, loc, shapeTy, shapeAttr);

  Value reshaped = ReshapeOp::create(b, loc, revOutTy, preInner, revShapeConst,
                                     outScale, outZp);

  if (static_cast<int64_t>(newShape.size()) >= 2) {
    auto outTy = RankedTensorType::get(newShape, elemTy);
    return TransposeOp::create(b, loc, outTy, reshaped,
                               reversePerm(newShape.size()),
                               outScale, outZp);
  }
  return reshaped;
}

/// tosa.matmul -> timvx.fully_connected (preferred when B is a constant
/// weight) or timvx.matmul (fallback for activation×activation).
struct MatMulOpConversion : public OpConversionPattern<tosa::MatMulOp> {
  using OpConversionPattern<tosa::MatMulOp>::OpConversionPattern;

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

    auto weightTy = RankedTensorType::get({K, N}, elemTy);
    Value weight = ConstOp::create(rewriter, loc, weightTy, transposed,
                                    /*quant_scale=*/FloatAttr{},
                                    /*quant_zp=*/IntegerAttr{});

    auto biasTy = RankedTensorType::get({N}, elemTy);
    auto biasAttr = DenseElementsAttr::get(
        biasTy, cast<TypedAttr>(rewriter.getZeroAttr(elemTy)));
    Value bias = ConstOp::create(rewriter, loc, biasTy, biasAttr,
                                  /*quant_scale=*/FloatAttr{},
                                  /*quant_zp=*/IntegerAttr{});

    Value reshapedA = emitReshape(rewriter, loc, op.getA(), {Ba * M, K});
    auto fcOutTy = RankedTensorType::get({Ba * M, N}, outTy.getElementType());
    Value fc = FullyConnectedOp::create(rewriter, loc, fcOutTy, reshapedA,
                                         weight, bias,
                                         /*output_scale=*/FloatAttr{},
                                         /*output_zp=*/IntegerAttr{});

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

//===----------------------------------------------------------------------===//
// Conv2D / Pool2D (with NHWC<->WHCN wrapping)
//===----------------------------------------------------------------------===//

/// tosa.conv2d -> [transpose -> timvx.conv2d -> transpose]
///
/// TODO: Dropped attributes / operands:
///   - acc_type: tosa carries an explicit accumulator type;
///     tim::vx::ops::Conv2d picks its accumulator internally (i32 for
///     INT8, f32 for FP). We have no knob to forward this onto, so we
///     drop it.
///   - local_bound: rare flag (default false); we don't model it.
///     Match-fail when set so it isn't dropped silently.
struct Conv2DOpConversion : public OpConversionPattern<tosa::Conv2DOp> {
  Conv2DOpConversion(MLIRContext *ctx, const QuantInfoMap &qm)
      : OpConversionPattern(ctx), quantInfo(qm) {}
  const QuantInfoMap &quantInfo;
  LogicalResult
  matchAndRewrite(tosa::Conv2DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isConstantZero(op.getInputZp()))
      return rewriter.notifyMatchFailure(op, "non-zero input zero-point");
    if (!isConstantZero(op.getWeightZp()))
      return rewriter.notifyMatchFailure(op, "non-zero weight zero-point");
    if (op.getLocalBound())
      return rewriter.notifyMatchFailure(op, "local_bound=true not supported");

    Location loc = op.getLoc();
    // Forward the input's quant onto the wrapping transpose so the runtime
    // sees a consistent ASYM chain at the IO boundary; without this, an
    // i8 input with seeded Quant(1.0, 0) goes through a no-quant PERMUTE
    // and the runtime auto-injects a DataConvert (i8->u8) that mismatches
    // PERMUTE's declared INT8 output.
    auto [inS, inZ] = qmapAttrs(quantInfo, op.getInput(), rewriter);
    Value inputWhcn = wrapWithTranspose(rewriter, loc, adaptor.getInput(),
                                         kPermNHWCToWHCN, inS, inZ);
    Value weightWhicoc =
        wrapWithTranspose(rewriter, loc, adaptor.getWeight(), kPermNHWCToWHCN);

    auto outNhwcTy = cast<RankedTensorType>(op.getType());
    auto outWhcnShape =
        applyPerm<int64_t>(outNhwcTy.getShape(), kPermNHWCToWHCN);
    auto outWhcnTy = outNhwcTy.clone(outWhcnShape);

    // Output transpose: forward output quant if known. For a conv with no
    // rescale partner the output is typically i32 (accumulator) and there's
    // no quant to forward — qmapAttrs returns null attrs.
    auto [outS, outZ] = qmapAttrs(quantInfo, op.getResult(), rewriter);
    Value convOut = Conv2DOp::create(
        rewriter, loc, outWhcnTy, inputWhcn, weightWhicoc, adaptor.getBias(),
        tosaPadToTIMVX(rewriter, op.getPad()),
        tosaHWToTIMVXWH(rewriter, op.getStride()),
        tosaHWToTIMVXWH(rewriter, op.getDilation()),
        outS, outZ);

    rewriter.replaceOpWithNewOp<TransposeOp>(
        op, outNhwcTy, convOut,
        rewriter.getDenseI32ArrayAttr(kPermWHCNToNHWC),
        outS, outZ);
    return success();
  }
};

/// pool2d rewrites. Both pool kinds wrap with NHWC<->WHCN transposes;
/// `quant_scale`/`quant_zp` is carried on the input/output transposes
/// because pool preserves quant (in == out).
template <typename TosaPool>
static LogicalResult lowerPool2D(
    TosaPool op, typename OpConversionPattern<TosaPool>::OpAdaptor adaptor,
    PoolType kind, const QuantInfoMap &quantInfo,
    ConversionPatternRewriter &rewriter) {
  Location loc = op.getLoc();
  auto [s, z] = qmapAttrs(quantInfo, op.getResult(), rewriter);

  Value inputWhcn = wrapWithTranspose(rewriter, loc, adaptor.getInput(),
                                        kPermNHWCToWHCN, s, z);

  auto outNhwcTy = cast<RankedTensorType>(op.getType());
  auto outWhcnShape =
      applyPerm<int64_t>(outNhwcTy.getShape(), kPermNHWCToWHCN);
  auto outWhcnTy = outNhwcTy.clone(outWhcnShape);

  Value pooledWhcn = Pool2DOp::create(
      rewriter, loc, outWhcnTy, inputWhcn, kind,
      tosaHWToTIMVXWH(rewriter, op.getKernel()),
      tosaHWToTIMVXWH(rewriter, op.getStride()),
      tosaPadToTIMVX(rewriter, op.getPad()), s, z);

  rewriter.replaceOpWithNewOp<TransposeOp>(
      op, outNhwcTy, pooledWhcn,
      rewriter.getDenseI32ArrayAttr(kPermWHCNToNHWC), s, z);
  return success();
}

struct MaxPool2DConversion : public OpConversionPattern<tosa::MaxPool2dOp> {
  MaxPool2DConversion(MLIRContext *ctx, const QuantInfoMap &qm)
      : OpConversionPattern(ctx), quantInfo(qm) {}
  const QuantInfoMap &quantInfo;
  LogicalResult
  matchAndRewrite(tosa::MaxPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    return lowerPool2D(op, adaptor, PoolType::MAX, quantInfo, rewriter);
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
    // linear in real-value space, so the zp passes through). The upstream
    // peephole that emits this op already chooses the same zp on both
    // sides; for hand-written TOSA the user's responsibility.
    return lowerPool2D(op, adaptor, PoolType::AVG, quantInfo, rewriter);
  }
};

//===----------------------------------------------------------------------===//
// Rescale fusion / standalone, Cast, Pad, ReduceSum, Transpose
//===----------------------------------------------------------------------===//

// Matches `tosa.rescale` whose input is a `tosa.conv2d`, and replaces both
// with a single quantized `timvx.conv2d` carrying output_scale / output_zp.
struct RescaleConvFusion : public OpConversionPattern<tosa::RescaleOp> {
  RescaleConvFusion(MLIRContext *ctx, const QuantInfoMap &qm,
                     bool *failed)
      : OpConversionPattern(ctx, /*benefit=*/10), quantInfo(qm),
        strictFailed(failed) {}
  const QuantInfoMap &quantInfo;
  bool *strictFailed;

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

    auto siOpt = quantInfo.lookup(conv.getInput());
    if (!siOpt) {
      conv->emitError()
          << "RescaleConvFusion: conv input has no tracked quant info "
             "(buildQuantInfoMap should have seeded every quantized "
             "value — this op's input slipped through)";
      *strictFailed = true;
      return failure();
    }
    QuantInfo si = *siOpt;
    auto soOpt = quantInfo.lookup(op.getResult());
    if (!soOpt)
      return rewriter.notifyMatchFailure(
          op, "rescale result missing from QuantInfoMap "
              "(buildQuantInfoMap should have populated it)");
    QuantInfo so = *soOpt;

    // Solve M = (Si * Sw) / So for Sw, given the (Si, So) buildQuantInfoMap
    // pinned and the rescale's M = mul/2^shift. The Sw we derive keeps
    // the conv's TIM-VX-internal scale_factor invariant (storage matches
    // TFLite). If the rescale doesn't carry constant mul/shift or si.scale
    // is degenerate, the math is broken — strict-fail rather than silently
    // pick Sw = 1/128.
    auto mulVal = matchConstScalarInt(op.getMultiplier());
    auto shiftVal = matchConstScalarInt(op.getShift());
    if (!mulVal || !shiftVal) {
      op.emitError()
          << "RescaleConvFusion: rescale multiplier/shift is not a "
             "scalar constant — cannot solve for Sw";
      *strictFailed = true;
      return failure();
    }
    if (si.scale == 0.0) {
      op.emitError()
          << "RescaleConvFusion: si.scale is zero — cannot solve M = "
             "(Si*Sw)/So for Sw";
      *strictFailed = true;
      return failure();
    }
    double M = static_cast<double>(*mulVal) *
               std::pow(2.0, -double(*shiftVal));
    double sw = (M * so.scale) / si.scale;
    if (!(sw > 0.0) || !std::isfinite(sw)) {
      op.emitError()
          << "RescaleConvFusion: derived Sw = " << sw << " is not a "
             "positive finite value — the (Si, So, M) triple is "
             "internally inconsistent";
      *strictFailed = true;
      return failure();
    }
    double sb = si.scale * sw;

    // Re-emit the weight + bias consts with symmetric-int8 quant attrs.
    // We create a fresh `timvx.const` here rather than retagging the
    // upstream `tosa.const`, because canonicalize before this pass may
    // have CSE'd the bias `dense<0>` with the rescale's input_zp
    // `dense<0>` — the upstream const can have multiple users we don't
    // want to retag.
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

    Location loc = op.getLoc();
    auto siScaleAttr = rewriter.getF64FloatAttr(si.scale);
    auto siZpAttr = rewriter.getI64IntegerAttr(si.zp);
    auto soScaleAttr = rewriter.getF64FloatAttr(so.scale);
    auto soZpAttr = rewriter.getI64IntegerAttr(so.zp);
    auto swScaleAttr = rewriter.getF64FloatAttr(sw);
    auto wzpAttr = rewriter.getI64IntegerAttr(0);

    Value inputWhcn = wrapWithTranspose(rewriter, loc, conv.getInput(),
                                         kPermNHWCToWHCN,
                                         siScaleAttr, siZpAttr);
    Value weightWhicoc = wrapWithTranspose(rewriter, loc, weight,
                                            kPermNHWCToWHCN,
                                            swScaleAttr, wzpAttr);

    auto outNhwcTy = cast<RankedTensorType>(op.getType());
    auto outWhcnShape =
        applyPerm<int64_t>(outNhwcTy.getShape(), kPermNHWCToWHCN);
    auto outWhcnTy = outNhwcTy.clone(outWhcnShape);

    Value convOut = Conv2DOp::create(
        rewriter, loc, outWhcnTy, inputWhcn, weightWhicoc, bias,
        tosaPadToTIMVX(rewriter, conv.getPad()),
        tosaHWToTIMVXWH(rewriter, conv.getStride()),
        tosaHWToTIMVXWH(rewriter, conv.getDilation()),
        soScaleAttr, soZpAttr);

    rewriter.replaceOpWithNewOp<TransposeOp>(
        op, outNhwcTy, convOut,
        rewriter.getDenseI32ArrayAttr(kPermWHCNToNHWC),
        soScaleAttr, soZpAttr);

    rewriter.eraseOp(conv);
    return success();
  }
};

// Standalone `tosa.rescale`: emit `timvx.dataconvert` carrying
// (output_scale, output_zp) from QuantInfoMap.
//
// Pattern benefit (1) is intentionally lower than `RescaleConvFusion`'s
// (10) — when a rescale's input is a `tosa.conv2d`, the framework tries
// fusion FIRST and only falls through to standalone when fusion bails
// (per-channel weight, non-constant mul/shift, missing quant info, etc).
// We deliberately do NOT skip conv2d-fed rescales here: the conv→rescale
// chain is otherwise unhandled when fusion declines, and TIM-VX's
// DataConvert handles `i32 → i8/u8` (NN-engine accumulator → per-tensor
// requant) just as well as the `i8 → i8/u8` case the standalone path was
// originally written for. The only price for not fusing is that the
// conv2d emits its i32 accumulator separately and DataConvert then
// requantizes it — semantically identical, just an extra op_create.
struct RescaleStandaloneConversion
    : public OpConversionPattern<tosa::RescaleOp> {
  RescaleStandaloneConversion(MLIRContext *ctx, const QuantInfoMap &qm)
      : OpConversionPattern(ctx), quantInfo(qm) {}
  const QuantInfoMap &quantInfo;

  LogicalResult
  matchAndRewrite(tosa::RescaleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (op.getPerChannel())
      return rewriter.notifyMatchFailure(op,
                                         "per-channel standalone rescale");
    auto inTy = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto outTy = dyn_cast<RankedTensorType>(op.getType());
    if (!inTy || !outTy)
      return rewriter.notifyMatchFailure(op, "non-ranked tensor");
    // Storage type may be wrapped in a `quant.uniform<i8:f32, …>` after
    // `tosa-quant-anchor` ran — unwrap before the integer-only check.
    auto storageOf = [](Type t) -> Type {
      if (auto qt = dyn_cast<quant::QuantizedType>(t))
        return qt.getStorageType();
      return t;
    };
    if (!isa<IntegerType>(storageOf(inTy.getElementType())) ||
        !isa<IntegerType>(storageOf(outTy.getElementType())))
      return rewriter.notifyMatchFailure(
          op, "DataConvert only supports int->int requantize on this chip");

    auto so = quantInfo.lookup(op.getResult());
    if (!so)
      return rewriter.notifyMatchFailure(
          op, "rescale result missing from QuantInfoMap");

    rewriter.replaceOpWithNewOp<DataConvertOp>(
        op, op.getType(), adaptor.getInput(),
        rewriter.getF64FloatAttr(so->scale),
        rewriter.getI64IntegerAttr(so->zp));
    return success();
  }
};

// `tosa.cast` -> `timvx.cast`. TIM-VX's Cast is the value-cast op (it
// dispatches to the GPU `cast` kernel, ignores scale/zp on either end for
// the cast itself). DataConvert is NOT used here — on this chip the
// `vivante.nn.tensorcopy` path COMPILE_FAILs for every f32->int direction.
//
// The i8↔u8 promotion bridge that used to live in this pattern has moved
// to `--timvx-promote-i8-to-u8`, which runs after `--tosa-to-timvx` and
// inserts/folds the ±128 compensation around every f32↔u8 cast in one
// pass. This conversion just lowers the cast as-is.
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
// has rewritten to a `tensor<2Nxindex>` constant.
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

    FloatAttr scale;
    IntegerAttr zp;
    if (auto qi = quantInfo.lookup(op.getResult())) {
      scale = rewriter.getF64FloatAttr(qi->scale);
      zp = rewriter.getI64IntegerAttr(qi->zp);
    }

    rewriter.replaceOpWithNewOp<PadOp>(
        op, op.getType(), adaptor.getInput1(), padding,
        rewriter.getF32FloatAttr(static_cast<float>(padConstF)), scale, zp);
    return success();
  }
};

// Decompose `tosa.reduce_sum` over an i8/f32 tensor into a slice+add chain.
// The NPU's REDUCE op decomposes to REDUCE_MEAN_INTERNAL at compile time,
// which lacks an FP32 kernel on this chip. Slice + Add are FP32-PASS, so
// we emit `dim_K - 1` adds of `dim_K` unit-thickness slices along the
// reduce axis.
struct ReduceSumConversion : public OpConversionPattern<tosa::ReduceSumOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::ReduceSumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ReduceSumOp>(
        op, op.getType(), adaptor.getInput(), adaptor.getAxis(), true);
    return success();
  }
  // matchAndRewrite(tosa::ReduceSumOp op, OpAdaptor adaptor,
  //                 ConversionPatternRewriter &rewriter) const final {
  //   auto inTy = dyn_cast<RankedTensorType>(op.getInput().getType());
  //   auto outTy = dyn_cast<RankedTensorType>(op.getType());
  //   if (!inTy || !inTy.hasStaticShape() || !outTy)
  //     return rewriter.notifyMatchFailure(op, "non-static input shape");

  //   int64_t axis = static_cast<int64_t>(op.getAxis());
  //   int64_t rank = inTy.getRank();
  //   if (axis < 0 || axis >= rank)
  //     return rewriter.notifyMatchFailure(op, "axis out of range");

  //   int64_t reduceDim = inTy.getShape()[axis];
  //   if (reduceDim <= 0)
  //     return rewriter.notifyMatchFailure(op, "non-positive reduce dim");
  //   if (reduceDim == 1) {
  //     rewriter.replaceOp(op, adaptor.getInput());
  //     return success();
  //   }

  //   Location loc = op.getLoc();
  //   Type elemTy = inTy.getElementType();
  //   auto sliceShape = llvm::to_vector(inTy.getShape());
  //   sliceShape[axis] = 1;
  //   auto sliceTy = RankedTensorType::get(sliceShape, elemTy);

  //   auto shapeIdxTy = RankedTensorType::get({rank}, rewriter.getIndexType());
  //   SmallVector<int64_t> startBase(rank, 0);

  //   auto makeShape = [&](ArrayRef<int64_t> v) {
  //     auto attr = DenseIntElementsAttr::get(shapeIdxTy, v);
  //     auto shapeTy = tosa::shapeType::get(rewriter.getContext(), rank);
  //     return tosa::ConstShapeOp::create(rewriter, loc, shapeTy, attr)
  //         .getResult();
  //   };

  //   Value sizeShape = makeShape(sliceShape);

  //   SmallVector<Value> slices;
  //   slices.reserve(reduceDim);
  //   for (int64_t i = 0; i < reduceDim; ++i) {
  //     auto start = startBase;
  //     start[axis] = i;
  //     Value startShape = makeShape(start);
  //     Value sl = tosa::SliceOp::create(rewriter, loc, sliceTy,
  //                                       adaptor.getInput(), startShape,
  //                                       sizeShape);
  //     slices.push_back(sl);
  //   }

  //   Value acc = slices[0];
  //   for (size_t i = 1; i < slices.size(); ++i) {
  //     acc = tosa::AddOp::create(rewriter, loc, sliceTy, acc, slices[i]);
  //   }
  //   rewriter.replaceOp(op, acc);
  //   return success();
  // }
};

/// transpose rewrite. nan_mode dropped.
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

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

struct TosaToTIMVXPass : public impl::TosaToTIMVXPassBase<TosaToTIMVXPass> {
  void runOnOperation() final {
    ConversionTarget target(getContext());
    target.addLegalDialect<TIMVXDialect>();
    target.addIllegalDialect<tosa::TosaDialect>();

    // A `tosa.conv2d` that's the producer of a `tosa.rescale` will be
    // subsumed by RescaleConvFusion (which erases the conv after fusing).
    // Marking it dynamically legal here lets the driver defer it to the
    // rescale match instead of failing on it via the FP Conv2DOpConversion.
    target.addDynamicallyLegalOp<tosa::Conv2DOp>([](tosa::Conv2DOp op) {
      return op->hasOneUse() &&
             isa<tosa::RescaleOp>(*op->getUsers().begin());
    });

    // Pre-walk to derive (scale, zp) for every quantized SSA value. The
    // rescale-conv fusion pattern queries this so it doesn't have to
    // re-derive scales for each instance.
    //
    // `strictFailed` propagates from buildQuantInfoMap and from the
    // RescaleConvFusion pattern; both use it to flag IR-level
    // inconsistencies (missing seed quant, no way to pin So) that
    // strict mode treats as halt-the-pipeline rather than silently
    // assuming the canonical Sw=1/128 / Si=1.0.
    bool strictFailed = false;
    QuantInfoMap qmap;
    getOperation().walk([&](func::FuncOp f) {
      buildQuantInfoMap(f, qmap, &strictFailed);
    });
    if (strictFailed) {
      signalPassFailure();
      return;
    }

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
        ReduceSumConversion>(&getContext());
    patterns.add<Conv2DOpConversion,
                 RescaleStandaloneConversion,
                 CastConversion, PadConversion,
                 MaxPool2DConversion, AvgPool2DConversion,
                 ReshapeOpConversion, SliceOpConversion,
                 TransposeConversion>(&getContext(), qmap);
    // RescaleConvFusion needs the strict flag too — its M/Si/So sanity
    // checks (after buildQuantInfoMap pinned the QM entries) are also
    // halt-pipeline failures rather than silent fall-through.
    patterns.add<RescaleConvFusion>(&getContext(), qmap, &strictFailed);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
    if (strictFailed) signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createTosaToTIMVXPass() {
  return std::make_unique<TosaToTIMVXPass>();
}

} // namespace timvx
} // namespace mlir
