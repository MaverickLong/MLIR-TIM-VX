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
//   tosa.matmul → timvx.matmul → tim::vx::ops::Matmul
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

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace timvx {

#define GEN_PASS_DEF_TOSATOTIMVXPASS
#include "TIMVX/TIMVXPasses.h.inc"

namespace {

// Helper for finding zero-valued tensors
static bool isConstantZero(Value v) {
  return matchPattern(v, m_Zero()) || matchPattern(v, m_AnyZeroFloat());
}

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

    rewriter.replaceOpWithNewOp<ConstOp>(op, op.getType(), values);
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
struct ReshapeOpConversion : public OpConversionPattern<tosa::ReshapeOp> {
  using OpConversionPattern<tosa::ReshapeOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value shape = adaptor.getShape();
    if (!isa<RankedTensorType>(shape.getType()))
      return rewriter.notifyMatchFailure(
          op, "shape operand not converted to tensor<Nxindex>");
    rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(),
                                           adaptor.getInput1(), shape);
    return success();
  }
};

struct SliceOpConversion : public OpConversionPattern<tosa::SliceOp> {
  using OpConversionPattern<tosa::SliceOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value start = adaptor.getStart();
    Value size = adaptor.getSize();
    if (!isa<RankedTensorType>(start.getType()) ||
        !isa<RankedTensorType>(size.getType()))
      return rewriter.notifyMatchFailure(
          op, "start/size operands not converted to tensor<Nxindex>");
    rewriter.replaceOpWithNewOp<SliceOp>(op, op.getType(), adaptor.getInput1(),
                                         start, size);
    return success();
  }
};

/// tosa.matmul -> timvx.matmul
//
/// TOSA's matmul has scalar-tensor zero-point operands (a_zp / b_zp); TIM-VX's
/// matmul carries quant params on the tensor type instead, so we drop them.
/// As with conv2d, only the constant-zero zp case is handled here — non-zero
/// or non-constant zp would change semantics.
struct MatMulOpConversion : public OpConversionPattern<tosa::MatMulOp> {
  using OpConversionPattern<tosa::MatMulOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::MatMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isConstantZero(op.getAZp()))
      return rewriter.notifyMatchFailure(op, "non-zero a zero-point");
    if (!isConstantZero(op.getBZp()))
      return rewriter.notifyMatchFailure(op, "non-zero b zero-point");

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
        op.getDilationAttr());
    return success();
  }
};

/// pool2d rewrites. nan_mode / acc_type dropped
struct MaxPool2DConversion : public OpConversionPattern<tosa::MaxPool2dOp> {
  using OpConversionPattern<tosa::MaxPool2dOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::MaxPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<Pool2DOp>(
        op, op.getType(), adaptor.getInput(), PoolType::MAX,
        adaptor.getKernelAttr(), adaptor.getStrideAttr(), op.getPadAttr());
    return success();
  }
};
struct AvgPool2DConversion : public OpConversionPattern<tosa::AvgPool2dOp> {
  using OpConversionPattern<tosa::AvgPool2dOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::AvgPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isConstantZero(op.getInputZp()))
      return rewriter.notifyMatchFailure(op, "non-zero input zero-point");
    if (!isConstantZero(op.getOutputZp()))
      return rewriter.notifyMatchFailure(op, "non-zero output zero-point");
    rewriter.replaceOpWithNewOp<Pool2DOp>(
        op, op.getType(), adaptor.getInput(), PoolType::AVG,
        adaptor.getKernelAttr(), adaptor.getStrideAttr(), op.getPadAttr());
    return success();
  }
};

/// transpose rewrite. nan_mode dropped
struct TransposeConversion : public OpConversionPattern<tosa::TransposeOp> {
  using OpConversionPattern<tosa::TransposeOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<TransposeOp>(
        op, op.getType(), adaptor.getInput1(), adaptor.getPermsAttr());
    return success();
  }
};

}; // namespace

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

struct TosaToTIMVXPass : public impl::TosaToTIMVXPassBase<TosaToTIMVXPass> {
  void runOnOperation() final {
    ConversionTarget target(getContext());
    target.addLegalDialect<TIMVXDialect>();

    target.addIllegalDialect<tosa::TosaDialect>();

    // target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(&getContext());
    patterns.add<
        // Tensor-only passthroughs (no attrs to translate).
        TensorOnlyOpConversion<tosa::PowOp, PowOp>,
        TensorOnlyOpConversion<tosa::ReciprocalOp, RcpOp>,
        TensorOnlyOpConversion<tosa::AddOp, AddOp>,
        TensorOnlyOpConversion<tosa::SubOp, SubOp>,
        // Bespoke (attribute / operand translation).
        ClampOpConversion, MulOpConversion, MatMulOpConversion,
        MaxPool2DConversion, ConstOpConversion, ConstShapeOpConversion,
        ReshapeOpConversion, SliceOpConversion, TransposeConversion,
        Conv2DOpConversion, AvgPool2DConversion>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createTosaToTIMVXPass() {
  return std::make_unique<TosaToTIMVXPass>();
}

} // namespace timvx
} // namespace mlir
