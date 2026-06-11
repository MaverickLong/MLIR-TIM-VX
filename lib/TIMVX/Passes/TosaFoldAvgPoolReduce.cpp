//===- TosaFoldAvgPoolReduce.cpp - tosa-fold-avgpool-reduce -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Replace `cast(scale * sum(x)/N + zp)` (the global avg-pool emitted by
// tflite as a `reduce_sum × 2 -> mul(1/N) -> requant tail`) with
// `avg_pool2d(cast(scale * x + zp))` so the average runs as a single
// Pool2D AVG kernel on u8 instead of 14 unique slice/add shaders. See
// the pass description in TIMVXPasses.td for the full rationale.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace timvx {

#define GEN_PASS_DEF_TOSAFOLDAVGPOOLREDUCEPASS
#include "TIMVX/TIMVXPasses.h.inc"

namespace {
using namespace ::mlir::timvx::detail;

// Read a scalar fp32 value from a `tosa.const`. Used to recognize the
// 1/N, scale, and zp constants in the requant chain.
std::optional<double> getConstFp32Scalar(Value v) {
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
std::optional<std::pair<Value, double>> splitBinaryConstOperand(BinOp op) {
  Value a = op.getInput1(), b = op.getInput2();
  if (auto s = getConstFp32Scalar(b)) return {{a, *s}};
  if (auto s = getConstFp32Scalar(a)) return {{b, *s}};
  return std::nullopt;
}

// Match a tosa.cast f32 -> narrow_int that carries the
// `timvx.output_scale` / `timvx.output_zp` discardable attrs deposited
// by `RequantI32SkipFold` — that's the requant-tail's terminal cast.
// Walk back through:
//   (cast_terminal) <- add(zp_const) <- mul(scale_const) <- mul(1/N const)
//   <- reshape <- reduce_sum(W) <- reduce_sum(H) <- <fp32 spatial input>
//
// and rewrite to:
//   <fp32 spatial> -> mul(scale) -> add(zp) -> cast f32->int -> avg_pool2d
//   -> reshape (back to the cast_terminal's original output shape).
struct AvgPoolReduceFold : public OpRewritePattern<tosa::CastOp> {
  using OpRewritePattern<tosa::CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::CastOp castOp,
                                 PatternRewriter &rewriter) const final {
    // (1) cast must be the post-RequantI32SkipFold terminal cast: f32 ->
    //     narrow int with our discardable quant attrs.
    auto srcTy = dyn_cast<RankedTensorType>(castOp.getInput().getType());
    auto dstTy = dyn_cast<RankedTensorType>(castOp.getType());
    if (!srcTy || !dstTy) return failure();
    if (!srcTy.getElementType().isF32()) return failure();
    // Destination element may be `quant.uniform<iN:fXX, S:Z>` from the
    // tflite-tagged TOSA upstream; unwrap to the underlying narrow int
    // for the storage-width check.
    Type dstElem = dstTy.getElementType();
    if (auto qty = dyn_cast<quant::QuantizedType>(dstElem))
      dstElem = qty.getStorageType();
    auto outI = dyn_cast<IntegerType>(dstElem);
    if (!outI || outI.getWidth() >= 32) return failure();
    auto castScale = castOp->getAttrOfType<FloatAttr>("timvx.output_scale");
    auto castZp = castOp->getAttrOfType<IntegerAttr>("timvx.output_zp");
    if (!castScale || !castZp) return failure();
    // When the upstream-tagged `quant.uniform` stamp is available on
    // dstTy, prefer ITS scale over the one `RequantI32SkipFold` recovered
    // from the inv-scale const. The two agree at fp32 precision but
    // differ at the 9th significant digit; downstream `tosa-quant-anchor`
    // rewraps the rebuilt avg_pool2d output using `castScale`, so a drift
    // between the cast's `output_scale` attr and the original stamp's
    // scale would cause the wrapping reshape to fail the TOSA same-elem
    // verifier (input wrapped with the recovered scale, output kept at
    // the tagged scale).
    if (auto qty = dyn_cast<quant::UniformQuantizedType>(
            dstTy.getElementType())) {
      castScale = rewriter.getF64FloatAttr(qty.getScale());
    }

    auto onlyUseIs = [](Value v, Operation *user) {
      return v.hasOneUse() && *v.getUsers().begin() == user;
    };

    // (2) Optional `tosa.add %y, zp_const`. Canonicalize folds `add %f,
    //     0.0` away when the recovered zp is 0, so the chain may shrink
    //     to `cast <- mul <- mul <- reshape <- reduce_sum × 2` directly.
    Value chainAfterZp;
    Operation *consumerOfMulScale = nullptr;
    double zpVal = 0.0;
    if (auto add = castOp.getInput().getDefiningOp<tosa::AddOp>()) {
      if (!onlyUseIs(add.getResult(), castOp)) return failure();
      auto addSplit = splitBinaryConstOperand<tosa::AddOp>(add);
      if (!addSplit) return failure();
      auto [addVar, addZpVal] = *addSplit;
      chainAfterZp = addVar;
      consumerOfMulScale = add;
      zpVal = addZpVal;
    } else {
      // Folded-zp shape: the cast input must be the scale multiply
      // directly. The zp is implicit zero; cross-check against the
      // cast's recorded zp attr.
      if (castZp.getInt() != 0) return failure();
      chainAfterZp = castOp.getInput();
      consumerOfMulScale = castOp;
      zpVal = 0.0;
    }

    // (3) chainAfterZp <- tosa.mul %z, scale_const (splat fp32)
    auto mulScale = chainAfterZp.getDefiningOp<tosa::MulOp>();
    if (!mulScale) return failure();
    if (!onlyUseIs(mulScale.getResult(), consumerOfMulScale)) return failure();
    auto mulScaleSplit = splitBinaryConstOperand<tosa::MulOp>(mulScale);
    if (!mulScaleSplit) return failure();
    auto [mulScaleVar, scaleVal] = *mulScaleSplit;

    // (4) mulScaleVar <- tosa.mul %w, inv_n_const (splat fp32 = 1/N)
    auto mulInvN = mulScaleVar.getDefiningOp<tosa::MulOp>();
    if (!mulInvN) return failure();
    if (!onlyUseIs(mulInvN.getResult(), mulScale)) return failure();
    auto mulInvNSplit = splitBinaryConstOperand<tosa::MulOp>(mulInvN);
    if (!mulInvNSplit) return failure();
    auto [mulInvNVar, invNVal] = *mulInvNSplit;
    if (invNVal <= 0.0) return failure();
    int64_t expectedN = static_cast<int64_t>(std::llround(1.0 / invNVal));
    if (expectedN <= 1) return failure();
    if (std::abs(invNVal - 1.0 / static_cast<double>(expectedN)) > 1e-6)
      return failure();

    // (5) Optional reshape (drops singleton dims).
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
      if (axis == 1) kernelH *= inT.getDimSize(1);
      else if (axis == 2) kernelW *= inT.getDimSize(2);
      else return failure();
      reduces.push_back(rs);
      cur = rs.getInput();
      lastConsumer = rs;
    }
    if (reduces.empty()) return failure();
    if (kernelH * kernelW != expectedN) return failure();

    // Bail on kernels larger than what the chip's NN-engine Pool2d can
    // actually execute correctly. VIP9000 silently produces wrong
    // values (no Compile error, just garbage output) when the kernel
    // exceeds typical CNN sizes (~32×32). Empirically a 224×224 fold for
    // simple_v1 produced an avg_pool with mean ~0.40 vs CPU's ~-0.05.
    // Refusing the fold here leaves the reduce_sum chain intact so the
    // CustomReduceSum OpenCL kernel under example/custom_ops/ picks it
    // up via timvx_runtime::reduce_sum (fp32 path on the PPU, which
    // does handle arbitrary sizes).
    constexpr int64_t kMaxPoolKernel = 64;
    if (kernelH > kMaxPoolKernel || kernelW > kMaxPoolKernel)
      return failure();

    // (7) `cur` is now the spatial fp32 tensor that the reduce_sum chain
    //     consumes. Verify shape; rebuild the chain.
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
    // The rebuilt cast/avg_pool2d/reshape chain uses plain integer storage
    // so the per-pixel const inputs (zp const) can be built as plain int.
    // `castOp`'s users still consume the original `dstTy` (possibly
    // quant.uniform-wrapped); the final reshape's result type matches
    // that so the SSA boundary stays type-stable.
    Type narrowInt = dstElem;

    auto splat4D = [&](double v) {
      auto ty = RankedTensorType::get({1, 1, 1, 1}, fp32);
      auto attr = DenseElementsAttr::get(ty, static_cast<float>(v));
      return tosa::ConstOp::create(rewriter, loc, ty, attr).getResult();
    };

    Value perPixelMul = tosa::MulOp::create(
        rewriter, loc, RankedTensorType::get(spatialShape, fp32),
        cur, splat4D(scaleVal),
        /*shift=*/mulScale.getShift());
    Value perPixelAdd = tosa::AddOp::create(
        rewriter, loc, RankedTensorType::get(spatialShape, fp32),
        perPixelMul, splat4D(zpVal));
    Value perPixelInt = tosa::CastOp::create(
        rewriter, loc, RankedTensorType::get(spatialShape, narrowInt),
        perPixelAdd);
    perPixelInt.getDefiningOp()->setAttr("timvx.output_scale", castScale);
    perPixelInt.getDefiningOp()->setAttr("timvx.output_zp", castZp);

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
    // Reshape lands on plain integer storage. If the original `castOp`
    // result type is `quant.uniform<iN:fXX, S:Z>` (tflite-tagged
    // upstream), append a `tosa.cast` int→quant.uniform to match the
    // SSA users' expected type. `tosa-quant-anchor`'s strip phase will
    // peel the quant.uniform off the cast output, leaving a same-width
    // int→int cast that canonicalize folds away.
    auto reshapedTy = RankedTensorType::get(finalShape, narrowInt);
    Value reshaped = tosa::ReshapeOp::create(rewriter, loc, reshapedTy,
                                               pooled, shapeConst);
    Value finalVal = reshaped;
    if (finalTy != reshapedTy)
      finalVal = tosa::CastOp::create(rewriter, loc, finalTy, reshaped);

    rewriter.replaceOp(castOp, finalVal);
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

} // namespace timvx
} // namespace mlir
