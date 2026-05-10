//===- TosaConstFold.cpp - tosa-const-fold pass --------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pure tosa->tosa rewrites that fold elementwise / reshape ops on constant
// tensors. Run via the greedy driver to fixed point, so chains collapse
// inside-out (e.g. BatchNorm's `var -> +eps -> ^0.5 -> 1/x -> *gamma`).
//
// Why this pass exists:  TIM-VX's eltwise kernels reject same-rank,
// different-size broadcast (`tensor<1xf32> + tensor<64xf32>`) even though
// the per-dim compatibility check would let it through. The BatchNorm
// scalar chain hits exactly that pattern, and every op in the chain has
// constant operands — so the cleanest fix is to fold them away at the
// tosa level before lowering.
//
// FP32 only (matches the immediate need).
//
// The pass also includes the int-rewrite peepholes that pre-canonicalize
// the IR before `tosa-to-timvx`:
//
//   * `PadFoldIntoConv`: absorb `tosa.pad -> tosa.conv2d/max_pool` into
//     the consumer's `pad` attribute (vivante.nn.tensor.pad COMPILE_FAILs
//     on this hardware; the spatial op carries pad internally).
//   * `PadRescaleSwap`: swap `pad -> rescale` so the pad becomes
//     `rescale -> pad` and `PadFoldIntoConv` can subsume it.
//   * `RequantI32SkipFold`: rewrite the f32->i32->add(zp)->cast(narrow)
//     quantize tail to f32->add(zp_f32)->cast(narrow), avoiding the
//     near-dead int32 column on this NPU.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace timvx {

#define GEN_PASS_DEF_TOSACONSTFOLDPASS
#include "TIMVX/TIMVXPasses.h.inc"

namespace {
using namespace ::mlir::timvx::detail;

//===----------------------------------------------------------------------===//
// Elementwise / reshape FP32 const folders
//===----------------------------------------------------------------------===//

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
// common for large weight tensors) and emit a fresh tosa.const at the
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
// Pad / rescale / requantize-tail rewrites
//===----------------------------------------------------------------------===//

// Fold `tosa.pad -> tosa.conv2d` (or `tosa.pad -> tosa.max_pool2d`) into
// the consumer's `pad` attribute. The standalone NN-core pad kernel
// (`vivante.nn.tensor.pad`) refuses to initialize on this hardware, but
// conv2d / pool2d carry their own padding internally — so absorbing the
// pad into the spatial op sidesteps that codepath.
struct PadFoldIntoConv : public OpRewritePattern<tosa::PadOp> {
  using OpRewritePattern<tosa::PadOp>::OpRewritePattern;

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
    if ((*pairs)[0] != 0 || (*pairs)[1] != 0 ||
        (*pairs)[6] != 0 || (*pairs)[7] != 0)
      return failure(); // non-spatial padding can't be folded into conv/pool.

    int64_t hFront = (*pairs)[2], hBack = (*pairs)[3];
    int64_t wFront = (*pairs)[4], wBack = (*pairs)[5];

    if (auto conv = dyn_cast<tosa::Conv2DOp>(user)) {
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

// Swap `tosa.pad -> tosa.rescale` to `tosa.rescale -> tosa.pad`. With the
// pad moved past the rescale, `PadFoldIntoConv` can fold it into the
// consuming conv / max_pool's spatial pad attribute.
//
// Soundness: pad fills with `pad_const = Zi` (= real-value 0 in input
// quant). After rescale, the same real-value 0 corresponds to `Zo` in
// output quant. So moving pad to after the rescale and changing
// `pad_const` to Zo preserves real-value semantics.
struct PadRescaleSwap : public OpRewritePattern<tosa::RescaleOp> {
  using OpRewritePattern<tosa::RescaleOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::RescaleOp resc,
                                 PatternRewriter &rewriter) const final {
    auto pad = resc.getInput().getDefiningOp<tosa::PadOp>();
    if (!pad || !pad->hasOneUse()) return failure();
    if (resc.getPerChannel()) return failure();

    auto outZpVal = matchConstScalarInt(resc.getOutputZp());
    if (!outZpVal) return failure();

    auto unpaddedTy = dyn_cast<RankedTensorType>(pad.getInput1().getType());
    auto paddedDstTy = dyn_cast<RankedTensorType>(resc.getType());
    if (!unpaddedTy || !paddedDstTy) return failure();
    auto outElemTy = dyn_cast<IntegerType>(paddedDstTy.getElementType());
    if (!outElemTy) return failure();

    Location loc = resc.getLoc();

    auto newRescTy = RankedTensorType::get(unpaddedTy.getShape(), outElemTy);
    Value newResc = tosa::RescaleOp::create(
        rewriter, loc, newRescTy, pad.getInput1(), resc.getMultiplier(),
        resc.getShift(), resc.getInputZp(), resc.getOutputZp(),
        resc.getScale32Attr(), resc.getRoundingModeAttr(),
        resc.getPerChannelAttr(), resc.getInputUnsignedAttr(),
        resc.getOutputUnsignedAttr());

    auto padConstTy = RankedTensorType::get({1}, outElemTy);
    APInt padVal(outElemTy.getWidth(), static_cast<uint64_t>(*outZpVal),
                 /*isSigned=*/true);
    auto padConstAttr = DenseIntElementsAttr::get(padConstTy, padVal);
    Value newPadConst =
        tosa::ConstOp::create(rewriter, loc, padConstTy, padConstAttr);

    Value newPad = tosa::PadOp::create(rewriter, loc, paddedDstTy, newResc,
                                         pad.getPadding(), newPadConst);

    rewriter.replaceOp(resc, newPad);
    return success();
  }
};

// Skip the i32 detour in TOSA's quantize-tail decomposition. The chain
// TOSA's rescale lowering emits is:
//
//   %a   = tosa.mul %x, %inv_scale_f32, %shift   : f32   (1/output_scale)
//   %b   = tosa.cast %a   : f32 -> i32
//   %z   = tosa.const dense<zp_int>              : i32   (rank-0 / broadcast)
//   %c   = tosa.add %b, %z                       : i32
//   %out = tosa.cast %c   : i32 -> i8 (or u8/i16) : narrow int
//
// On VIP9000Nano-DI plain int32 is a near-dead column. The equivalent fp32
// form runs end-to-end on supported kernels:
//
//   %z_f = tosa.const dense<float(zp_int)>       : f32   (same shape as %z)
//   %c_f = tosa.add %a, %z_f                     : f32
//   %out = tosa.cast %c_f : f32 -> narrow int
//
// We also recover the (output_scale, output_zp) pair here and stash them
// on the new `tosa.cast` as discardable `timvx.output_scale` /
// `timvx.output_zp` attributes. `buildQuantInfoMap` reads them so
// `CastConversion` can attach the right Quantization() downstream.
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

    auto zpConst = zpSide.getDefiningOp<tosa::ConstOp>();
    if (!zpConst) return failure();
    auto zpAttr = dyn_cast<DenseIntElementsAttr>(zpConst.getValuesAttr());
    if (!zpAttr || !zpAttr.getElementType().isInteger(32)) return failure();
    auto zpTy = dyn_cast<RankedTensorType>(zpConst.getType());
    if (!zpTy || zpAttr.getNumElements() < 1) return failure();
    int64_t zpScalar =
        (*zpAttr.getValues<APInt>().begin()).getSExtValue();

    // Try to recover the upstream multiplier const so we can derive the
    // output scale that downstream conv2d expects on its input.
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
    (void)mulValue;

    SmallVector<float> zpFp;
    zpFp.reserve(zpAttr.getNumElements());
    for (APInt v : zpAttr.getValues<APInt>())
      zpFp.push_back(static_cast<float>(v.getSExtValue()));
    auto f32 = rewriter.getF32Type();
    auto zpFpTy = RankedTensorType::get(zpTy.getShape(), f32);
    auto zpFpConst = tosa::ConstOp::create(
        rewriter, zpConst.getLoc(), zpFpTy,
        DenseElementsAttr::get(zpFpTy, ArrayRef<float>(zpFp)));

    auto i32AddTy = dyn_cast<RankedTensorType>(add.getType());
    if (!i32AddTy) return failure();
    auto fAddTy = RankedTensorType::get(i32AddTy.getShape(), f32);
    auto newAdd = tosa::AddOp::create(rewriter, add.getLoc(), fAddTy,
                                       innerCast.getInput(),
                                       zpFpConst.getResult());

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

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

struct TosaConstFoldPass
    : public impl::TosaConstFoldPassBase<TosaConstFoldPass> {
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<AddConstFold, SubConstFold, MulConstFold, PowConstFold,
                 ReciprocalConstFold, ReshapeConstFold, PadFoldIntoConv,
                 PadRescaleSwap, RequantI32SkipFold>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createTosaConstFoldPass() {
  return std::make_unique<TosaConstFoldPass>();
}

} // namespace timvx
} // namespace mlir
