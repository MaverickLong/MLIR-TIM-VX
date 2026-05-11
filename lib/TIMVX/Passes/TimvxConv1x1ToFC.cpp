//===- TimvxConv1x1ToFC.cpp - timvx-conv1x1-to-fc pass -------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pattern: when a `timvx.conv2d` has W=H=1 input, W=H=1 weight, and unit
// pad/stride/dilation, it's mathematically a fully-connected layer. Routes
// through `timvx.fully_connected` (which has a tighter NN-engine path on
// Vivante NPUs — the weight is pre-tiled at bind vs. Conv2D's per-call
// spatial setup). Triggered by tflite's quantized exporter, which emits
// the trailing classifier as a 1x1 Conv2D + reshape pair.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace timvx {

#define GEN_PASS_DEF_TIMVXCONV1X1TOFCPASS
#include "TIMVX/TIMVXPasses.h.inc"

namespace {
using namespace ::mlir::timvx::detail;

// Match a 1x1 timvx.conv2d and rewrite to reshape -> fully_connected ->
// (optional) reshape.
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

    auto wConst = op.getWeight().getDefiningOp<ConstOp>();
    if (!wConst) return failure();
    auto wAttr = dyn_cast<DenseElementsAttr>(wConst.getValuesAttr());
    if (!wAttr) return failure();

    Location loc = op.getLoc();
    Type elemTy = wTy.getElementType();

    // [1,1,K,M] -> [K,M] — same memory order, dropping size-1 outer dims.
    auto newWTy = RankedTensorType::get({K, M}, elemTy);
    auto newWAttr = wAttr.reshape(newWTy);
    Value newWeight = ConstOp::create(rewriter, loc, newWTy, newWAttr,
                                       wConst.getQuantScaleAttr(),
                                       wConst.getQuantZpAttr());

    // Helper: build a `timvx.reshape` carrying explicit `output_scale`/
    // `output_zp` attrs so `fmtTensorSpec` emits a fully-quantized spec
    // for the new transient. Without these the emitted runtime tensor
    // would land as a plain INT8 (no Quantization()), creating a dtype-
    // vs-quant mismatch with the surrounding u8 ops once the i8|asym->u8
    // promotion fires.
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
    // `output_scale` / `output_zp` attrs.
    FloatAttr inScale;
    IntegerAttr inZp;
    if (auto *defOp = op.getInput().getDefiningOp()) {
      inScale = defOp->getAttrOfType<FloatAttr>("output_scale");
      inZp = defOp->getAttrOfType<IntegerAttr>("output_zp");
    }

    // Reshape input [1,1,K,batch] -> [K,batch].
    //
    // Why [K,batch] instead of the MLIR FC contract [batch,K]: TIM-VX's
    // FC kernel (`vsi_nn_op_fullconnect2.c`) internally reshapes its
    // input to `{K, batch}` (K innermost) and its output to `{M, batch}`
    // (M innermost) before dispatching to `vxFullyConnectedLayer`. Our
    // "TIM-VX shape mirrors MLIR shape" convention places dim 0 as
    // innermost, so feeding the kernel input shape `{batch, K}` makes
    // the kernel walk bytes in batch-fastest order — wrong for batch>1
    // and producing a column-permuted output even at batch=1 because
    // the weight's row-major MLIR-shape mirror also gets the wrong
    // outer/inner roles. Match the kernel's convention explicitly:
    // shape both input and output as `{K, batch}`/`{M, batch}` with
    // the contraction dim at index 0, and use `axis=0` in the runtime
    // FC call.
    Value xR = reshapeWithQuant(op.getInput(), {K, batch}, inScale, inZp);

    // FC: [K,batch] x [K,M] + [M] -> [M,batch].
    auto fcOutTy = RankedTensorType::get({M, batch}, outTy.getElementType());
    Value fc = FullyConnectedOp::create(rewriter, loc, fcOutTy, xR, newWeight,
                                         op.getBias(),
                                         op.getOutputScaleAttr(),
                                         op.getOutputZpAttr());

    // If the conv1x1's only user is a `timvx.reshape`, fuse with it: the
    // typical tflite tail is `conv1x1 [1,1,M,1] -> reshape -> [1,M]`,
    // which would otherwise become `reshape -> FC -> reshape (rank-
    // restore) -> reshape (the user's)` — back-and-forth shape thrashing
    // that TIM-VX rejects at compile (multiple `RESHAPE2` nodes around
    // the FC trip graph->Compile on VIP9000Nano-DI on this chip).
    auto userReshape = op->hasOneUse()
        ? dyn_cast<ReshapeOp>(*op->getUsers().begin())
        : ReshapeOp();
    if (userReshape) {
      auto userOutTy = cast<RankedTensorType>(userReshape.getType());
      Value reshaped = reshapeWithQuant(
          fc, userOutTy.getShape(),
          userReshape.getOutputScaleAttr(),
          userReshape.getOutputZpAttr());
      rewriter.replaceOp(userReshape, reshaped);
      rewriter.eraseOp(op);
      return success();
    }

    // Reshape FC result back to [1,1,M,batch] so downstream consumers see
    // the same type as the original Conv2D output.
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

} // namespace timvx
} // namespace mlir
