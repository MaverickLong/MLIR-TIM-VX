//===- TimvxCanonicalizeTranspose.cpp - timvx-canonicalize-transpose -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Push `timvx.transpose` ops upward through see-through producers until
// they cancel against the other half of a sandwich pair (`{3,1,0,2}` after
// a WHCN-native spatial op meets the `{2,1,3,0}` going into the next
// spatial op — `composed = identity`, both transposes erased by the
// existing `TransposeOfTransposeFold` peephole on the composed perm).
//
// What "upward" means here: a transpose `T` whose input is the result of
// some elementwise op `E(operands...)` becomes `E(T(operand_0),
// T(operand_1), ...)` — same op, applied to transposed operands, with the
// result type permuted by `T.perms`. The outer `T` is gone; the new `T`s
// in front of each operand continue propagating in the next iteration of
// the greedy driver.
//
// See-through producers handled (one rewrite pattern per kind):
//
//   * Shape-preserving unary: `cast`, `dataconvert`, `clip`.
//     `T(U(x)) → U(T(x))`. Optional `output_scale`/`output_zp` attrs are
//     preserved on the rewritten op.
//
//   * Elementwise binary: `add`, `sub`, `multiply`. Both operands get
//     their own transpose wrap; 1×…×1 scalar broadcast consts permute to
//     themselves (the existing `TransposeOfConstFold` canonicalizer
//     erases the no-op transpose on those).
//
//   * `slice`: `T(slice(t, start, size)) → slice(T(t), perm(start),
//     perm(size))`. The `start` and `size` operands come from
//     `timvx.const_shape` (`tensor<Nxindex>`); the rewrite emits new
//     `const_shape` ops with the dense values permuted by `perm`. If
//     either operand isn't a const_shape, the rewrite bails — a runtime-
//     computed slice can't be permuted at compile time.
//
// Barriers (no pattern; transpose halts here):
//
//   * `conv2d`, `pool2d`, `fully_connected`: each owns a WHCN layout
//     contract that the transpose is precisely there to satisfy.
//   * `reshape`: layout-dependent; pushing a transpose through it
//     changes which axes get coalesced. Resnet18's reshapes are at the
//     pre-FC tail where the transpose count is already small, so
//     blocking is fine.
//   * `reduce_sum`, `pad`: axis-indexed; not currently emitted in the
//     middle of a transpose-elimination opportunity.
//   * `func.func` arguments and any multi-use producer: pushing past a
//     fan-out would re-permute every other consumer's view of the data.
//
// The existing canonicalizers on `timvx.transpose` (in `TIMVXOps.cpp`)
// do the cancellation work and live in the same pattern set:
//
//   * `TransposeOfConstFold`: `T(const) → const'` (permuted dense data).
//     Fires on transposes that land on a `timvx.const` after propagation
//     has carried them down to an operand-const.
//   * `TransposeOfTransposeFold`: `T1(T2(x)) → T_composed(x)` or, when
//     the composed perm is identity, `x`. Fires when the two halves of
//     a sandwich pair finally meet (e.g. after both walk up to the same
//     `timvx.add` and become its two operand transposes, which then both
//     walk up to the prior spatial op's post-transpose). This is what
//     actually retires the sandwich.
//
// The pass uses the greedy driver with default folding/constant CSE so
// the existing op folders keep working. No discardable attrs are read
// or written; quant-relevant `output_scale`/`output_zp` attrs are passed
// through unchanged on rewrites that emit a new op of the same kind.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace timvx {

#define GEN_PASS_DEF_TIMVXCANONICALIZETRANSPOSEPASS
#include "TIMVX/TIMVXPasses.h.inc"

namespace {
using namespace ::mlir::timvx::detail;

// Permute a ranked-tensor type's shape by `perm`. Element type and
// `quant.uniform` element-type stamp (if any) carry through verbatim —
// transpose is value-preserving, so the (S, Z) stays the same.
static RankedTensorType permTensorType(RankedTensorType ty,
                                        ArrayRef<int32_t> perm) {
  auto src = ty.getShape();
  SmallVector<int64_t> out;
  out.reserve(perm.size());
  for (int32_t p : perm)
    out.push_back(src[p]);
  return ty.clone(out);
}

// Build a `timvx.transpose %v {perms = perm}` whose result type is `v`'s
// shape permuted by `perm`. The optional (`output_scale`, `output_zp`)
// gets copied off the original transpose so the wrapped operand carries
// the quant context downstream codegen expects to see at this point in
// the chain — typically the producer op already stamps this on its
// result, but propagating through an op that doesn't (e.g. `timvx.const`
// without quant_*) keeps the consumer well-typed.
static Value buildWrap(OpBuilder &b, Location loc, Value v,
                       ArrayRef<int32_t> perm,
                       FloatAttr scaleAttr, IntegerAttr zpAttr) {
  auto vTy = cast<RankedTensorType>(v.getType());
  auto dstTy = permTensorType(vTy, perm);
  return TransposeOp::create(b, loc, dstTy, v,
                              b.getDenseI32ArrayAttr(perm),
                              scaleAttr, zpAttr);
}

//===----------------------------------------------------------------------===//
// Shape-preserving unary producers: cast / dataconvert / clip
//===----------------------------------------------------------------------===//

template <typename UnaryOp>
struct PushTransposeThroughUnary : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern<TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TransposeOp op,
                                 PatternRewriter &rewriter) const final {
    auto producer = op.getInput1().getDefiningOp<UnaryOp>();
    if (!producer)
      return failure();

    // We do NOT bail on multi-use producers: the rewrite only replaces
    // the transpose op, not the producer, so other consumers of the
    // producer stay valid (they keep reading the original shape). At
    // worst we duplicate the producer's op (one in the original layout
    // for the other consumers, one in the permuted layout for the
    // propagating chain). Resnet-style fan-outs occur at the clip
    // output between the main path (→ next conv) and the skip path
    // (→ next residual add); duplicating a `clip` so the transpose can
    // propagate further is cheaper than leaving the boundary transpose
    // in place.
    auto perm = op.getPerms();
    Value wrapped =
        buildWrap(rewriter, op.getLoc(), producer->getOperand(0), perm,
                  op.getOutputScaleAttr(), op.getOutputZpAttr());
    auto newOp = UnaryOp::create(rewriter, producer.getLoc(),
                                  op.getType(), wrapped,
                                  producer->getAttrs());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

// Clip carries `min_val` / `max_val` F32Attrs (not optional quant attrs).
// Same propagation, just keep its attrs.
struct PushTransposeThroughClip : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern<TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TransposeOp op,
                                 PatternRewriter &rewriter) const final {
    auto producer = op.getInput1().getDefiningOp<ClipOp>();
    if (!producer)
      return failure();

    auto perm = op.getPerms();
    Value wrapped = buildWrap(rewriter, op.getLoc(), producer.getInput(),
                              perm, /*scale=*/FloatAttr{},
                              /*zp=*/IntegerAttr{});
    rewriter.replaceOpWithNewOp<ClipOp>(
        op, op.getType(), wrapped,
        producer.getMinValAttr(), producer.getMaxValAttr());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Elementwise binary producers: add / sub / multiply
//===----------------------------------------------------------------------===//
//
// `T(binary(a, b)) → binary(T(a), T(b))`. Both operands get wrapped.
// `T(const_with_numel_1)` is folded to a same-shape permuted const by
// `TransposeOfConstFold` — that op stays scalar and the wrap evaporates.
// `T(transpose(x, q))` is folded to `T_composed(x)` by
// `TransposeOfTransposeFold`. So the cleanup is automatic.

template <typename BinaryOp>
struct PushTransposeThroughBinary : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern<TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TransposeOp op,
                                 PatternRewriter &rewriter) const final {
    auto producer = op.getInput1().getDefiningOp<BinaryOp>();
    if (!producer)
      return failure();

    // Per-operand handling:
    //   * rank == perm size: wrap in transpose; the wrap folds when the
    //     operand is a const or another transpose with inverse perm.
    //   * numel == 1 (true broadcast scalar, any rank): leave alone.
    //     A scalar broadcasts identically to every position regardless
    //     of how the other operand is permuted, so a rank-N + rank-1
    //     stays well-formed after we permute only the rank-N side.
    //     This catches every tflite chain-const shape we emit
    //     (`tensor<1xf32>` zp, `tensor<1x1x1x1xf32>` scale, ...).
    //   * everything else (e.g. [1,1,1,C] per-channel const with a
    //     non-unit dim that DOES move under perm): bail. Permuting the
    //     C dim out of its position would change which axis the
    //     broadcast targets.
    auto perm = op.getPerms();
    auto isScalarBcast = [](Value v) {
      auto rt = dyn_cast<RankedTensorType>(v.getType());
      if (!rt || !rt.hasStaticShape()) return false;
      return rt.getNumElements() == 1;
    };
    auto isFullRank = [&](Value v) {
      auto rt = dyn_cast<RankedTensorType>(v.getType());
      return rt && rt.getRank() == static_cast<int64_t>(perm.size());
    };
    Value lhs = producer.getInput1();
    Value rhs = producer.getInput2();
    if (!(isFullRank(lhs) || isScalarBcast(lhs))) return failure();
    if (!(isFullRank(rhs) || isScalarBcast(rhs))) return failure();

    auto loc = op.getLoc();
    Value a = isFullRank(lhs)
                  ? buildWrap(rewriter, loc, lhs, perm,
                              /*scale=*/FloatAttr{}, /*zp=*/IntegerAttr{})
                  : lhs;
    Value b = isFullRank(rhs)
                  ? buildWrap(rewriter, loc, rhs, perm,
                              /*scale=*/FloatAttr{}, /*zp=*/IntegerAttr{})
                  : rhs;
    rewriter.replaceOpWithNewOp<BinaryOp>(op, op.getType(), a, b);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Slice producer
//===----------------------------------------------------------------------===//

struct PushTransposeThroughSlice : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern<TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TransposeOp op,
                                 PatternRewriter &rewriter) const final {
    auto producer = op.getInput1().getDefiningOp<SliceOp>();
    if (!producer)
      return failure();

    // start / size must both be timvx.const_shape — runtime-computed
    // bounds can't be permuted at compile time.
    auto startCS = producer.getStart().getDefiningOp<ConstShapeOp>();
    auto sizeCS  = producer.getSize().getDefiningOp<ConstShapeOp>();
    if (!startCS || !sizeCS)
      return failure();

    auto startVals = dyn_cast<DenseIntElementsAttr>(startCS.getValuesAttr());
    auto sizeVals  = dyn_cast<DenseIntElementsAttr>(sizeCS.getValuesAttr());
    if (!startVals || !sizeVals)
      return failure();

    auto perm = op.getPerms();
    if (startVals.getNumElements() != static_cast<int64_t>(perm.size()) ||
        sizeVals.getNumElements() != static_cast<int64_t>(perm.size()))
      return failure();

    // Permute the dense values: new[i] = old[perm[i]].
    auto startVec = llvm::to_vector(startVals.getValues<APInt>());
    auto sizeVec  = llvm::to_vector(sizeVals.getValues<APInt>());
    SmallVector<APInt> newStart(perm.size(), APInt(64, 0));
    SmallVector<APInt> newSize(perm.size(), APInt(64, 0));
    for (size_t i = 0; i < perm.size(); ++i) {
      newStart[i] = startVec[perm[i]];
      newSize[i]  = sizeVec[perm[i]];
    }

    auto idxTy = rewriter.getIndexType();
    auto shapeTy = RankedTensorType::get(
        {static_cast<int64_t>(perm.size())}, idxTy);
    auto newStartAttr = DenseIntElementsAttr::get(shapeTy, newStart);
    auto newSizeAttr  = DenseIntElementsAttr::get(shapeTy, newSize);

    auto loc = op.getLoc();
    Value newStartV =
        ConstShapeOp::create(rewriter, loc, shapeTy, newStartAttr);
    Value newSizeV =
        ConstShapeOp::create(rewriter, loc, shapeTy, newSizeAttr);

    Value wrappedIn = buildWrap(rewriter, loc, producer.getInput1(), perm,
                                /*scale=*/FloatAttr{},
                                /*zp=*/IntegerAttr{});
    rewriter.replaceOpWithNewOp<SliceOp>(
        op, op.getType(), wrappedIn, newStartV, newSizeV,
        producer.getOutputScaleAttr(), producer.getOutputZpAttr());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct TIMVXCanonicalizeTransposePass
    : public impl::TIMVXCanonicalizeTransposePassBase<
          TIMVXCanonicalizeTransposePass> {
  void runOnOperation() final {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    // Propagation patterns (push transpose upward through producers).
    patterns.add<PushTransposeThroughUnary<CastOp>>(ctx);
    patterns.add<PushTransposeThroughUnary<DataConvertOp>>(ctx);
    patterns.add<PushTransposeThroughClip>(ctx);
    patterns.add<PushTransposeThroughBinary<AddOp>>(ctx);
    patterns.add<PushTransposeThroughBinary<SubOp>>(ctx);
    patterns.add<PushTransposeThroughBinary<MultiplyOp>>(ctx);
    patterns.add<PushTransposeThroughSlice>(ctx);

    // Cancellation peepholes: `TransposeOp::getCanonicalizationPatterns`
    // registers `TransposeOfConstFold` and `TransposeOfTransposeFold`.
    // Run them in the same fixpoint so each newly-created
    // `transpose(operand)` either folds into a permuted const, composes
    // with a pre-existing transpose, or stops at a barrier.
    TransposeOp::getCanonicalizationPatterns(patterns, ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createTIMVXCanonicalizeTransposePass() {
  return std::make_unique<TIMVXCanonicalizeTransposePass>();
}

} // namespace timvx
} // namespace mlir
