//===- TimvxArithFold.cpp - timvx-arith-fold --------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Collapse chains of `add` / `sub` / `multiply` whose RHS is a scalar
// f32 splat const, by const-folding the two scalars into one and
// emitting a single combined op. Targets the f32-space dequant /
// requant chain that surrounds every residual block:
//
//     cast → sub(_, 128.0)  ← injected by timvx-promote-i8-to-u8
//          → sub(_, zp_f32) ← from the original TFLite dequant
//          → multiply(_, scale)
//          → ...
//          → multiply(_, 1/Sout)
//          → add(_, Zout_f32)
//          → add(_, -128.0)  ← injected by promote
//          → cast(f32 → u8)
//
// The two `sub` ops at the head and the two `add` ops at the tail can
// each be folded into one op with the constants combined at compile
// time, eliminating one GC kernel dispatch per chain. Resnet50 has 16
// residual chains × 2 fold sites per chain = ~32 dispatches saved.
// Pairs combined:
//
//   sub(sub(x, a), b) → sub(x, a + b)
//   add(add(x, a), b) → add(x, a + b)
//   add(sub(x, a), b) → add(x, b - a)         // = (x - a) + b
//   sub(add(x, a), b) → add(x, a - b)         // = (x + a) - b
//   multiply(multiply(x, a), b) → multiply(x, a * b)
//
// Constraints:
//   * Both constants must be 1-element f32 splat values (`tensor<1xf32>`
//     or `tensor<1x1x1x1xf32>` etc.) reachable via
//     `matchConstScalarFloat`. The patterns deliberately do NOT handle
//     per-channel broadcast consts — `multiply(x, [1×1×1×C])` followed
//     by another `multiply(x, [1×1×1×C'])` would need a same-shape
//     fold (vector × vector), which is correct math but more code; we
//     don't see that shape in tflite chains.
//   * The inner op must have `hasOneUse()` on its result. If the
//     intermediate is shared with another consumer, folding would
//     remove an op the other consumer depends on without saving a
//     dispatch — same cost, lost intermediate. The `timvx-cse` pass
//     is the right tool for that direction (it removes duplicates of
//     the *same* sub-expression, not chains).
//
// Pattern symmetry — Add and Multiply are commutative; the matcher
// checks both `(rhs is scalar, lhs is inner op)` and `(lhs is scalar,
// rhs is inner op)` for those. Sub is non-commutative; the inner
// const must be on the RHS for the algebraic identities to hold.
//
// Run order: `... → timvx-canonicalize-transpose → canonicalize →
//             **timvx-arith-fold** → canonicalize → timvx-cse → ...`.
// We rely on canonicalize before/after to fold `add x, 0.0` and
// `multiply x, 1.0` to identity (the standard upstream tosa/arith
// canonicalizers do this on f32 splat zero/one).
//
//===----------------------------------------------------------------------===//

#include "Common.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace timvx {

#define GEN_PASS_DEF_TIMVXARITHFOLDPASS
#include "TIMVX/TIMVXPasses.h.inc"

namespace {
using namespace ::mlir::timvx::detail;

// Build a `timvx.const` whose dense storage is a single f32 element
// equal to `value`. The result type matches the original outer-op's
// scalar-const operand type (typically `tensor<1x1x1x1xf32>` from
// promote / dequant chains, or `tensor<1xf32>` from the QRF tail).
// Reusing the original const's type keeps broadcast semantics
// unchanged at the consumer.
static Value buildScalarF32Const(PatternRewriter &rewriter, Location loc,
                                  Type templateType, float value) {
  auto ty = cast<RankedTensorType>(templateType);
  auto attr = DenseElementsAttr::get(ty, ArrayRef<float>{value});
  return ConstOp::create(rewriter, loc, ty, attr,
                          /*quant_scale=*/FloatAttr{},
                          /*quant_zp=*/IntegerAttr{});
}

// Identify which (if any) operand of a 2-input op is a scalar f32 const,
// and return the OTHER operand plus the scalar value. Returns nullopt
// if neither operand is a scalar f32 const.
//
// `commutative` flag controls whether to consider operand1 as the
// const slot: for Sub the const must be in operand2 (the subtrahend
// in `a - b`); for Add / Multiply either side is fine because both
// ops are commutative.
struct ScalarSplit {
  Value other;
  double constVal;
  Type constTy;
};
static std::optional<ScalarSplit> splitScalarConst(Value lhs, Value rhs,
                                                     bool commutative) {
  if (auto v = matchConstScalarFloat(rhs))
    return ScalarSplit{lhs, *v, rhs.getType()};
  if (commutative)
    if (auto v = matchConstScalarFloat(lhs))
      return ScalarSplit{rhs, *v, lhs.getType()};
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// sub(sub(x, a), b) → sub(x, a + b)
//===----------------------------------------------------------------------===//

struct FoldSubOfSub : public OpRewritePattern<SubOp> {
  using OpRewritePattern<SubOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SubOp outer,
                                 PatternRewriter &rewriter) const final {
    auto outerConst = matchConstScalarFloat(outer.getInput2());
    if (!outerConst) return failure();
    auto inner = outer.getInput1().getDefiningOp<SubOp>();
    if (!inner || !inner->hasOneUse()) return failure();
    auto innerConst = matchConstScalarFloat(inner.getInput2());
    if (!innerConst) return failure();

    double combined = *innerConst + *outerConst;
    Value newConst = buildScalarF32Const(rewriter, outer.getLoc(),
                                          inner.getInput2().getType(),
                                          static_cast<float>(combined));
    rewriter.replaceOpWithNewOp<SubOp>(outer, outer.getType(),
                                        inner.getInput1(), newConst);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// add(add(x, a), b) → add(x, a + b)        (Add commutative both sides)
//===----------------------------------------------------------------------===//

struct FoldAddOfAdd : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp outer,
                                 PatternRewriter &rewriter) const final {
    auto outerSplit = splitScalarConst(outer.getInput1(), outer.getInput2(),
                                         /*commutative=*/true);
    if (!outerSplit) return failure();
    auto inner = outerSplit->other.getDefiningOp<AddOp>();
    if (!inner || !inner->hasOneUse()) return failure();
    auto innerSplit = splitScalarConst(inner.getInput1(), inner.getInput2(),
                                         /*commutative=*/true);
    if (!innerSplit) return failure();

    double combined = innerSplit->constVal + outerSplit->constVal;
    Value newConst = buildScalarF32Const(rewriter, outer.getLoc(),
                                          innerSplit->constTy,
                                          static_cast<float>(combined));
    rewriter.replaceOpWithNewOp<AddOp>(outer, outer.getType(),
                                        innerSplit->other, newConst);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// add(sub(x, a), b) → add(x, b - a)
//===----------------------------------------------------------------------===//
//
// The outer add is commutative, so the inner sub may be on either of
// its operands. The inner sub itself is not commutative.

struct FoldAddOfSub : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp outer,
                                 PatternRewriter &rewriter) const final {
    auto outerSplit = splitScalarConst(outer.getInput1(), outer.getInput2(),
                                         /*commutative=*/true);
    if (!outerSplit) return failure();
    auto inner = outerSplit->other.getDefiningOp<SubOp>();
    if (!inner || !inner->hasOneUse()) return failure();
    auto innerConst = matchConstScalarFloat(inner.getInput2());
    if (!innerConst) return failure();

    double combined = outerSplit->constVal - *innerConst;
    Value newConst = buildScalarF32Const(rewriter, outer.getLoc(),
                                          inner.getInput2().getType(),
                                          static_cast<float>(combined));
    rewriter.replaceOpWithNewOp<AddOp>(outer, outer.getType(),
                                        inner.getInput1(), newConst);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// sub(add(x, a), b) → add(x, a - b)
//===----------------------------------------------------------------------===//

struct FoldSubOfAdd : public OpRewritePattern<SubOp> {
  using OpRewritePattern<SubOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SubOp outer,
                                 PatternRewriter &rewriter) const final {
    auto outerConst = matchConstScalarFloat(outer.getInput2());
    if (!outerConst) return failure();
    auto inner = outer.getInput1().getDefiningOp<AddOp>();
    if (!inner || !inner->hasOneUse()) return failure();
    auto innerSplit = splitScalarConst(inner.getInput1(), inner.getInput2(),
                                         /*commutative=*/true);
    if (!innerSplit) return failure();

    double combined = innerSplit->constVal - *outerConst;
    Value newConst = buildScalarF32Const(rewriter, outer.getLoc(),
                                          innerSplit->constTy,
                                          static_cast<float>(combined));
    rewriter.replaceOpWithNewOp<AddOp>(outer, outer.getType(),
                                        innerSplit->other, newConst);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// multiply(multiply(x, a), b) → multiply(x, a * b)
//===----------------------------------------------------------------------===//

struct FoldMulOfMul : public OpRewritePattern<MultiplyOp> {
  using OpRewritePattern<MultiplyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MultiplyOp outer,
                                 PatternRewriter &rewriter) const final {
    auto outerSplit = splitScalarConst(outer.getInput1(), outer.getInput2(),
                                         /*commutative=*/true);
    if (!outerSplit) return failure();
    auto inner = outerSplit->other.getDefiningOp<MultiplyOp>();
    if (!inner || !inner->hasOneUse()) return failure();
    auto innerSplit = splitScalarConst(inner.getInput1(), inner.getInput2(),
                                         /*commutative=*/true);
    if (!innerSplit) return failure();

    double combined = innerSplit->constVal * outerSplit->constVal;
    Value newConst = buildScalarF32Const(rewriter, outer.getLoc(),
                                          innerSplit->constTy,
                                          static_cast<float>(combined));
    rewriter.replaceOpWithNewOp<MultiplyOp>(outer, outer.getType(),
                                              innerSplit->other, newConst);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct TIMVXArithFoldPass
    : public impl::TIMVXArithFoldPassBase<TIMVXArithFoldPass> {
  void runOnOperation() final {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<FoldSubOfSub, FoldAddOfAdd, FoldAddOfSub, FoldSubOfAdd,
                  FoldMulOfMul>(ctx);
    // Disable the greedy driver's constant CSE: the per-pattern
    // `buildScalarF32Const` emits a fresh `timvx.const` each time and
    // CSE-merging two consts that happen to share the same `values`
    // attr but differ in (currently-absent) `quant_scale` / `quant_zp`
    // would breach the convention spelled out in `TIMVX_ConstOp`'s
    // td comment. `timvx-cse` does the structural CSE explicitly with
    // attribute-aware equality.
    GreedyRewriteConfig cfg;
    cfg.enableConstantCSE(false);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                       cfg)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createTIMVXArithFoldPass() {
  return std::make_unique<TIMVXArithFoldPass>();
}

} // namespace timvx
} // namespace mlir
