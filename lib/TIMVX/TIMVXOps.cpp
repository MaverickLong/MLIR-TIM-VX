//===- TIMVXOps.cpp - TIMVX dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TIMVX/TIMVXOps.h"
#include "TIMVX/TIMVXDialect.h"

#include "Passes/Common.h"

#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/LogicalResult.h"

#define GET_OP_CLASSES
#include "TIMVX/TIMVXOps.cpp.inc"

namespace mlir {
namespace timvx {

// Folders for ConstantLike ops.
OpFoldResult ConstOp::fold(FoldAdaptor) { return getValuesAttr(); }
OpFoldResult ConstShapeOp::fold(FoldAdaptor) { return getValuesAttr(); }

//===----------------------------------------------------------------------===//
// TransposeOp canonicalizations
//===----------------------------------------------------------------------===//
//
// Two compile-time peepholes that the spatial-op wrapper relies on for
// correctness, not just performance:
//
//   * `transpose(const) -> permuted const` — TIM-VX's NN-engine kernel
//     selector refuses to bind a CONSTANT weight tensor through a runtime
//     transpose; the shader-compile stage will COMPILE_FAIL otherwise.
//   * `transpose(transpose(x))` -> `transpose(x)` with composed perms.
//     Coalesces back-to-back boundary transposes (no NHWC<->WHCN dance
//     between adjacent spatial ops).
//
// Both run via the standard `--canonicalize` pipeline.

namespace {
using namespace ::mlir::timvx::detail;

struct TransposeOfConstFold : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern<TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TransposeOp op,
                                  PatternRewriter &rewriter) const final {
    auto producer = op.getInput1().getDefiningOp<ConstOp>();
    if (!producer)
      return rewriter.notifyMatchFailure(op, "input is not a timvx.const");
    auto values = dyn_cast<ElementsAttr>(producer.getValuesAttr());
    if (!values)
      return rewriter.notifyMatchFailure(
          op, "const values are not ElementsAttr (resource w/o backing data?)");

    auto permI64 = permToInt64(op.getPerms());
    auto permuted = permuteDenseElements(values, permI64);
    if (!permuted)
      return rewriter.notifyMatchFailure(
          op, "permuteDenseElements returned null (unsupported dtype?)");

    rewriter.replaceOpWithNewOp<ConstOp>(op, op.getType(), permuted,
                                          producer.getQuantScaleAttr(),
                                          producer.getQuantZpAttr());
    return success();
  }
};

struct TransposeOfTransposeFold : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern<TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TransposeOp op,
                                  PatternRewriter &rewriter) const final {
    auto inner = op.getInput1().getDefiningOp<TransposeOp>();
    if (!inner)
      return rewriter.notifyMatchFailure(op, "input is not a timvx.transpose");

    auto outer = op.getPerms();
    auto innerPerms = inner.getPerms();
    if (outer.size() != innerPerms.size())
      return rewriter.notifyMatchFailure(op, "perm rank mismatch");

    // Compose: result_axis_i = inner_input_axis[inner[outer[i]]].
    SmallVector<int32_t> composed(outer.size());
    for (size_t i = 0; i < outer.size(); ++i)
      composed[i] = innerPerms[outer[i]];

    bool identity = true;
    for (size_t i = 0; i < composed.size(); ++i) {
      if (composed[i] != static_cast<int32_t>(i)) {
        identity = false;
        break;
      }
    }
    if (identity) {
      rewriter.replaceOp(op, inner.getInput1());
      return success();
    }

    rewriter.replaceOpWithNewOp<TransposeOp>(
        op, op.getType(), inner.getInput1(),
        rewriter.getDenseI32ArrayAttr(composed),
        op.getOutputScaleAttr(), op.getOutputZpAttr());
    return success();
  }
};

} // namespace

void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *ctx) {
  results.add<TransposeOfConstFold, TransposeOfTransposeFold>(ctx);
}

} // namespace timvx
} // namespace mlir