//===- TimvxDequantFuse.cpp - timvx-dequant-fuse ----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Collapse the per-operand dequant chain
//
//     %f = timvx.cast %u8q : (<...x!quant.uniform<u8:f32, S:Z>>) -> <...xf32>
//     %n = timvx.sub %f, %zp_const_f32     (omitted when Z == 0 — canonicalize
//                                            folds `sub _, 0.0`)
//     %r = timvx.multiply %n, %scale_const_f32
//
// into a single
//
//     %r = timvx.dataconvert %u8q : (<...x!quant.uniform<u8:f32, S:Z>>) -> <...xf32>
//
// `tim::vx::ops::DataConvert` (vxTensorCopyNode) reads the input tensor's
// (S, Z) at runtime and produces `f32 = (byte - Z) * S` natively — the
// same math the cast+sub+mul triple does, in one NN-engine kernel
// dispatch instead of three PPU kernel launches.
//
// What this pass does NOT handle:
//   * f32 → u8 requant (`mul → add → cast(f32→u8)`): DataConvert
//     COMPILE_FAILs for f32→int on VIP9000 despite the
//     op_check table claiming support. Keep f32→int on `timvx.cast`.
//     See the comment on TIMVX_DataConvertOp in TIMVXOps.td.
//   * Residual-add chains: `QuantResidualFuse` already collapses those
//     into a single quantized `timvx.add`; this pass picks up the
//     leftovers — dequants whose downstream consumer wasn't a
//     residual-style requant tail (e.g. the global-avg-pool input,
//     ops the rescale-fusion couldn't subsume).
//
// Run order: `tosa-to-timvx` → `canonicalize` → `timvx-promote-i8-to-u8`
//            → `timvx-quant-residual-fuse` → `canonicalize`
//            → **`timvx-dequant-fuse`** → `timvx-conv1x1-to-fc` → emitc.
//
// QRF must run first because its pattern relies on the un-fused
// cast+sub+mul shape to recognise the residual-add operand chain.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace timvx {

#define GEN_PASS_DEF_TIMVXDEQUANTFUSEPASS
#include "TIMVX/TIMVXPasses.h.inc"

namespace {
using namespace ::mlir::timvx::detail;

// Tolerant float compare for the (scale, zp) cross-check — the chain
// constants are TFLite-published values that may differ from the type
// stamp by a few ULPs after canonicalize re-materialisation.
inline bool approxEq(double a, double b,
                     double rel = 1e-4, double abs_eps = 1e-6) {
  double diff = std::abs(a - b);
  if (diff <= abs_eps) return true;
  double mag = std::max(std::abs(a), std::abs(b));
  return diff <= rel * mag;
}

// Read (scale, zp) off a tensor's quant.uniform element type. Returns
// nullopt if the element type isn't UniformQuantizedType (per-axis is not
// handled — DataConvert on this chip is per-tensor only, see the project
// memory note about per-channel weights crashing conv2d).
inline std::optional<std::pair<double, int64_t>>
readQuantUniform(Type elem) {
  if (auto qt = dyn_cast<quant::UniformQuantizedType>(elem))
    return std::make_pair(qt.getScale(), qt.getZeroPoint());
  return std::nullopt;
}

// Match `mul(sub(cast(int_q → f32), zp_const), scale_const)` or the
// folded-zp-zero variant `mul(cast(int_q → f32), scale_const)`.
// Rewrite to `dataconvert(int_q → f32)`. Cross-checks the constants
// against the cast input's quant.uniform (S, Z).
struct DequantChainFuse : public OpRewritePattern<MultiplyOp> {
  using OpRewritePattern<MultiplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MultiplyOp mul,
                                 PatternRewriter &rewriter) const final {
    // Result must be f32 — this pass only handles the dequant direction.
    auto resTy = dyn_cast<RankedTensorType>(mul.getType());
    if (!resTy || !resTy.getElementType().isF32())
      return failure();

    // Scale constant on the multiply's RHS.
    auto scaleConst = matchConstScalarFloat(mul.getInput2());
    if (!scaleConst)
      return failure();

    // The middle `sub(_, zp_const)` is present iff Z != 0; canonicalize
    // folds `sub _, 0.0` to identity.
    CastOp cast;
    std::optional<double> zpConst;
    SubOp sub;
    if (auto s = mul.getInput1().getDefiningOp<SubOp>()) {
      auto z = matchConstScalarFloat(s.getInput2());
      if (!z) return failure();
      zpConst = *z;
      sub = s;
      cast = s.getInput1().getDefiningOp<CastOp>();
    } else {
      zpConst = 0.0;
      cast = mul.getInput1().getDefiningOp<CastOp>();
    }
    if (!cast) return failure();

    // The cast must go `int_q (quant.uniform) → f32`. We pull (S, Z) off
    // the input's element type, not the cast's discardable attrs —
    // discardable attrs are an instruction to a *downstream* op, while
    // the input type is the authoritative source of the data's quant
    // context (the same convention `dataconvert` reads at runtime).
    auto castInTy = dyn_cast<RankedTensorType>(cast.getInput().getType());
    auto castOutTy = dyn_cast<RankedTensorType>(cast.getType());
    if (!castInTy || !castOutTy) return failure();
    if (!castOutTy.getElementType().isF32()) return failure();

    auto sz = readQuantUniform(castInTy.getElementType());
    if (!sz) return failure();
    double S = sz->first;
    int64_t Z = sz->second;

    // Cross-check the chain constants against the type's (S, Z).
    // If they disagree, the chain is doing something OTHER than the
    // standard dequant (e.g. a TFLite-published `Si != S_type` case);
    // bail rather than silently produce wrong values.
    if (!approxEq(*scaleConst, S)) return failure();
    if (!approxEq(*zpConst, static_cast<double>(Z))) return failure();

    // Build the replacement: dataconvert with the cast's input. Output
    // type is the multiply's result type (the f32 tensor). DataConvert
    // doesn't need explicit output_scale/output_zp attrs because both
    // are read from the operand specs at runtime — the input carries the
    // quant.uniform (S, Z), the output is f32 with no quant context.
    rewriter.replaceOpWithNewOp<DataConvertOp>(
        mul, mul.getType(), cast.getInput(),
        /*output_scale=*/FloatAttr{}, /*output_zp=*/IntegerAttr{});

    // The sub and cast may now be dead; let canonicalize CSE them out
    // (don't eraseOp here — they could legitimately have other users in
    // adjacent dequant chains for the same source).
    return success();
  }
};

struct TIMVXDequantFusePass
    : public impl::TIMVXDequantFusePassBase<TIMVXDequantFusePass> {
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<DequantChainFuse>(&getContext());
    // Disable folding + constant CSE for the same reason
    // TimvxQuantResidualFusePass does: the greedy driver's default
    // ConstOp folder re-materialises `timvx.const` ops without the
    // `quant_scale`/`quant_zp` discardable attrs, breaking downstream
    // codegen. This pass only does a peephole rewrite.
    GreedyRewriteConfig cfg;
    cfg.enableFolding(false);
    cfg.enableConstantCSE(false);
    if (failed(applyPatternsGreedily(getOperation(),
                                      std::move(patterns), cfg)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createTIMVXDequantFusePass() {
  return std::make_unique<TIMVXDequantFusePass>();
}

} // namespace timvx
} // namespace mlir
