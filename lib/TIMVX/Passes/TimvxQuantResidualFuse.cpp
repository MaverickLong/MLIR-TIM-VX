//===- TimvxQuantResidualFuse.cpp - timvx-quant-residual-fuse -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Collapse the residual-add fp32 detour (`cast(i8->f32) -> sub(zp) ->
// mul(scale)` ×2 -> `add` -> optional `clip` -> `mul(1/Sout) -> add(Zout)
// -> cast(f32->i8)`) into a single quantized `timvx.add` carrying
// (output_scale, output_zp). On VIP9000Nano-DI the NN-core's quant `Add`
// (probe-matrix u8 <-> u8 <-> u8 PASS) is dramatically cheaper than the
// CL/EVIS path the f32 chain compiles to.
//
// COUPLING WITH tosa-quant-anchor — IMPORTANT
// -------------------------------------------
// QRF is correctness-safe only when every operand's `quant.uniform`
// type stamp agrees with the dequant chain's mul const (the per-operand
// scale const) and sub const (the per-operand zp const). The runtime
// `timvx.add` reads its operand quant from the operand tensor's spec
// (= the type stamp); the chain const is what the un-fused dequant
// would multiply through. If they disagree, the fused add dequantizes
// each operand with the type-stamped S while QRF's chain-replacement
// math assumed the chain-const S — output is wrong by exactly the
// ratio (chain_S / type_S).
//
// `tosa-quant-anchor`'s `Sw=1/128` symmetric-int8 convention path
// stamps a CONVENTION scale on every conv2d output whose TFLite-real
// scale isn't recoverable from the local IR (no in-IR dequant chain to
// pin Si, no arg-attr to anchor against). The TFLite-real scale lives
// in the per-operand mul const of the downstream residual chain, NOT
// on the type. That's a stable mismatch on any TFLite resnet-style
// model that mixes symmetric (`Z=-128`) and asymmetric (`Z!=±128`)
// conv outputs.
//
// In the un-fused path this divergence is invisible because TIM-VX's
// `cast` op IGNORES the type's (S, Z) — the chain math operates on
// raw byte values with chain consts directly. Fusing the chain hands
// the math back to the type-stamped (S, Z), exposing the divergence.
//
// `matchDequantChain` cross-checks chain-const-S vs type-stamp-S for
// every operand and SOFT-fails (leaves the chain unfused) on
// disagreement. That's a LOCAL guard — it prevents emitting a wrong
// fused add for THIS chain. It does NOT catch the global cascade: if
// an upstream conv was anchored to convention-S and a different
// downstream chain WAS fuseable (its scales locally agreed), the
// downstream-fused add still sees the upstream operand decoded
// through the convention-S, not the TFLite-real S the network was
// trained for.
//
// Until `tosa-quant-anchor` recovers TFLite-real S for every tensor
// (no `Sw=1/128` fallback), QRF on a model with mixed
// symmetric+asymmetric quant cannot be guaranteed end-to-end correct
// even with the local cross-check. For now: the local cross-check
// minimises damage on partial-mismatch models; for full correctness
// on those models the user should pass `--no-fuse`. simple_v1 and
// `add_residual_i8` op_tests are designed to keep type-S == chain-S
// throughout, so QRF is byte-identical to no-fuse on those.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace timvx {

#define GEN_PASS_DEF_TIMVXQUANTRESIDUALFUSEPASS
#include "TIMVX/TIMVXPasses.h.inc"

namespace {
using namespace ::mlir::timvx::detail;

// Tolerant match for f32 values that came through the ASYMMETRIC quant
// chain. The constants in the IR are the result of the TOSA rescale
// lowering's float arithmetic, so they may differ from the "ideal"
// reconstructed value by a few ULPs.
bool approxEq(double a, double b, double rel = 1e-4, double abs_eps = 1e-6) {
  double diff = std::abs(a - b);
  if (diff <= abs_eps) return true;
  double mag = std::max(std::abs(a), std::abs(b));
  return diff <= rel * mag;
}

// Match `mul(X, scale_const) <- sub(Y, zp_const) <- cast(i8->f32, I_i8)`
// — the per-operand dequant chain. Verifies the consts match the
// producer's (Sa, Za) attrs and returns the i8 producer.
//
// The middle `sub(_, zp_const)` is omitted when the producer's zero-point
// is 0 — `--canonicalize` folds `sub %f, 0.0` to identity.
//
// Strict mode: the early shape failures (no upstream mul / no constant
// scale / no upstream cast / wrong element types) are silent — they
// just mean "this isn't a dequant chain, the caller should try
// something else." But once the shape matches and the IR has clearly
// committed to representing a dequant chain, any (S, Z) cross-check
// failure against the producer's declared attrs is a user-fixable
// inconsistency: emit a diagnostic and set the strict-failed flag so
// the pass exits with `signalPassFailure` rather than silently
// dropping the fuse.
Value matchDequantChain(Operation *contextOp, Value real,
                        double &outScale, int64_t &outZp,
                        bool *strictFailed) {
  auto mul = real.getDefiningOp<MultiplyOp>();
  if (!mul) return {};
  auto scaleConstF = matchConstScalarFloat(mul.getInput2());
  if (!scaleConstF) return {};

  CastOp cast;
  std::optional<double> zpConstF;
  if (auto sub = mul.getInput1().getDefiningOp<SubOp>()) {
    zpConstF = matchConstScalarFloat(sub.getInput2());
    if (!zpConstF) return {};
    cast = sub.getInput1().getDefiningOp<CastOp>();
  } else {
    zpConstF = 0.0;
    cast = mul.getInput1().getDefiningOp<CastOp>();
  }
  if (!cast) return {};

  auto srcTy = dyn_cast<RankedTensorType>(cast.getInput().getType());
  auto dstTy = dyn_cast<RankedTensorType>(cast.getType());
  if (!srcTy || !dstTy) return {};
  // Storage element type may be a plain IntegerType OR a
  // quant.uniform<int:f32, ...> (after `tosa-quant-anchor` ran).
  Type srcElem = srcTy.getElementType();
  if (auto qty = dyn_cast<quant::QuantizedType>(srcElem))
    srcElem = qty.getStorageType();
  if (!isa<IntegerType>(srcElem)) return {};
  if (!dstTy.getElementType().isF32()) return {};

  // Cross-check the chain const against the cast input's RUNTIME quant
  // — i.e. the (S, Z) the TIM-VX runtime will use to dequantize the
  // operand inside the fused `timvx.add` we're about to emit.
  //
  // The runtime quant comes from one of two places, in priority:
  //   1. For a BlockArg, the `timvx.output_scale`/`timvx.output_zp`
  //      arg-attr (`getProducerQuant`).
  //   2. For an op-defined value, the `quant.uniform<...>` element type
  //      stamped by `tosa-quant-anchor` and propagated through
  //      tosa-to-timvx (read here off `cast.getInput().getType()`).
  //
  // BOTH must match the chain const, because once we fuse the chain
  // away the runtime add reads its operand quant from the operand
  // tensor's spec (= the type stamp). If `tosa-quant-anchor` derived
  // S from its Sw=1/128 convention path while TFLite published a
  // different S in the chain (a 3.4× drift on the symmetric-Z=-128
  // resnet18 conv outputs), trusting the chain const after fusing
  // gives a runtime-time math mismatch — the fused add dequantizes
  // operand B with the typed S, but our `outScale` here used the
  // chain S. Soft-fail and leave the chain unfused so the explicit
  // mul-by-chain-const continues to do the right math at runtime.
  auto checkQuant = [&](double prodS, int64_t prodZ,
                         const char *prodSrc) -> bool {
    if (!approxEq(prodS, *scaleConstF)) {
      // Soft-fail: the chain isn't a faithful dequant of this operand.
      // Don't strict-fail; the model is internally consistent, just
      // not fuseable.
      return false;
    }
    if (!approxEq(static_cast<double>(prodZ),
                  static_cast<double>(*zpConstF))) {
      return false;
    }
    (void)prodSrc;
    return true;
  };

  if (isa<BlockArgument>(cast.getInput())) {
    if (auto prod = getProducerQuant(cast.getInput())) {
      if (!checkQuant(prod->first, prod->second, "BlockArg arg-attr"))
        return {};
    }
  } else {
    // Op-defined producer. Read (S, Z) off the cast's input type's
    // quant.uniform stamp (storage=int, expressed=f32).
    auto castInTy = dyn_cast<RankedTensorType>(cast.getInput().getType());
    if (castInTy) {
      if (auto qt = dyn_cast<quant::UniformQuantizedType>(
              castInTy.getElementType())) {
        if (!checkQuant(qt.getScale(), qt.getZeroPoint(), "type stamp"))
          return {};
      }
    }
  }

  // Use the chain's reconstructed values as the source of truth — at
  // this point we've cross-checked them against the operand's runtime
  // quant (BlockArg arg-attr or op-defined type stamp) so the fused
  // add's runtime dequant of this operand will agree.
  outScale = *scaleConstF;
  outZp = static_cast<int64_t>(std::lround(*zpConstF));
  return cast.getInput();
}

// Resolve the i8 representation of an f32 real-valued operand of the
// residual add. Three cases, in priority order:
//   (A) `cast(i8->f32) <- sub <- mul` dequant chain.
//   (B) the operand is `cast(int -> f32)` produced by a previously-fused
//       block's chain-replacement step. Walk through to the int input
//       directly.
//   (C) the operand is an annotated f32 op (clip/add) and the chain-
//       replacement hasn't run yet. Synthesise a `cast(f32->i8)` bridge;
//       case (B) on the next iteration collapses it.
Value getI8Operand(Operation *contextOp, Value real,
                    double &outScale, int64_t &outZp,
                    PatternRewriter &rewriter, bool *strictFailed) {
  if (Value i8 = matchDequantChain(contextOp, real, outScale, outZp,
                                    strictFailed))
    return i8;
  // matchDequantChain may have set *strictFailed if it found a partial
  // shape match with a cross-check failure. In that case the IR is
  // already known-inconsistent; don't try cases B/C as a workaround.
  if (*strictFailed) return {};

  auto *def = real.getDefiningOp();
  if (!def) return {};

  // Case (B).
  if (auto castOp = dyn_cast<CastOp>(def)) {
    auto castInTy = dyn_cast<RankedTensorType>(castOp.getInput().getType());
    if (castInTy && isa<IntegerType>(castInTy.getElementType())) {
      auto sAttr = castOp.getOutputScaleAttr();
      auto zAttr = castOp.getOutputZpAttr();
      if (sAttr && zAttr) {
        outScale = sAttr.getValueAsDouble();
        outZp = zAttr.getInt();
        return castOp.getInput();
      }
    }
  }

  // Case (C).
  auto sAttr = def->getAttrOfType<FloatAttr>("output_scale");
  auto zAttr = def->getAttrOfType<IntegerAttr>("output_zp");
  if (!sAttr || !zAttr) return {};
  outScale = sAttr.getValueAsDouble();
  outZp = zAttr.getInt();

  auto realTy = dyn_cast<RankedTensorType>(real.getType());
  if (!realTy) return {};
  auto i8Ty = RankedTensorType::get(realTy.getShape(),
                                     rewriter.getIntegerType(8));
  return CastOp::create(rewriter, def->getLoc(), i8Ty, real, sAttr, zAttr)
      .getResult();
}

struct QuantResidualFuse : public OpRewritePattern<CastOp> {
  bool *strictFailed;
  // Higher benefit than `StripUnfusedTailQuant` (which also matches the
  // residual-tail CastOp). The greedy driver tries higher-benefit
  // patterns first; we want the fuse to run before the strip
  // unconditionally erases the discardable attrs the fuse depends on.
  QuantResidualFuse(MLIRContext *ctx, bool *failed)
      : OpRewritePattern<CastOp>(ctx, /*benefit=*/10),
        strictFailed(failed) {}

  // Strict-mode helper: emit a diagnostic, set the pass-level flag,
  // and return failure so the rewriter moves on. The pass driver
  // checks the flag after greedy application and signalPassFailure.
  LogicalResult strictFail(CastOp finalCast, const Twine &msg) const {
    finalCast.emitError() << "QuantResidualFuse: " << msg;
    *strictFailed = true;
    return failure();
  }

  LogicalResult matchAndRewrite(CastOp finalCast,
                                 PatternRewriter &rewriter) const final {
    auto srcTy = dyn_cast<RankedTensorType>(finalCast.getInput().getType());
    auto dstTy = dyn_cast<RankedTensorType>(finalCast.getType());
    if (!srcTy || !dstTy) return failure();
    if (!srcTy.getElementType().isF32()) return failure();
    // Storage type may be unwrapped from a quant.uniform<i8:f32, S:Z>.
    Type dstElem = dstTy.getElementType();
    if (auto qty = dyn_cast<quant::QuantizedType>(dstElem))
      dstElem = qty.getStorageType();
    auto dstInt = dyn_cast<IntegerType>(dstElem);
    if (!dstInt) return failure();

    auto soAttr = finalCast.getOutputScaleAttr();
    auto zoAttr = finalCast.getOutputZpAttr();
    if (!soAttr || !zoAttr) return failure();
    double Sout = soAttr.getValueAsDouble();
    int64_t Zout = zoAttr.getInt();

    // Past this point the cast carries `output_scale`/`output_zp`,
    // which means RequantI32SkipFold tagged it as the tail of a
    // quantized requant chain. Subsequent shape mismatches are
    // therefore inconsistencies (the chain shape RequantI32SkipFold
    // produced is fixed), not "try a different pattern" cases — with
    // one exception: `residualAdd` absent is a legitimate non-residual
    // single-chain pattern (handled below as a soft `failure()`).

    if (Sout == 0.0)
      return strictFail(finalCast,
                        "tail cast carries output_scale=0; degenerate "
                        "or unset quant context");
    MultiplyOp invMul;
    if (auto tailAdd = finalCast.getInput().getDefiningOp<AddOp>()) {
      auto tailAddRhs = matchConstScalarFloat(tailAdd.getInput2());
      if (!tailAddRhs)
        return strictFail(finalCast,
                          "tail-add RHS is not a scalar float constant; "
                          "expected the requant zp constant");
      if (!approxEq(static_cast<double>(*tailAddRhs),
                    static_cast<double>(Zout))) {
        finalCast.emitError() << "QuantResidualFuse: tail-add zp constant ("
                              << *tailAddRhs
                              << ") disagrees with cast's output_zp (" << Zout
                              << ")";
        *strictFailed = true;
        return failure();
      }
      invMul = tailAdd.getInput1().getDefiningOp<MultiplyOp>();
    } else {
      // Either the canonical Zout=0 folded-zp shape (cast input is the
      // inv-scale multiply directly), or this isn't a requant tail at
      // all — `getI8Operand` case (C) synthesises bridge casts that
      // carry output_scale/output_zp attrs without preceding multiply
      // ops, and the greedy driver re-matches those bridges as fuse
      // candidates. Soft-fail on Zout!=0 here so those synthesised
      // bridges don't trip a spurious strict error.
      if (Zout != 0) return failure();
      invMul = finalCast.getInput().getDefiningOp<MultiplyOp>();
    }
    if (!invMul)
      return strictFail(finalCast,
                        "no inv-scale multiply feeds the requant tail; "
                        "RequantI32SkipFold guarantees one — the chain "
                        "has been mutated");
    auto invScaleConst = matchConstScalarFloat(invMul.getInput2());
    if (!invScaleConst)
      return strictFail(finalCast,
                        "inv-scale multiply RHS is not a scalar float "
                        "constant");
    if (!approxEq(static_cast<double>(*invScaleConst), 1.0 / Sout)) {
      // Soft fail: a chain whose immediate-upstream mul const doesn't
      // invert Sout is something other than a single-mul requant tail
      // — e.g. a TFLite FC `mul(invScale) → mul(extra_factor) → cast`
      // sequence where the inv-scale is split across multiple muls,
      // or the cast's stamped output_scale was set by back-propagation
      // through a downstream conv-rescale rather than by this chain
      // itself. Either way, this isn't a residual fuse target — just
      // try a different pattern, no diagnostic.
      return failure();
    }

    // Optional `clip(0, +inf)` (post-add ReLU). The clip commonly has
    // multiple users in resnet — its output feeds the next block's main
    // path AND the next residual chain — that's fine since we don't
    // modify the clip itself.
    Value preTail = invMul.getInput1();
    ClipOp postReluClip = preTail.getDefiningOp<ClipOp>();
    if (postReluClip)
      preTail = postReluClip.getInput();

    auto residualAdd = preTail.getDefiningOp<AddOp>();
    // Soft-fail: this is the legitimate "single quant chain, not a
    // residual" pattern — `StripUnfusedTailQuant` handles that case.
    if (!residualAdd) return failure();

    // Past this point: the chain shape definitely matches a residual
    // fuse candidate. Any operand or type mismatch is inconsistency.
    double Sa = 0, Sb = 0;
    int64_t Za = 0, Zb = 0;
    Value Ai8 = getI8Operand(finalCast, residualAdd.getInput1(),
                              Sa, Za, rewriter, strictFailed);
    if (!Ai8) {
      // Soft fail: an upstream `add` whose operand isn't a quantized
      // i8 source (e.g. a non-residual add fed by a constant or fp32
      // value) just means this isn't a residual-fuse candidate. Strict-
      // mode flags from `getI8Operand` (set on a partial-shape
      // dequant-chain mismatch) are propagated separately via
      // `strictFailed`.
      return failure();
    }
    Value Bi8 = getI8Operand(finalCast, residualAdd.getInput2(),
                              Sb, Zb, rewriter, strictFailed);
    if (!Bi8) {
      // Same rationale as Ai8 above.
      return failure();
    }

    auto aTy = dyn_cast<RankedTensorType>(Ai8.getType());
    auto bTy = dyn_cast<RankedTensorType>(Bi8.getType());
    if (!aTy || !bTy)
      return strictFail(finalCast,
                        "residual-add operand has non-ranked tensor type");
    // Compare STORAGE types only — `!quant.uniform<i8:f32, S:Z>` types
    // with different (S, Z) all share i8 storage, so the runtime add can
    // honor them via the per-tensor scale_factor math regardless of the
    // exact (S, Z) labels. Comparing element types directly would reject
    // the common case where each operand and the result carry distinct
    // (S, Z) (every TFLite residual block).
    auto storageOf = [](Type t) -> Type {
      if (auto qty = dyn_cast<quant::QuantizedType>(t))
        return qty.getStorageType();
      return t;
    };
    Type aStorage = storageOf(aTy.getElementType());
    Type bStorage = storageOf(bTy.getElementType());
    Type outStorage = storageOf(dstTy.getElementType());
    if (aStorage != outStorage)
      return strictFail(finalCast,
                        "residual-add operand 0 storage type does not "
                        "match the requant tail's output storage type");
    if (bStorage != outStorage)
      return strictFail(finalCast,
                        "residual-add operand 1 storage type does not "
                        "match the requant tail's output storage type");

    Location loc = finalCast.getLoc();
    auto newAdd = AddOp::create(rewriter, loc, dstTy, Ai8, Bi8);
    newAdd->setAttr("output_scale", soAttr);
    newAdd->setAttr("output_zp", zoAttr);
    Value out = newAdd.getResult();

    if (postReluClip) {
      double maxStorage =
          dstInt.isUnsigned() ? 255.0 : 127.0;
      double minReal = 0.0;
      double maxReal = (maxStorage - static_cast<double>(Zout)) * Sout;
      auto clipped = ClipOp::create(rewriter, loc, dstTy, out,
                                      rewriter.getF32FloatAttr(
                                          static_cast<float>(minReal)),
                                      rewriter.getF32FloatAttr(
                                          static_cast<float>(maxReal)));
      clipped->setAttr("output_scale", soAttr);
      clipped->setAttr("output_zp", zoAttr);
      out = clipped.getResult();
    }

    // Bridge casts (i8 quant → f32) for any remaining f32 consumers of
    // the residualAdd / postReluClip values. The output is plain fp32
    // — DO NOT attach `output_scale`/`output_zp` attrs: those are the
    // output tensor's quant context, and a fp32 tensor has none. The
    // runtime's CAST kernel rejects fp32 outputs with ASYM
    // Quantization() ("Inputs/Outputs data type not support: ASYM
    // UINT8, ASYM FLOAT32" on VIP9000Nano-DI). The numeric conversion
    // (storage byte → real value) is performed by the kernel using
    // the INPUT tensor's quant context, which the runtime reads off
    // the input spec independently.
    auto residualF32Ty =
        cast<RankedTensorType>(residualAdd.getType());
    auto deqResAdd = CastOp::create(
        rewriter, loc, residualF32Ty, newAdd, FloatAttr{}, IntegerAttr{});
    rewriter.replaceOp(residualAdd, deqResAdd.getResult());
    if (postReluClip) {
      auto clipF32Ty = cast<RankedTensorType>(postReluClip.getType());
      Value i8AfterClip = out;  // out is i8 fused clip when present
      auto deqClip = CastOp::create(
          rewriter, loc, clipF32Ty, i8AfterClip, FloatAttr{}, IntegerAttr{});
      rewriter.replaceOp(postReluClip, deqClip.getResult());
    }

    rewriter.replaceOp(finalCast, out);
    return success();
  }
};

// Post-fuse cleanup: any f32->narrow_int `timvx.cast` whose sole consumer
// is `func.return` and that still carries `output_scale` / `output_zp`
// attrs is the tail of a requant chain that QuantResidualFuse did *not*
// fuse (e.g. an atomic test where canonicalize folded `sub(zp=0)` so the
// dequant chain shrank below the matcher's window). Leaving the attrs on
// means EmitC will promote the cast's output and TIM-VX will divide by
// `output_scale` again on the f32->u8 path — double-scaling, saturating
// everything to u8 max. Strip the attrs so the cast becomes a plain
// numerical convert that just rounds + saturates to int.
// Strip the discardable `output_scale`/`output_zp` attrs from any
// `cast(f32 → narrow-int)` carrying RequantI32SkipFold's tail-tag that
// `QuantResidualFuse` didn't subsume — i.e. the final-return cast of a
// non-residual quant chain, or an intermediate cast whose upstream
// shape isn't fuseable. The (S, Z) is already on the result type; the
// discardable attrs are only needed by QRF's pattern match. By the time
// we get here, `--timvx-promote-i8-to-u8` has already inserted the ±128
// real-value bridge around every f32↔u8 cast, so this pattern is now a
// pure attr-strip.
struct StripUnfusedTailQuant : public OpRewritePattern<CastOp> {
  using OpRewritePattern<CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CastOp op,
                                 PatternRewriter &rewriter) const final {
    auto srcTy = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto dstTy = dyn_cast<RankedTensorType>(op.getType());
    if (!srcTy || !dstTy) return failure();
    if (!srcTy.getElementType().isF32()) return failure();
    Type dstElem = dstTy.getElementType();
    if (auto qty = dyn_cast<quant::QuantizedType>(dstElem))
      dstElem = qty.getStorageType();
    auto dstInt = dyn_cast<IntegerType>(dstElem);
    if (!dstInt || dstInt.getWidth() >= 32) return failure();
    if (!op.getOutputScaleAttr() && !op.getOutputZpAttr())
      return failure();

    rewriter.replaceOpWithNewOp<CastOp>(op, op.getType(), op.getInput(),
                                         FloatAttr{}, IntegerAttr{});
    return success();
  }
};

struct TIMVXQuantResidualFusePass
    : public impl::TIMVXQuantResidualFusePassBase<TIMVXQuantResidualFusePass> {
  void runOnOperation() final {
    bool strictFailed = false;
    RewritePatternSet patterns(&getContext());
    patterns.add<QuantResidualFuse>(&getContext(), &strictFailed);
    patterns.add<StripUnfusedTailQuant>(&getContext());
    // Same rationale as TIMVXConv1x1ToFCPass: the greedy driver's default
    // folding + CSE strips quant attrs from re-materialized timvx.const
    // ops. Disable both — this pass only needs the peephole rewrite, no
    // constant folding.
    GreedyRewriteConfig cfg;
    cfg.enableFolding(false);
    cfg.enableConstantCSE(false);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), cfg)))
      signalPassFailure();
    // Strict mode: a quantized requant tail that didn't fuse because
    // the chain is internally inconsistent is a user-fixable bug, not
    // a "fall back to the unfused fp32 path" condition. Emit-error +
    // signalPassFailure so the lowering halts and the user sees
    // exactly which mismatch needs fixing.
    if (strictFailed) signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createTIMVXQuantResidualFusePass() {
  return std::make_unique<TIMVXQuantResidualFusePass>();
}

} // namespace timvx
} // namespace mlir
