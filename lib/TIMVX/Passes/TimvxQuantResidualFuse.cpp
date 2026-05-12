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
// CL/EVIS path the f32 chain compiles to. On resnet50 (16 residual
// blocks), correct cascade-fusing brings end-to-end inference from
// ~68 ms (un-fused) down to ~7.5 ms via the OVXLIB path — matching the
// vendor's ACUITY-compiled NBG running on VIPLite, with the same
// quantized-only graph structure as `samples/multi_device/vx_resnet50.cc`.
//
// Two-phase pass architecture — load-bearing for the cascade
// ----------------------------------------------------------
// `runOnOperation` applies its patterns in two SEPARATE greedy fixpoints:
//
//   Phase 1: `QuantResidualFuse` only, run to fixpoint.
//   Phase 2: `StripUnfusedTailQuant` only, run to fixpoint.
//
// They CANNOT share a pattern set. The reason is a cascade dependency:
//
//   * Transition residuals (one per resnet stage — the block where both
//     operands come from `conv->rescale->cast->sub->mul` chains) match
//     QRF on the FIRST greedy iteration. Each one fuses, leaving a
//     bridge `cast(quant.uniform<i8>, S:Z) -> f32` in the IR whose
//     input is the fused `timvx.add` result.
//
//   * Identity residuals (the per-stage blocks where the skip operand
//     is the previous block's `clip(0, +inf)` output rather than a fresh
//     conv-rescale chain) DON'T match QRF on iteration 1 — their skip
//     operand is an f32 `clip` op, which `getI8Operand` rejects (cases
//     A/B/C all fail). They only become matchable on iteration 2+,
//     once the upstream transition fuse has replaced the f32 clip with
//     the bridge cast and the skip operand is now a CastOp whose input
//     carries a quant-typed stamp.
//
//   * `StripUnfusedTailQuant`'s job is to remove the `output_scale` /
//     `output_zp` discardable attrs from any `cast(f32 -> u8)` QRF
//     couldn't fuse — the attrs are only meaningful to QRF's matcher
//     and would cause double-scaling at the f32->u8 emit boundary if
//     left in place. But its match condition is satisfied IMMEDIATELY
//     for every requant-tail cast in the IR, including the identity
//     residuals' casts whose dependents (the upstream transitions)
//     haven't fused yet.
//
//   * If both patterns share a pattern set, the greedy driver applies
//     them in benefit order. On iteration 1, every cast whose block
//     QRF can't yet match gets visited by StripUnfusedTailQuant and
//     loses its quant attrs. On iteration 2, when the bridge cast
//     finally exists for the identity blocks, QRF's matcher rejects
//     them at the entry check `if (!soAttr || !zoAttr) return failure()`
//     — the attrs are gone. Net result: only the 4 per-stage
//     transition residuals fuse on resnet50, and the 12 identity
//     residuals stay f32-chained.
//
// Splitting the phases means StripUnfusedTailQuant runs ONLY after QRF
// has reached fixpoint across all greedy iterations. Casts that QRF
// genuinely cannot fuse still get stripped; casts QRF would have
// fused on a later iteration get to keep their attrs long enough to
// be matched.
//
// Cross-checks and runtime stamp dependency
// -----------------------------------------
// QRF's runtime semantic depends on every operand's `quant.uniform`
// element-type stamp agreeing with the dequant chain's `mul`/`sub`
// const operand values. The runtime `timvx.add` reads operand quant
// from the tensor spec built from the type stamp; the chain const is
// what the un-fused dequant would multiply through. `matchDequantChain`
// cross-checks them and SOFT-fails on disagreement, so a tensor whose
// upstream anchor produced (S, Z) that don't match the chain const
// (e.g. on a model that legitimately mixes scales) stays un-fused
// rather than producing a wrong fused add. This cross-check matters
// for atomic-test correctness but on TFLite-imported resnets is
// rarely tripped — IREE's TFLite frontend stamps chain consts and
// rescale-derived (S, Z) consistently with each other.
//
// `getI8Operand` case B accepts both `IntegerType` storage and
// `quant::QuantizedType` wrapping integer storage as the cast
// operand element type, and reads (S, Z) from the input's
// `quant.uniform` element-type stamp when the bridge cast's own
// discardable attrs are empty. This is what makes the second-iteration
// match work for identity residuals downstream of a fused transition.
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
  // different S in the chain, trusting the chain const after fusing
  // gives a runtime-time math mismatch — empirically this shifts the
  // argmax (resnet50 cat105.jpg: 285→852 with cross-check disabled,
  // i.e. wrong class). Soft-fail and leave the chain unfused so the
  // explicit mul-by-chain-const continues to do the right math at
  // runtime. The architectural fix is to make `tosa-quant-anchor`
  // recover the TFLite-real `So` for these tensors (see the
  // back-propagation logic in TosaQuantAnchor.cpp).
  auto checkQuant = [&](double prodS, int64_t prodZ,
                         const char *prodSrc) -> bool {
    if (!approxEq(prodS, *scaleConstF)) return false;
    if (!approxEq(static_cast<double>(prodZ),
                  static_cast<double>(*zpConstF)))
      return false;
    (void)prodSrc;
    return true;
  };

  if (isa<BlockArgument>(cast.getInput())) {
    if (auto prod = getProducerQuant(cast.getInput())) {
      if (!checkQuant(prod->first, prod->second, "BlockArg arg-attr"))
        return {};
    }
  } else {
    auto castInTy = dyn_cast<RankedTensorType>(cast.getInput().getType());
    if (castInTy) {
      if (auto qt = dyn_cast<quant::UniformQuantizedType>(
              castInTy.getElementType())) {
        if (!checkQuant(qt.getScale(), qt.getZeroPoint(), "type stamp"))
          return {};
      }
    }
  }

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

  // Case (B). Accept both bare `IntegerType` storage (un-anchored i8
  // tensor passed by `RequantI32SkipFold`) AND `quant::QuantizedType`
  // wrapping an integer storage (the anchored / fused-block-output case
  // — every `timvx.add` produced by QRF on a previous greedy iteration
  // has `quant.uniform<i8:f32, S:Z>` on its result, and the bridge cast
  // QRF emits from that result inherits the quant type as its input).
  //
  // (S, Z) sources tried in priority:
  //   1. the cast's own `output_scale`/`output_zp` attrs (legacy from
  //      `RequantI32SkipFold` / `Cast` lowering paths)
  //   2. the `quant.uniform<...>` element type on the cast INPUT — this
  //      is the QRF-fused-block-output path. Reading the stamp is
  //      required because the bridge cast QRF emits deliberately leaves
  //      `output_scale`/`output_zp` empty (the runtime cast kernel
  //      rejects fp32 outputs with an attached ASYM Quantization()).
  //      Without this lookup, every identity-residual block downstream
  //      of an already-fused block fails `getI8Operand` and the QRF
  //      cascade stops at the per-stage transition residual.
  if (auto castOp = dyn_cast<CastOp>(def)) {
    auto castInTy = dyn_cast<RankedTensorType>(castOp.getInput().getType());
    if (castInTy) {
      Type elem = castInTy.getElementType();
      bool intStorage = isa<IntegerType>(elem);
      auto qty = dyn_cast<quant::QuantizedType>(elem);
      bool quantIntStorage =
          qty && isa<IntegerType>(qty.getStorageType());
      if (intStorage || quantIntStorage) {
        auto sAttr = castOp.getOutputScaleAttr();
        auto zAttr = castOp.getOutputZpAttr();
        if (sAttr && zAttr) {
          outScale = sAttr.getValueAsDouble();
          outZp = zAttr.getInt();
          return castOp.getInput();
        }
        // No explicit attrs — read (S, Z) off the input's quant.uniform
        // stamp. This is the canonical bridge-cast case after a prior
        // greedy iteration of QRF.
        if (auto uniform = dyn_cast<quant::UniformQuantizedType>(elem)) {
          outScale = uniform.getScale();
          outZp = uniform.getZeroPoint();
          return castOp.getInput();
        }
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

    // PHASE 1 — Fuse to fixpoint.
    //
    // Only the `QuantResidualFuse` pattern runs here. `StripUnfusedTailQuant`
    // is deliberately deferred to phase 2 because of a cascade interaction:
    //
    //   * QRF on an IDENTITY residual block needs the previous block's
    //     output to come from a CastOp whose input carries a `quant.uniform`
    //     stamp (the bridge-cast path in `getI8Operand` case B). That bridge
    //     cast only exists *after* the previous (transition) block has been
    //     fused — which can happen multiple greedy iterations into the run.
    //   * If `StripUnfusedTailQuant` shares the pattern set, it matches the
    //     identity block's `finalCast` on its FIRST visit (when QRF returns
    //     failure because the skip operand is still an f32 `clip`) and
    //     strips the `output_scale`/`output_zp` discardable attrs. Once
    //     stripped, the cast no longer matches QRF's entry check on any
    //     subsequent iteration, even after the transition block has fused
    //     and made the bridge cast available.
    //
    // Net effect of the bug: on resnet50, only the 4 transition residuals
    // fired, and the 12 identity residuals stayed f32-chained. Separating
    // the phases lets QRF iterate to fixpoint without the strip clobbering
    // unfused-but-still-fuseable casts.
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<QuantResidualFuse>(&getContext(), &strictFailed);
      GreedyRewriteConfig cfg;
      cfg.enableFolding(false);
      cfg.enableConstantCSE(false);
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                          cfg)))
        signalPassFailure();
    }

    if (strictFailed) {
      signalPassFailure();
      return;
    }

    // PHASE 2 — Strip the residual `output_scale`/`output_zp` discardable
    // attrs from any cast QRF couldn't fuse. The (S, Z) is already on the
    // result type stamp; the discardable attrs are only meaningful to
    // QRF's matcher and would confuse downstream codegen if left in place
    // (double-scaling at the f32→u8 emit boundary, saturating everything).
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<StripUnfusedTailQuant>(&getContext());
      GreedyRewriteConfig cfg;
      cfg.enableFolding(false);
      cfg.enableConstantCSE(false);
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                          cfg)))
        signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createTIMVXQuantResidualFusePass() {
  return std::make_unique<TIMVXQuantResidualFusePass>();
}

} // namespace timvx
} // namespace mlir
