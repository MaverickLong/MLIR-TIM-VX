//===- TosaQuantAnchor.cpp - tosa-quant-anchor ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Anchors absolute (scale, zp) on every quantized i8 SSA value in a func by
// rewriting the tensor element type to !quant.uniform<i8:f32, S:Z>. Single
// place where the Sw=1/128 convention is applied — every other downstream
// pass reads the type and trusts it.
//
// Algorithm overview:
//   1. Pre-walk: collect (scale, zp) for every quantized i8 value, in
//      priority order:
//        (a) BlockArg from `timvx.output_scale`/`timvx.output_zp` arg-attrs
//            (default to (1.0, 0) anchor when absent — preserves the
//            ratio-only equivalence of int8-only chains).
//        (b) Func result from `timvx.output_scale`/`timvx.output_zp`
//            res-attrs.
//        (c) tosa.rescale: walk forward to find the immediate (or
//            quant-preserving-prefixed) `cast(i8→f32) → sub(zp_const) →
//            mul(scale_const)` dequant chain; if present, So = mul const,
//            Zo = rescale's output_zp constant.
//        (d) tosa.rescale: fall back to So = Si * Sw / M with Sw = 1/128.
//            Emits a remark naming this tensor as convention-derived.
//        (e) Pool / pad / transpose / reshape / slice propagate from
//            operand 0 to result.
//        (f) tosa.cast f32→narrow-int with `timvx.output_scale`/
//            `timvx.output_zp` discardable attrs (RequantI32SkipFold tail).
//
//   2. Type rewrite: for every value with a recovered (S, Z), replace its
//      tensor element type with !quant.uniform<i8:f32, S:Z>. Update
//      func signature, return op operand types, and (where the value's
//      defining op was a tosa.const) the const's `values` attribute type
//      so the verifier still accepts it.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace timvx {

#define GEN_PASS_DEF_TOSAQUANTANCHORPASS
#include "TIMVX/TIMVXPasses.h.inc"

namespace {
using ::mlir::timvx::detail::matchConstScalarFloat;
using ::mlir::timvx::detail::matchConstScalarInt;

// Per-tensor (scale, zero-point) record carried through the anchor pass.
struct QZ {
  double scale;
  int64_t zp;
  bool fromConvention; // true if Sw=1/128 was used to derive scale.
  bool userExplicit;   // true when (S, Z) was declared by the user via
                       // a `timvx.output_scale`/`timvx.output_zp`
                       // arg-attr or res-attr. Cross-check on
                       // back-propagation arrival fires only against
                       // these — the chain-vs-back-prop derived values
                       // can legitimately differ when the model's
                       // actual Sw isn't the canonical 1/128.
};

// True if the given element type is the storage form we anchor on
// (signed 8-bit integer). i16/u8/etc. fall through unchanged — TIM-VX's
// existing per-tensor int8 path is what `RescaleConvFusion` /
// `QuantResidualFuse` actually consume.
static bool isAnchorableI8(Type elem) {
  auto it = dyn_cast<IntegerType>(elem);
  return it && it.getWidth() == 8 && !it.isUnsigned();
}

static bool isQuantTensor(Type t) {
  auto rt = dyn_cast<RankedTensorType>(t);
  return rt && isAnchorableI8(rt.getElementType());
}

// Skip zero-or-more quant-preserving ops (pad / pool / transpose /
// reshape / slice) on a single-use chain. Used by both the dequant-chain
// recovery (a tosa.rescale's eventual dequant cast) and forward
// propagation walks.
static Value walkSingleUseQuantPreserving(Value v) {
  while (v.hasOneUse()) {
    Operation *user = *v.getUsers().begin();
    if (isa<tosa::MaxPool2dOp, tosa::AvgPool2dOp, tosa::PadOp,
            tosa::TransposeOp, tosa::ReshapeOp, tosa::SliceOp>(user) &&
        user->getNumResults() == 1) {
      v = user->getResult(0);
      continue;
    }
    break;
  }
  return v;
}

// Recover (S_dequant, Z_dequant) from a `cast(i8→f32) → sub(zp_const)
// → mul(scale_const)` chain immediately downstream of `v` (skipping
// quant-preserving ops on a single-use path). Returns std::nullopt if no
// such chain exists. Robust against the canonical form where
// canonicalize folded `sub %f, 0.0` away (zp_const = 0 implicit).
static std::optional<std::pair<double, int64_t>>
recoverFromDequantChain(Value v) {
  Value cur = walkSingleUseQuantPreserving(v);
  if (!cur.hasOneUse()) return std::nullopt;
  auto cast = dyn_cast<tosa::CastOp>(*cur.getUsers().begin());
  if (!cast || !cast->hasOneUse()) return std::nullopt;
  auto castOutTy = dyn_cast<RankedTensorType>(cast.getType());
  if (!castOutTy || !castOutTy.getElementType().isF32()) return std::nullopt;

  Value next = cast.getResult();
  double zpD = 0.0;
  if (auto sub = dyn_cast<tosa::SubOp>(*next.getUsers().begin())) {
    if (!sub->hasOneUse()) return std::nullopt;
    if (sub.getInput1() != next) return std::nullopt;
    auto z = matchConstScalarFloat(sub.getInput2());
    if (!z) return std::nullopt;
    zpD = *z;
    next = sub.getResult();
  }
  auto mul = dyn_cast<tosa::MulOp>(*next.getUsers().begin());
  if (!mul) return std::nullopt;
  Value other;
  if (mul.getInput1() == next) other = mul.getInput2();
  else if (mul.getInput2() == next) other = mul.getInput1();
  else return std::nullopt;
  auto sc = matchConstScalarFloat(other);
  if (!sc) return std::nullopt;
  return std::make_pair(*sc, static_cast<int64_t>(std::lround(zpD)));
}

struct TosaQuantAnchorPass
    : public impl::TosaQuantAnchorPassBase<TosaQuantAnchorPass> {
  bool strictFailed = false;
  void runOnOperation() final {
    ModuleOp mod = getOperation();
    strictFailed = false;
    mod.walk([&](func::FuncOp f) { processFunc(f); });
    if (strictFailed) signalPassFailure();
  }

  void processFunc(func::FuncOp f) {
    MLIRContext *ctx = f.getContext();
    auto i8 = IntegerType::get(ctx, 8);
    auto f32 = Float32Type::get(ctx);

    // Step 1: build (S, Z) per Value.
    DenseMap<Value, QZ> qz;

    auto setIfMissing = [&](Value v, QZ rec) {
      if (qz.find(v) == qz.end()) qz[v] = rec;
    };

    // (a0) "Calibrated TOSA" path: tensors whose element type is already
    // a `!quant.uniform<iN:fXX, S:Z>` carry the authoritative TFLite
    // calibration. Harvest (S, Z) directly from those types and strip the
    // quant wrapper back to plain int storage so every downstream
    // pattern (PadRescaleSwap, QRF, RequantI32SkipFold, …) keeps its
    // existing `dyn_cast<IntegerType>` element-type checks.
    //
    // (Why strip vs. let everyone read types: every pattern currently
    // matches plain-int TOSA shapes — refactoring them all is a flag day.
    // This single import-and-strip step preserves the calibration without
    // forcing that. Long-term, patterns can migrate to reading types
    // directly and this step + `qz` become vestigial.)
    // Strip-and-restore: harvest (S, Z), then rewrite the tensor type to
    // the plain integer storage form. The existing `rewriteTypes` step at
    // the end of this pass will re-stamp `!quant.uniform<iN:fXX, S:Z>`
    // using these authoritative values. Between import and re-stamp,
    // every downstream pattern (PadRescaleSwap, RequantI32SkipFold, …
    // and tosa-to-timvx / QRF after this pass) sees plain int element
    // types and keeps its existing matchers working unchanged.
    //
    // Const handling: the TFLite-tagged TOSA emits `tosa.const` with the
    // VALUES attribute already in plain-int storage form (e.g.
    // `dense<bytes> : tensor<7x7x3x64xi8>`) while the RESULT type wraps
    // it in `quant.uniform`. setType on the result to the plain-int form
    // therefore restores symmetry without rebuilding the values attr.
    // importAndStrip:
    //   * For every value whose element type is quant.uniform, record the
    //     calibrated (S, Z) into qz with userExplicit=true.
    //   * For non-const values, strip the wrapper to plain int storage so
    //     downstream patterns (tosa-to-timvx / QRF / RequantI32SkipFold)
    //     keep their existing `dyn_cast<IntegerType>` element-type checks.
    //   * For tosa.const results, leave the type alone — keeping the
    //     wrapper means rewriteTypes' setType+bitcast path runs into
    //     `DenseElementsAttr::bitcast(QuantizedType)` which is UNREACHABLE
    //     in MLIR's attribute storage. RescaleConvFusion re-derives the
    //     weight scale `sw = M × So / Si` from the (calibrated) qz values
    //     and stamps a fresh timvx.const with quant_scale/quant_zp attrs,
    //     so const-wrapper preservation isn't load-bearing for emission.
    auto importAndStrip = [&](Value v, bool isConst) {
      auto rt = dyn_cast<RankedTensorType>(v.getType());
      if (!rt) return;
      auto qty = dyn_cast<quant::UniformQuantizedType>(rt.getElementType());
      if (!qty) return;
      qz[v] = {qty.getScale(), qty.getZeroPoint(),
                /*fromConvention=*/false, /*userExplicit=*/true};
      if (isConst) return;
      auto stripped = RankedTensorType::get(rt.getShape(),
                                              qty.getStorageType());
      v.setType(stripped);
    };
    for (BlockArgument a : f.getArguments())
      importAndStrip(a, /*isConst=*/false);
    f.walk([&](Operation *op) {
      bool isConst = isa<tosa::ConstOp>(op);
      for (Value r : op->getResults()) importAndStrip(r, isConst);
    });
    // Patch the func signature now that BlockArg / return types may have
    // changed under us.
    {
      SmallVector<Type> newIns(f.getArgumentTypes());
      for (auto en : llvm::enumerate(f.getArguments()))
        newIns[en.index()] = en.value().getType();
      SmallVector<Type> newOuts(f.getResultTypes());
      f.walk([&](func::ReturnOp ret) {
        for (auto en : llvm::enumerate(ret.getOperands()))
          newOuts[en.index()] = en.value().getType();
      });
      auto newFnTy = FunctionType::get(ctx, newIns, newOuts);
      if (newFnTy != f.getFunctionType())
        f.setType(newFnTy);
    }

    // (a) BlockArgs. Track whether (S, Z) came from an EXPLICIT
    // arg-attr declaration vs. the default (1.0, 0) convention anchor
    // — back-propagation later overrides the convention silently but
    // CROSS-CHECKS against explicit declarations.
    for (BlockArgument a : f.getArguments()) {
      if (!isQuantTensor(a.getType())) continue;
      double s = 1.0;
      int64_t z = 0;
      bool explicitArg = false;
      unsigned i = a.getArgNumber();
      if (auto sa = f.getArgAttrOfType<FloatAttr>(i, "timvx.output_scale")) {
        s = sa.getValueAsDouble();
        explicitArg = true;
      }
      if (auto za = f.getArgAttrOfType<IntegerAttr>(i, "timvx.output_zp")) {
        z = za.getInt();
        explicitArg = true;
      }
      // If (a0) already populated qz from a quant.uniform type, keep it.
      if (qz.find(a) != qz.end()) continue;
      qz[a] = {s, z, /*fromConvention=*/!explicitArg,
                /*userExplicit=*/explicitArg};
    }

    // (b) Func result attrs (lets atomic tests pin So at the function
    // boundary; the value here is the return-op's operand, recorded so
    // a subsequent rescale handler can prefer this over chain recovery).
    f.walk([&](func::ReturnOp ret) {
      for (auto en : llvm::enumerate(ret.getOperands())) {
        Value v = en.value();
        if (!isQuantTensor(v.getType())) continue;
        unsigned idx = en.index();
        auto sa = f.getResultAttrOfType<FloatAttr>(idx, "timvx.output_scale");
        auto za = f.getResultAttrOfType<IntegerAttr>(idx, "timvx.output_zp");
        if (sa && za)
          setIfMissing(v, {sa.getValueAsDouble(), za.getInt(),
                            /*fromConvention=*/false,
                            /*userExplicit=*/true});
      }
    });

    // (c) (d) (e) (f) — single forward walk in program order.
    f.walk([&](Operation *op) {
      if (auto resc = dyn_cast<tosa::RescaleOp>(op)) {
        handleRescale(resc, qz);
        return;
      }
      if (isa<tosa::MaxPool2dOp, tosa::AvgPool2dOp, tosa::PadOp,
              tosa::TransposeOp, tosa::ReshapeOp, tosa::SliceOp>(op)) {
        if (op->getNumOperands() < 1 || op->getNumResults() != 1) return;
        Value out = op->getResult(0);
        if (!isQuantTensor(out.getType())) return;
        if (auto it = qz.find(op->getOperand(0)); it != qz.end())
          setIfMissing(out, it->second);
        return;
      }
      if (auto c = dyn_cast<tosa::CastOp>(op)) {
        // RequantI32SkipFold tail: f32→narrow-int with discardable attrs.
        Value out = c.getResult();
        if (!isQuantTensor(out.getType())) return;
        auto sa = c->getAttrOfType<FloatAttr>("timvx.output_scale");
        auto za = c->getAttrOfType<IntegerAttr>("timvx.output_zp");
        if (sa && za)
          setIfMissing(out, {sa.getValueAsDouble(), za.getInt(),
                              /*fromConvention=*/false,
                              /*userExplicit=*/false});
      }
    });

    // Step 1.5: BACKWARD walk. The forward walk only anchors a
    // rescale's OUTPUT (So) when the chain immediately downstream
    // names it. Anything UPSTREAM of a chain anchor stays
    // convention-derived, including the BlockArg input — and the
    // harness then preprocesses the input image at the wrong scale,
    // collapsing all input information. Walk backward from each
    // chain-anchored tensor through the producing rescale (and
    // quant-preserving ops) to back-derive (Si, Zi) for every
    // ancestor, overriding their convention values with the
    // back-propagated TFLite-real values. Cross-checks any non-
    // convention QZ entries the back-walk arrives at — that's where
    // a user-declared arg-attr disagreeing with a downstream chain
    // const surfaces as a strict-mode bug.
    backPropagateAnchors(f, qz);

    // Step 2: stamp tensor types.
    rewriteTypes(f, qz, i8, f32);

    // Diagnostic summary: how many tensors were anchored vs. convention-
    // derived. A pure TFLite-derived model will have most rescales hit
    // the convention path (no in-IR Sw signal); a hand-crafted atomic
    // test should have zero (everything anchored from arg/res attrs).
    unsigned conv = 0, real = 0;
    for (auto &kv : qz) {
      if (kv.second.fromConvention) ++conv;
      else ++real;
    }
    if (conv > 0) {
      f->emitRemark()
          << "tosa-quant-anchor: anchored " << real
          << " tensor(s) from explicit declarations / dequant chains; "
          << conv
          << " tensor(s) used the Sw=1/128 convention (no in-IR signal). "
             "These are byte-output equivalent — the convention only "
             "affects dequant printout — but if you want them anchored, "
             "either insert a downstream dequant chain or annotate the "
             "func arg/result with `timvx.output_scale`/`timvx.output_zp`.";
    }
  }

  // Process a tosa.rescale: derive (So, Zo) for its result, store in qz.
  void handleRescale(tosa::RescaleOp resc,
                     DenseMap<Value, QZ> &qz) {
    auto outZpVal = matchConstScalarInt(resc.getOutputZp());
    auto mulVal = matchConstScalarInt(resc.getMultiplier());
    auto shiftVal = matchConstScalarInt(resc.getShift());
    if (!outZpVal || !mulVal || !shiftVal) return;

    Value result = resc.getResult();
    if (!isQuantTensor(result.getType())) return;

    // Path (b) preset (already in qz from func res-attrs): respect.
    if (auto it = qz.find(result); it != qz.end()) return;

    // Path (c): downstream dequant chain names So.
    if (auto dq = recoverFromDequantChain(result)) {
      qz[result] = {dq->first, dq->second, /*fromConvention=*/false,
                     /*userExplicit=*/false};
      return;
    }

    // Path (d): Sw = 1/128 convention. Need Si from input.
    auto siIt = qz.find(resc.getInput());
    double si = 1.0;
    if (siIt != qz.end()) si = siIt->second.scale;
    // Else: fall back to (1.0, 0) anchor implicit for upstream tensors
    // whose seed slot was never populated (shouldn't happen in well-
    // formed input — every i8 SSA value the rescale consumes either
    // started from a BlockArg or from an upstream rescale we already
    // walked). Tracking it here so the convention is still applied
    // consistently rather than emitting a cryptic mid-walk error.

    double M = static_cast<double>(*mulVal) *
               std::pow(2.0, -double(*shiftVal));

    // Conv-rescale fusion target: M = (Si * Sw) / So  ⇒  So = Si * Sw / M
    // Standalone rescale: M = Si / So               ⇒  So = Si / M
    double so;
    if (auto conv = resc.getInput().getDefiningOp<tosa::Conv2DOp>()) {
      (void)conv;
      double sw = 1.0 / 128.0;
      so = (M != 0.0) ? (si * sw / M) : si;
    } else {
      so = (M != 0.0) ? (si / M) : si;
    }
    qz[result] = {so, *outZpVal, /*fromConvention=*/true,
                   /*userExplicit=*/false};
  }

  // Helper: write `cand` into `qz[v]`, or cross-check if `qz[v]` is
  // already a non-convention anchor. Returns true when the value was
  // newly written or updated (so the caller knows to push `v` further
  // up the worklist); false if it agreed with an existing anchor (or
  // disagreed and triggered strict-fail).
  bool updateOrCheck(Value v, QZ cand, DenseMap<Value, QZ> &qz,
                      Operation *contextOp) {
    auto it = qz.find(v);
    if (it == qz.end()) {
      qz[v] = cand;
      return true;
    }
    auto approxEq = [](double a, double b, double rel = 1e-4,
                        double abs_eps = 1e-6) {
      double diff = std::abs(a - b);
      if (diff <= abs_eps) return true;
      return diff <= rel * std::max(std::abs(a), std::abs(b));
    };

    // Both sides user-explicit: real cross-check ground.
    if (it->second.userExplicit && cand.userExplicit) {
      if (!approxEq(it->second.scale, cand.scale) ||
          it->second.zp != cand.zp) {
        contextOp->emitError()
            << "tosa-quant-anchor: two user-declared quants disagree on "
               "the same tensor — ("
            << it->second.scale << ", " << it->second.zp << ") vs ("
            << cand.scale << ", " << cand.zp << ")";
        strictFailed = true;
      }
      return false;
    }

    // User-explicit candidate beats any non-user existing (chain const
    // or convention). The user value is TFLite ground truth; chain
    // consts in the IR may be fp32-precision-losing, and convention
    // values are derived. Propagating the user value through quant-
    // preserving ops keeps reshape/pool/etc. operand-vs-result types
    // consistent (TOSA's verifier requires exact element-type match).
    if (cand.userExplicit && !it->second.userExplicit) {
      qz[v] = cand;
      return true;
    }

    // User-explicit existing wins over back-prop / convention
    // candidate — the back-propped value is convention-influenced
    // (uses Sw=1/128) and will legitimately differ from a TFLite-real
    // user declaration whenever the model's actual Sw ≠ 1/128. Don't
    // emit an error; the convention divergence isn't an IR bug.
    if (it->second.userExplicit) return false;

    if (it->second.fromConvention) {
      // The forward walk anchored this with the Sw=1/128 fallback;
      // back-prop has a value derived through actual rescale ratios —
      // override silently. Affects intermediate-tensor metadata only
      // (bias_scale display, etc.), not byte-level computation.
      qz[v] = cand;
      return true;
    }
    // Both non-userExplicit, non-convention: forward-walk chain anchor
    // (TFLite-real) vs back-prop (convention-influenced). First-write-
    // wins, no error.
    return false;
  }

  // Walk backward from each chain-anchored tensor, deriving (Si, Zi)
  // for every ancestor through the producing op. Worklist-driven so
  // back-propagation chains naturally (anchor at residual block N
  // back-derives block N-1, which then back-derives block N-2, etc.).
  // Visited set guards against revisiting the same value — the rescale
  // chain in TFLite is a DAG, so a single value can be on multiple
  // back-paths; we want to compute its derived (Si, Zi) once.
  void backPropagateAnchors(func::FuncOp f, DenseMap<Value, QZ> &qz) {
    SmallVector<Value> worklist;
    for (auto &kv : qz)
      if (!kv.second.fromConvention) worklist.push_back(kv.first);
    DenseSet<Value> visited;

    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      if (!visited.insert(v).second) continue;
      // BlockArg has no producer to walk through.
      if (isa<BlockArgument>(v)) continue;
      Operation *def = v.getDefiningOp();
      if (!def) continue;

      QZ here = qz[v];

      // Quant-preserving ops: pass (S, Z) up to operand 0 unchanged.
      if (isa<tosa::MaxPool2dOp, tosa::AvgPool2dOp, tosa::PadOp,
              tosa::TransposeOp, tosa::ReshapeOp, tosa::SliceOp>(def)) {
        if (def->getNumOperands() < 1) continue;
        Value in = def->getOperand(0);
        if (!isQuantTensor(in.getType())) continue;
        if (updateOrCheck(in, here, qz, def))
          worklist.push_back(in);
        continue;
      }

      // tosa.cast quant.uniform<i8> → quant.uniform<i8> (e.g. produced
      // by RequantI32SkipFold). Treat as identity for back-prop: the
      // input's (S, Z) was the upstream rescale's So/Zo, which we
      // back-prop further from there. But the cast input is fp32 in
      // most cases — skip if not quant.
      if (auto castOp = dyn_cast<tosa::CastOp>(def)) {
        Value in = castOp.getInput();
        if (!isQuantTensor(in.getType())) continue;
        if (updateOrCheck(in, here, qz, def))
          worklist.push_back(in);
        continue;
      }

      // tosa.rescale: derive upstream Si from downstream So.
      if (auto resc = dyn_cast<tosa::RescaleOp>(def)) {
        auto mulVal = matchConstScalarInt(resc.getMultiplier());
        auto shiftVal = matchConstScalarInt(resc.getShift());
        if (!mulVal || !shiftVal) continue;
        double M = static_cast<double>(*mulVal) *
                   std::pow(2.0, -double(*shiftVal));

        // Conv-rescale fusion target: M = (Si_conv × Sw) / So_resc.
        // Solve for Si_conv = M × So_resc / Sw  with Sw = 1/128
        // convention. Zi_conv comes directly from the conv's
        // input_zp operand (TFLite serialised it explicitly).
        if (auto conv = resc.getInput().getDefiningOp<tosa::Conv2DOp>()) {
          Value convIn = conv.getInput();
          if (!isQuantTensor(convIn.getType())) continue;
          auto convInZp = matchConstScalarInt(conv.getInputZp());
          if (!convInZp) continue;
          double Sw = 1.0 / 128.0;
          double Si = M * here.scale / Sw;
          QZ cand{Si, *convInZp, /*fromConvention=*/false,
                  /*userExplicit=*/false};
          if (updateOrCheck(convIn, cand, qz, conv))
            worklist.push_back(convIn);
          continue;
        }

        // Standalone i8→i8 rescale: out_real = in_real, M = Si / So,
        // so Si = M × So. Zi from the rescale's input_zp operand.
        Value rescIn = resc.getInput();
        if (!isQuantTensor(rescIn.getType())) continue;
        auto rescInZp = matchConstScalarInt(resc.getInputZp());
        if (!rescInZp) continue;
        double Si = M * here.scale;
        QZ cand{Si, *rescInZp, /*fromConvention=*/false,
                /*userExplicit=*/false};
        if (updateOrCheck(rescIn, cand, qz, resc))
          worklist.push_back(rescIn);
        continue;
      }
      // Other producing ops: don't know how to back-derive — leave
      // upstream as-is. Most other quantized i8-producing ops are
      // already in the propagation list above.
    }
  }

  // Step 2: walk every recorded value, replace its tensor element type
  // with !quant.uniform<i8:f32, S:Z>. Patches up the func signature,
  // every func.return op, and tosa.const ops whose result type changed.
  void rewriteTypes(func::FuncOp f, DenseMap<Value, QZ> &qz,
                     IntegerType i8, FloatType f32) {
    auto qtyOf = [&](double s, int64_t z) {
      return quant::UniformQuantizedType::get(
          quant::QuantizationFlags::Signed, i8, f32, s, z,
          /*storageTypeMin=*/-128, /*storageTypeMax=*/127);
    };

    // Stamp every value's type. setType is a no-op when the type is
    // already what we want — safe to call unconditionally.
    for (auto &kv : qz) {
      Value v = kv.first;
      const QZ &rec = kv.second;
      auto rt = dyn_cast<RankedTensorType>(v.getType());
      if (!rt) continue;
      // Already a quant.uniform — skip (idempotent). Also skip tosa.const
      // results: `DenseElementsAttr::bitcast` to a `quant::QuantizedType`
      // hits an UNREACHABLE in MLIR's attribute-storage layer, and
      // downstream emission (RescaleConvFusion) re-derives Sw and
      // stamps timvx.const{quant_scale,quant_zp} from qz directly, so
      // the const result type doesn't need to carry the wrapper.
      if (isa<quant::UniformQuantizedType>(rt.getElementType())) continue;
      if (auto def = v.getDefiningOp()) {
        if (isa<tosa::ConstOp>(def)) continue;
      }
      auto qty = qtyOf(rec.scale, rec.zp);
      auto newRT = RankedTensorType::get(rt.getShape(), qty);
      v.setType(newRT);
    }

    // If any BlockArg got rewritten, the function signature must match.
    SmallVector<Type> newInputs(f.getArgumentTypes());
    bool sigChanged = false;
    for (auto en : llvm::enumerate(f.getArguments())) {
      newInputs[en.index()] = en.value().getType();
      if (newInputs[en.index()] != f.getArgumentTypes()[en.index()])
        sigChanged = true;
    }

    // Fix every return op's operand types (these were rewritten via
    // setType on the values themselves, but the func's recorded result
    // types haven't been updated yet).
    SmallVector<Type> newResults(f.getResultTypes());
    f.walk([&](func::ReturnOp ret) {
      for (auto en : llvm::enumerate(ret.getOperands())) {
        unsigned i = en.index();
        Type t = en.value().getType();
        if (t != newResults[i]) {
          newResults[i] = t;
          sigChanged = true;
        }
      }
    });

    if (sigChanged) {
      f.setType(FunctionType::get(f.getContext(), newInputs, newResults));
    }
  }
};

} // namespace

std::unique_ptr<Pass> createTosaQuantAnchorPass() {
  return std::make_unique<TosaQuantAnchorPass>();
}

} // namespace timvx
} // namespace mlir
