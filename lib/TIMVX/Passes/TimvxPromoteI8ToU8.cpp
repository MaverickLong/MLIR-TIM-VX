//===- TimvxPromoteI8ToU8.cpp - timvx-promote-i8-to-u8 -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Single-pass conversion of every quantized i8 tensor in the TIM-VX
// dialect to its u8 equivalent. After this pass runs, downstream passes
// see only u8 storage everywhere and don't need any promotion-aware
// logic. See the .td description for the rationale and pipeline order.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

#include "mlir/Dialect/Quant/IR/Quant.h"

namespace mlir {
namespace timvx {

#define GEN_PASS_DEF_TIMVXPROMOTEI8TOU8PASS
#include "TIMVX/TIMVXPasses.h.inc"

namespace {
using namespace ::mlir::timvx::detail;

// Build the u8-promoted equivalent of `elem`. Returns `elem` unchanged if
// it's not an i8-quantized type the pass touches.
inline Type promoteElementType(Type elem, MLIRContext *ctx) {
  if (auto qt = dyn_cast<quant::UniformQuantizedType>(elem)) {
    if (auto sty = dyn_cast<IntegerType>(qt.getStorageType())) {
      if (sty.getWidth() == 8 && qt.isSigned()) {
        Type ui8 = IntegerType::get(ctx, 8, IntegerType::Unsigned);
        // u8 storage range [0, 255]; the original i8 carried [-128, 127]
        // semantically, so the post-promotion zp = (orig_zp + 128) maps
        // the same real value to the same byte after `byte ^= 0x80`.
        return quant::UniformQuantizedType::get(
            /*flags=*/0, ui8, qt.getExpressedType(),
            qt.getScale(), qt.getZeroPoint() + 128,
            /*storageTypeMin=*/0, /*storageTypeMax=*/255);
      }
    }
  }
  if (auto qt = dyn_cast<quant::UniformQuantizedPerAxisType>(elem)) {
    if (auto sty = dyn_cast<IntegerType>(qt.getStorageType())) {
      if (sty.getWidth() == 8 && qt.isSigned()) {
        // Per-axis isn't expected on this chip's typical conv weights
        // (per-channel weights crash conv2d, see project memory), but
        // handle for completeness — same +128 transform per channel.
        Type ui8 = IntegerType::get(ctx, 8, IntegerType::Unsigned);
        SmallVector<int64_t> newZps;
        newZps.reserve(qt.getZeroPoints().size());
        for (int64_t z : qt.getZeroPoints()) newZps.push_back(z + 128);
        return quant::UniformQuantizedPerAxisType::get(
            /*flags=*/0, ui8, qt.getExpressedType(),
            qt.getScales(), newZps, qt.getQuantizedDimension(),
            /*storageTypeMin=*/0, /*storageTypeMax=*/255);
      }
    }
  }
  return elem;
}

// Promote a tensor type if its element type is i8-quantized.
inline Type promoteTensorType(Type ty, MLIRContext *ctx) {
  auto rt = dyn_cast<RankedTensorType>(ty);
  if (!rt) return ty;
  Type newElem = promoteElementType(rt.getElementType(), ctx);
  if (newElem == rt.getElementType()) return ty;
  return rt.cloneWith(rt.getShape(), newElem);
}

// Whether a value's type was rewritten to u8 by `promoteTensorType`.
inline bool wasPromotedToU8(Type ty) {
  if (auto rt = dyn_cast<RankedTensorType>(ty))
    if (auto qt = dyn_cast<quant::UniformQuantizedType>(rt.getElementType()))
      if (auto sty = dyn_cast<IntegerType>(qt.getStorageType()))
        return sty.getWidth() == 8 && sty.isUnsigned();
  return false;
}

// XOR each i8 byte with 0x80 to land on its u8 representation. Returns
// nullptr if `values` isn't a recognised i8 dense / resource attr.
ElementsAttr xorI8Bytes(ElementsAttr values, MLIRContext *ctx) {
  auto rt = dyn_cast<RankedTensorType>(values.getType());
  if (!rt) return {};
  Type oldElem = rt.getElementType();
  // Storage may be signless i8 (typical for `timvx.const`) or wrapped
  // in a quant type whose storage is i8.
  bool isSignedI8 = false;
  if (auto i = dyn_cast<IntegerType>(oldElem)) {
    isSignedI8 = (i.getWidth() == 8 && !i.isUnsigned());
  } else if (auto qt = dyn_cast<quant::QuantizedType>(oldElem)) {
    if (auto sty = dyn_cast<IntegerType>(qt.getStorageType()))
      isSignedI8 = (sty.getWidth() == 8 && !sty.isUnsigned());
  }
  if (!isSignedI8) return {};

  Type ui8 = IntegerType::get(ctx, 8, IntegerType::Unsigned);
  auto newRt = RankedTensorType::get(rt.getShape(), ui8);
  size_t numel = static_cast<size_t>(rt.getNumElements());

  if (auto dense = dyn_cast<DenseElementsAttr>(values)) {
    SmallVector<APInt> newVals;
    newVals.reserve(numel);
    if (dense.isSplat()) {
      APInt v = dense.getSplatValue<APInt>();
      uint8_t b = static_cast<uint8_t>(v.getZExtValue() & 0xFF) ^ 0x80;
      newVals.assign(numel, APInt(8, b, /*isSigned=*/false));
    } else {
      for (APInt v : dense.getValues<APInt>()) {
        uint8_t b = static_cast<uint8_t>(v.getZExtValue() & 0xFF) ^ 0x80;
        newVals.emplace_back(8, b, /*isSigned=*/false);
      }
    }
    return DenseElementsAttr::get(newRt, newVals);
  }
  if (auto r = dyn_cast<DenseI8ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return {};
    SmallVector<uint8_t> newBytes;
    newBytes.reserve(numel);
    for (int8_t v : *data)
      newBytes.push_back(static_cast<uint8_t>(v) ^ 0x80);
    SmallVector<APInt> newVals;
    newVals.reserve(numel);
    for (uint8_t b : newBytes)
      newVals.emplace_back(8, b, /*isSigned=*/false);
    return DenseElementsAttr::get(newRt, newVals);
  }
  return {};
}

// Try to update an existing `add(x, scalar_const)` or `sub(x, scalar_const)`
// op's RHS by `delta` (in real-value space). Returns true on success — the
// caller skips inserting a new op when this fires. Only matches when the
// RHS is a `timvx.const` with a single scalar f32 value (the canonical
// shape RequantI32SkipFold and the TFLite dequant chain produce).
template <typename BinOp>
bool tryFoldDeltaIntoConst(BinOp op, double delta) {
  Value rhs = op.getInput2();
  auto cst = rhs.getDefiningOp<ConstOp>();
  if (!cst) return false;
  if (!cst->hasOneUse()) return false;  // can't mutate a shared const
  auto valuesAttr = dyn_cast<DenseElementsAttr>(cst.getValuesAttr());
  if (!valuesAttr || !valuesAttr.isSplat()) return false;
  auto elemTy = cst.getType().getElementType();
  if (!elemTy.isF32()) return false;
  APFloat v = valuesAttr.getSplatValue<APFloat>();
  bool losesInfo = false;
  v.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &losesInfo);
  double newD = v.convertToDouble() + delta;
  APFloat newF(static_cast<float>(newD));
  auto newAttr = DenseElementsAttr::get(cst.getType(), newF);
  cst.setValuesAttr(newAttr);
  return true;
}

struct TIMVXPromoteI8ToU8Pass
    : public impl::TIMVXPromoteI8ToU8PassBase<TIMVXPromoteI8ToU8Pass> {
  void runOnOperation() final {
    ModuleOp m = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    // Step 1: Walk every Value in the module and rewrite its tensor type
    // to the u8 equivalent. Op results and block arguments share the same
    // setType(...) primitive, but FunctionType and tensor encodings need
    // a separate update.
    m.walk([&](Operation *op) {
      for (Value v : op->getResults()) {
        Type newTy = promoteTensorType(v.getType(), ctx);
        if (newTy != v.getType()) v.setType(newTy);
      }
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument arg : block.getArguments()) {
            Type newTy = promoteTensorType(arg.getType(), ctx);
            if (newTy != arg.getType()) arg.setType(newTy);
          }
    });

    // Step 2: For every func.func, refresh the FunctionType so it tracks
    // the (now u8) block arg / return operand types. The body has already
    // been updated by step 1.
    m.walk([&](func::FuncOp fn) {
      auto oldFn = fn.getFunctionType();
      SmallVector<Type> newIn, newOut;
      newIn.reserve(oldFn.getNumInputs());
      newOut.reserve(oldFn.getNumResults());
      for (Type t : oldFn.getInputs())
        newIn.push_back(promoteTensorType(t, ctx));
      for (Type t : oldFn.getResults())
        newOut.push_back(promoteTensorType(t, ctx));
      fn.setType(FunctionType::get(ctx, newIn, newOut));
    });

    // Step 3: Update `output_zp` discardable attrs on every op whose
    // result tensor was just promoted. The attr value pre-pass was the
    // signed-i8 zp; post-pass it should be the u8 zp = orig + 128.
    m.walk([&](Operation *op) {
      auto zpAttr = op->getAttrOfType<IntegerAttr>("output_zp");
      if (!zpAttr) return;
      bool anyPromoted = false;
      for (Value v : op->getResults())
        if (wasPromotedToU8(v.getType())) { anyPromoted = true; break; }
      if (!anyPromoted) return;
      op->setAttr("output_zp",
                  IntegerAttr::get(zpAttr.getType(), zpAttr.getInt() + 128));
    });

    // Step 4: Rewrite `timvx.const` ops with i8-derived data:
    //  * If the result type was promoted to u8 (asym i8 const consumed by
    //    a downstream op that expects u8), XOR the bytes.
    //  * Shift `quant_zp` by +128 to match the new storage interpretation.
    // The const's result type was set by step 1 (since its result Value
    // got the type-promotion treatment).
    m.walk([&](ConstOp cst) {
      if (!wasPromotedToU8(cst.getType())) {
        // Symmetric weight that wasn't a quant.uniform<i8> originally —
        // still need to flip bytes if its type is signless i8 AND a
        // quant_scale/quant_zp attr is present (a TIM-VX-emitted weight).
        // After flipping bytes, also flip the type to ui8 + shift zp.
        auto ty = cst.getType();
        Type elem = ty.getElementType();
        auto i = dyn_cast<IntegerType>(elem);
        if (!i || i.getWidth() != 8 || i.isUnsigned()) return;
        if (!cst.getQuantScaleAttr() || !cst.getQuantZpAttr()) return;
        // Build u8 result type, XOR bytes, shift zp.
        ElementsAttr newVals = xorI8Bytes(
            cast<ElementsAttr>(cst.getValuesAttr()), ctx);
        if (!newVals) return;
        cst.setValuesAttr(newVals);
        Type ui8 = IntegerType::get(ctx, 8, IntegerType::Unsigned);
        cst.getResult().setType(
            cast<RankedTensorType>(ty).cloneWith(ty.getShape(), ui8));
        int64_t newZp = cst.getQuantZpAttr().getInt() + 128;
        cst.setQuantZpAttr(
            IntegerAttr::get(cst.getQuantZpAttr().getType(), newZp));
        return;
      }
      // Promoted asym path: flip bytes, shift zp.
      ElementsAttr newVals = xorI8Bytes(
          cast<ElementsAttr>(cst.getValuesAttr()), ctx);
      if (!newVals) return;
      cst.setValuesAttr(newVals);
      if (auto z = cst.getQuantZpAttr())
        cst.setQuantZpAttr(IntegerAttr::get(z.getType(), z.getInt() + 128));
    });

    // Step 5: Insert ±128 compensation around every `timvx.cast` that
    // crosses the f32 ↔ u8 boundary. The TIM-VX cast op is a value-cast
    // (treats bytes as raw integers); under the u8 storage relabel, the
    // numeric representation in f32 land needs the +128 subtracted on
    // the dequant side and added on the requant side.
    //
    // Optimization: if the immediate neighbour is a `timvx.add` /
    // `timvx.sub` with a scalar f32 const RHS — the canonical shape of
    // the TFLite dequant/requant chain — fold the +128 into that const
    // instead of emitting a fresh op. That keeps the IR free of stray
    // `+128` adds that would otherwise break QRF's tail-add pattern
    // match.
    SmallVector<CastOp> castsToProcess;
    m.walk([&](CastOp cst) { castsToProcess.push_back(cst); });
    for (CastOp cst : castsToProcess) {
      auto srcTy = dyn_cast<RankedTensorType>(cst.getInput().getType());
      auto dstTy = dyn_cast<RankedTensorType>(cst.getType());
      if (!srcTy || !dstTy) continue;
      bool srcIsU8 = wasPromotedToU8(srcTy);
      bool dstIsU8 = wasPromotedToU8(dstTy);
      bool srcIsF32 = srcTy.getElementType().isF32();
      bool dstIsF32 = dstTy.getElementType().isF32();

      auto f32SplatConst = [&](Location loc, float v) -> Value {
        auto sty = RankedTensorType::get({1}, builder.getF32Type());
        auto attr = DenseElementsAttr::get(
            sty, builder.getF32FloatAttr(v).getValue());
        return ConstOp::create(builder, loc, sty, attr,
                               /*quant_scale=*/FloatAttr{},
                               /*quant_zp=*/IntegerAttr{});
      };

      if (srcIsU8 && dstIsF32) {
        // cast(u8→f32) emits 0..255; subtract 128.0 to land on the
        // [-128, 127] real-value space the rest of the chain expects.
        // Try folding into a downstream `sub(_, c)` first.
        bool folded = false;
        for (Operation *user : cst->getUsers()) {
          if (auto sub = dyn_cast<SubOp>(user)) {
            if (sub.getInput1() == cst.getResult() &&
                tryFoldDeltaIntoConst(sub, +128.0)) {
              folded = true;
              break;
            }
          }
        }
        if (folded) continue;
        builder.setInsertionPointAfter(cst);
        Value c128 = f32SplatConst(cst.getLoc(), 128.0f);
        Value bridged =
            SubOp::create(builder, cst.getLoc(), dstTy, cst.getResult(), c128);
        cst.getResult().replaceAllUsesExcept(
            bridged, bridged.getDefiningOp());
      } else if (srcIsF32 && dstIsU8) {
        // cast(f32→u8) needs the real-value input pre-shifted by +128 so
        // `(uint8_t)round(f32)` lands on the right byte under the u8
        // relabel. Try folding into the upstream `add(_, c)` first.
        bool folded = false;
        if (auto add = cst.getInput().getDefiningOp<AddOp>())
          if (tryFoldDeltaIntoConst(add, +128.0))
            folded = true;
        if (folded) continue;
        builder.setInsertionPoint(cst);
        Value c128 = f32SplatConst(cst.getLoc(), 128.0f);
        Value shifted =
            AddOp::create(builder, cst.getLoc(), srcTy, cst.getInput(), c128);
        cst.setOperand(shifted);
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createTIMVXPromoteI8ToU8Pass() {
  return std::make_unique<TIMVXPromoteI8ToU8Pass>();
}

} // namespace timvx
} // namespace mlir
