//===- Common.h - Shared helpers for TIMVX pass implementations -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal-only header used by every cpp under `lib/TIMVX/Passes/`.
// Aggregates the small helpers shared by multiple passes:
//
//   * Constant matching: `isConstantZero`, `matchConstScalarInt`,
//     `matchConstScalarFloat`, plus the FP32 broadcast-aware
//     `ConstF32View` / `foldBinaryFP` / `foldUnaryFP` family used by
//     `tosa-const-fold`'s elementwise patterns.
//   * NHWC <-> WHCN wrapping: `kPermNHWCToWHCN`, `kPermWHCNToNHWC`,
//     `applyPerm`, `wrapWithTranspose`, plus the TOSA-vs-TIM-VX
//     attribute reorderings (`tosaPadToTIMVX`, `tosaHWToTIMVXWH`).
//   * Dense-attr permute: `applyPermToShape`,
//     `computePermutationIndices`, `permuteDenseElements`, `permToInt64`
//     — used by the `TransposeOp` canonicalizers.
//   * Quant plumbing: `QuantInfo` / `QuantInfoMap` / `qmapAttrs` /
//     `consumerExpectedZp` / `tryRecoverChainQuant` / `buildQuantInfoMap`,
//     and the `getProducerQuant` lookup used by `QuantResidualFuse`.
//
//===----------------------------------------------------------------------===//

#ifndef TIMVX_PASSES_COMMON_H
#define TIMVX_PASSES_COMMON_H

#include "TIMVX/TIMVXDialect.h"
#include "TIMVX/TIMVXOps.h"
#include "TIMVX/TIMVXPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <array>
#include <cmath>
#include <optional>

namespace mlir {
namespace timvx {
namespace detail {

//===----------------------------------------------------------------------===//
// Constant-operand matchers
//===----------------------------------------------------------------------===//

inline bool isConstantZero(Value v) {
  return matchPattern(v, m_Zero()) || matchPattern(v, m_AnyZeroFloat());
}

// Read a scalar / 1-element integer from a constant. Used to extract scalar
// zp / multiplier / shift operands that quantized TOSA carries on conv2d /
// rescale / pad in operand form.
inline std::optional<int64_t> matchConstScalarInt(Value v) {
  ElementsAttr attr;
  if (!matchPattern(v, m_Constant(&attr)))
    return std::nullopt;
  auto dense = dyn_cast<DenseIntElementsAttr>(attr);
  if (!dense || dense.getNumElements() < 1)
    return std::nullopt;
  return (*dense.getValues<APInt>().begin()).getSExtValue();
}

// First-element scalar read for any 1-element splat float const
// (`tosa.const` or `timvx.const`). `timvx.const` is deliberately not
// `ConstantLike`, so `m_Constant` can't see it; we hit both dialects'
// const ops by extracting the `values` ElementsAttr directly.
inline std::optional<double> matchConstScalarFloat(Value v) {
  ElementsAttr attr;
  if (auto tc = v.getDefiningOp<tosa::ConstOp>())
    attr = dyn_cast_or_null<ElementsAttr>(tc.getValuesAttr());
  else if (auto vc = v.getDefiningOp<ConstOp>())
    attr = dyn_cast_or_null<ElementsAttr>(vc.getValuesAttr());
  else if (!matchPattern(v, m_Constant(&attr)))
    return std::nullopt;
  if (!attr) return std::nullopt;
  auto dense = dyn_cast<DenseFPElementsAttr>(attr);
  if (!dense || dense.getNumElements() < 1)
    return std::nullopt;
  return (*dense.getValues<APFloat>().begin()).convertToDouble();
}

//===----------------------------------------------------------------------===//
// FP32 broadcast-aware constant folding (used by `tosa-const-fold`)
//===----------------------------------------------------------------------===//

struct ConstF32View {
  ArrayRef<int64_t> shape;
  SmallVector<float> data;
};

inline std::optional<ConstF32View> getConstF32(Value v) {
  auto c = v.getDefiningOp<tosa::ConstOp>();
  if (!c)
    return std::nullopt;
  auto attr = dyn_cast<ElementsAttr>(c.getValuesAttr());
  if (!attr)
    return std::nullopt;

  auto rt = dyn_cast<RankedTensorType>(attr.getType());
  if (!rt || !rt.hasStaticShape() || !rt.getElementType().isF32())
    return std::nullopt;

  ConstF32View view;
  view.shape = rt.getShape();
  view.data.reserve(rt.getNumElements());

  if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
    for (APFloat f : dense.getValues<APFloat>())
      view.data.push_back(f.convertToFloat());
  } else if (auto r = dyn_cast<DenseF32ResourceElementsAttr>(attr)) {
    auto arr = r.tryGetAsArrayRef();
    if (!arr)
      return std::nullopt;
    view.data.assign(arr->begin(), arr->end());
  } else {
    return std::nullopt;
  }
  return view;
}

inline SmallVector<int64_t> broadcastShape(ArrayRef<int64_t> a,
                                            ArrayRef<int64_t> b) {
  size_t maxRank = std::max(a.size(), b.size());
  SmallVector<int64_t> out(maxRank);
  for (size_t i = 0; i < maxRank; ++i) {
    int64_t ai = i < a.size() ? a[a.size() - 1 - i] : 1;
    int64_t bi = i < b.size() ? b[b.size() - 1 - i] : 1;
    if (ai != 1 && bi != 1 && ai != bi)
      return {};
    out[maxRank - 1 - i] = std::max(ai, bi);
  }
  return out;
}

inline size_t broadcastSrc(size_t dst, ArrayRef<int64_t> outShape,
                           ArrayRef<int64_t> inShape) {
  SmallVector<int64_t, 8> coord(outShape.size());
  for (int64_t i = (int64_t)outShape.size() - 1; i >= 0; --i) {
    coord[i] = (int64_t)(dst % outShape[i]);
    dst /= outShape[i];
  }
  size_t pad = outShape.size() - inShape.size();
  size_t src = 0;
  for (size_t i = 0; i < inShape.size(); ++i) {
    int64_t c = inShape[i] == 1 ? 0 : coord[i + pad];
    src = src * inShape[i] + c;
  }
  return src;
}

inline void replaceWithF32Const(Operation *op, ArrayRef<int64_t> outShape,
                                ArrayRef<float> data,
                                PatternRewriter &rewriter) {
  auto f32 = rewriter.getF32Type();
  auto outTy = RankedTensorType::get(outShape, f32);
  rewriter.replaceOpWithNewOp<tosa::ConstOp>(
      op, outTy, DenseElementsAttr::get(outTy, data));
}

template <typename Compute>
LogicalResult foldBinaryFP(Operation *op, Value lhs, Value rhs, Compute compute,
                            PatternRewriter &rewriter) {
  auto lhsC = getConstF32(lhs);
  auto rhsC = getConstF32(rhs);
  if (!lhsC || !rhsC)
    return failure();

  SmallVector<int64_t> outShape = broadcastShape(lhsC->shape, rhsC->shape);
  if (outShape.empty() && (!lhsC->shape.empty() || !rhsC->shape.empty()))
    return failure();

  int64_t numel = 1;
  for (int64_t d : outShape) numel *= d;
  if (numel == 0) numel = 1; // both rank-0 scalars

  SmallVector<float> result(numel);
  for (size_t i = 0; i < (size_t)numel; ++i) {
    size_t li = broadcastSrc(i, outShape, lhsC->shape);
    size_t ri = broadcastSrc(i, outShape, rhsC->shape);
    result[i] = compute(lhsC->data[li], rhsC->data[ri]);
  }
  replaceWithF32Const(op, outShape, result, rewriter);
  return success();
}

template <typename Compute>
LogicalResult foldUnaryFP(Operation *op, Value src, Compute compute,
                          PatternRewriter &rewriter) {
  auto srcC = getConstF32(src);
  if (!srcC)
    return failure();

  SmallVector<float> result(srcC->data.size());
  for (size_t i = 0; i < srcC->data.size(); ++i)
    result[i] = compute(srcC->data[i]);
  replaceWithF32Const(op, srcC->shape, result, rewriter);
  return success();
}

inline float fAdd(float a, float b) { return a + b; }
inline float fSub(float a, float b) { return a - b; }
inline float fMul(float a, float b) { return a * b; }
inline float fPow(float a, float b) { return std::pow(a, b); }
inline float fReciprocal(float x) { return 1.0f / x; }

//===----------------------------------------------------------------------===//
// NHWC <-> WHCN wrapping helpers (used by `tosa-to-timvx` spatial-op
// converters and the TransposeOp canonicalizers).
//===----------------------------------------------------------------------===//

// NHWC -> WHCN: output_dim_i = input_dim_perm[i], perm = [W, H, C, N].
// WHCN -> NHWC: inverse.
inline constexpr std::array<int32_t, 4> kPermNHWCToWHCN = {2, 1, 3, 0};
inline constexpr std::array<int32_t, 4> kPermWHCNToNHWC = {3, 1, 0, 2};

template <typename T>
SmallVector<T> applyPerm(ArrayRef<T> src, ArrayRef<int32_t> perm) {
  SmallVector<T> out(perm.size());
  for (size_t i = 0; i < perm.size(); ++i)
    out[i] = src[perm[i]];
  return out;
}

inline SmallVector<int64_t> applyPermToShape(ArrayRef<int64_t> shape,
                                              ArrayRef<int64_t> perm) {
  SmallVector<int64_t> out;
  out.reserve(perm.size());
  for (int64_t p : perm)
    out.push_back(shape[p]);
  return out;
}

inline SmallVector<int64_t> permToInt64(ArrayRef<int32_t> perm) {
  return SmallVector<int64_t>(perm.begin(), perm.end());
}

// Compute the permutation `outIndex -> inIndex` for a row-major buffer
// reshape from `inShape` to `outShape = applyPermToShape(inShape, perm)`.
// Returns a vector of size product(inShape).
inline SmallVector<int64_t>
computePermutationIndices(ArrayRef<int64_t> inShape, ArrayRef<int64_t> perm,
                          ArrayRef<int64_t> outShape) {
  int64_t rank = inShape.size();
  int64_t numel = 1;
  for (int64_t d : inShape) numel *= d;
  SmallVector<int64_t> inStrides(rank), outStrides(rank);
  inStrides[rank - 1] = 1;
  outStrides[rank - 1] = 1;
  for (int64_t i = rank - 2; i >= 0; --i) {
    inStrides[i] = inStrides[i + 1] * inShape[i + 1];
    outStrides[i] = outStrides[i + 1] * outShape[i + 1];
  }
  SmallVector<int64_t> indices(numel);
  for (int64_t outLin = 0; outLin < numel; ++outLin) {
    int64_t rem = outLin;
    int64_t inLin = 0;
    for (int64_t i = 0; i < rank; ++i) {
      int64_t coord = rem / outStrides[i];
      rem -= coord * outStrides[i];
      inLin += coord * inStrides[perm[i]];
    }
    indices[outLin] = inLin;
  }
  return indices;
}

// Permute the values of a dense / dense-resource attribute according to
// `perm`. Handles f32, i8, i16, i32 storage; returns nullptr otherwise.
inline DenseElementsAttr permuteDenseElements(ElementsAttr in,
                                               ArrayRef<int64_t> perm) {
  auto inTy = dyn_cast<RankedTensorType>(in.getType());
  if (!inTy || !inTy.hasStaticShape())
    return nullptr;
  ArrayRef<int64_t> inShape = inTy.getShape();
  SmallVector<int64_t> outShape = applyPermToShape(inShape, perm);
  auto outTy = RankedTensorType::get(outShape, inTy.getElementType());

  Type elem = inTy.getElementType();
  int64_t numel = inTy.getNumElements();
  auto idx = computePermutationIndices(inShape, perm, outShape);

  if (elem.isF32()) {
    SmallVector<float> src;
    src.reserve(numel);
    if (auto dense = dyn_cast<DenseElementsAttr>(in)) {
      for (APFloat v : dense.getValues<APFloat>())
        src.push_back(v.convertToFloat());
    } else if (auto r = dyn_cast<DenseF32ResourceElementsAttr>(in)) {
      auto arr = r.tryGetAsArrayRef();
      if (!arr) return nullptr;
      src.assign(arr->begin(), arr->end());
    } else {
      return nullptr;
    }
    SmallVector<float> out(numel);
    for (int64_t i = 0; i < numel; ++i) out[i] = src[idx[i]];
    return DenseElementsAttr::get(outTy, ArrayRef<float>(out));
  }

  if (auto it = dyn_cast<IntegerType>(elem)) {
    unsigned bits = it.getWidth();
    SmallVector<APInt> src;
    src.reserve(numel);

    if (auto dense = dyn_cast<DenseElementsAttr>(in)) {
      for (APInt v : dense.getValues<APInt>()) src.push_back(v);
    } else if (auto r = dyn_cast<DenseI8ResourceElementsAttr>(in)) {
      auto arr = r.tryGetAsArrayRef();
      if (!arr) return nullptr;
      for (int8_t v : *arr) src.emplace_back(bits, uint64_t(v), /*signed=*/true);
    } else if (auto r = dyn_cast<DenseI16ResourceElementsAttr>(in)) {
      auto arr = r.tryGetAsArrayRef();
      if (!arr) return nullptr;
      for (int16_t v : *arr) src.emplace_back(bits, uint64_t(v), true);
    } else if (auto r = dyn_cast<DenseI32ResourceElementsAttr>(in)) {
      auto arr = r.tryGetAsArrayRef();
      if (!arr) return nullptr;
      for (int32_t v : *arr) src.emplace_back(bits, uint64_t(v), true);
    } else if (auto r = dyn_cast<DenseUI8ResourceElementsAttr>(in)) {
      auto arr = r.tryGetAsArrayRef();
      if (!arr) return nullptr;
      for (uint8_t v : *arr) src.emplace_back(bits, uint64_t(v), false);
    } else {
      return nullptr;
    }
    SmallVector<APInt> out(numel, APInt(bits, 0));
    for (int64_t i = 0; i < numel; ++i) out[i] = src[idx[i]];
    return DenseElementsAttr::get(outTy, out);
  }

  return nullptr;
}

// Wrap `value` in a `timvx.transpose` carrying `perm`. The returned value's
// MLIR type is `value`'s shape permuted by `perm` (same element type;
// optional `outputScale`/`outputZp` forwarded onto the transpose's quant
// attrs so downstream `TensorSpec` emission keeps the right Quantization()).
inline Value wrapWithTranspose(OpBuilder &builder, Location loc, Value value,
                                ArrayRef<int32_t> perm,
                                FloatAttr outputScale = {},
                                IntegerAttr outputZp = {}) {
  auto srcTy = cast<RankedTensorType>(value.getType());
  auto srcShape = srcTy.getShape();
  assert(perm.size() == srcShape.size() && "perm rank mismatch");
  auto dstShape = applyPerm<int64_t>(srcShape, perm);
  auto dstTy = srcTy.clone(dstShape);
  return TransposeOp::create(builder, loc, dstTy, value,
                              builder.getDenseI32ArrayAttr(perm),
                              outputScale, outputZp);
}

// TOSA conv/pool pad is `[top, bottom, left, right]`; TIM-VX expects
// `[left, right, top, bottom]`.
inline DenseI64ArrayAttr tosaPadToTIMVX(OpBuilder &builder,
                                         ArrayRef<int64_t> tosaPad) {
  assert(tosaPad.size() == 4 && "tosa pad must be rank-4 [t,b,l,r]");
  return builder.getDenseI64ArrayAttr(
      {tosaPad[2], tosaPad[3], tosaPad[0], tosaPad[1]});
}

// TOSA spatial `[H, W]` -> TIM-VX `[W, H]` (used for stride / dilation /
// kernel size).
inline DenseI64ArrayAttr tosaHWToTIMVXWH(OpBuilder &builder,
                                          ArrayRef<int64_t> tosaHW) {
  assert(tosaHW.size() == 2 && "tosa stride/dilation/kernel must be [H,W]");
  return builder.getDenseI64ArrayAttr({tosaHW[1], tosaHW[0]});
}

//===----------------------------------------------------------------------===//
// Generic 1-to-1 op rewrites (used by `tosa-to-timvx`)
//===----------------------------------------------------------------------===//

// Lower a TOSA op to a TIM-VX op when the source op takes only tensor
// operands and no attributes.
template <typename TosaOp, typename TIMVXOp>
struct TensorOnlyOpConversion : public OpConversionPattern<TosaOp> {
  using OpConversionPattern<TosaOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<TosaOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(TosaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<TIMVXOp>(op, op->getResultTypes(),
                                         adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Quantization plumbing
//===----------------------------------------------------------------------===//
//
// Quantized TOSA from the tflite import has the structure
//
//     tosa.conv2d (i8 in, i8 weight, i32 bias) -> i32 (accumulator)
//     tosa.rescale  i32 -> i8 (multiplier, shift, output_zp)
//
// TIM-VX wants the conv to be a single op carrying per-tensor (Si, Sw, So,
// in_zp, out_zp); it derives the integer multiplier+shift internally from
// those scales. So we fuse the pair, choosing canonical absolute scales:
//
//   Sw  = 1/128                          (symmetric int8 weight, weight_zp = 0)
//   Si  = (chained from upstream)        (input scale; anchor is 1.0 at func arg)
//   So  = Si * Sw / M_eff,  where M_eff = mul / 2^shift
//   Sb  = Si * Sw                        (i32 bias: scale = Si*Sw, zp = 0)

struct QuantInfo {
  double scale;
  int64_t zp;
};

struct QuantInfoMap {
  llvm::DenseMap<Value, QuantInfo> map;

  void seedFuncArgs(func::FuncOp f) {
    for (BlockArgument a : f.getArguments()) {
      auto rt = dyn_cast<RankedTensorType>(a.getType());
      if (!rt) continue;
      auto it = dyn_cast<IntegerType>(rt.getElementType());
      if (!it || it.getWidth() > 8) continue;
      // Anchor at scale=1.0 zp=0 by default — the int8 conv chain only
      // depends on scale ratios, so a fixed anchor produces the same int8
      // byte pattern. Two optional arg-attr conventions override:
      //
      //   * `timvx.output_scale` / `timvx.output_zp` — canonical;
      //     mirrors the discardable attr names every quant-aware timvx
      //     op carries on its result. `getProducerQuant` reads the same
      //     names for `BlockArgument`s, so the seeded map value matches
      //     the producer-quant lookup downstream.
      //
      //   * `tim::vx::input_scale` / `tim::vx::input_zp` — legacy;
      //     accepted for back-compat with hand-written tests pre-dating
      //     the unified `timvx.output_*` convention.
      double s = 1.0;
      int64_t z = 0;
      unsigned idx = a.getArgNumber();
      if (auto sa = f.getArgAttrOfType<FloatAttr>(idx, "timvx.output_scale"))
        s = sa.getValueAsDouble();
      else if (auto sa = f.getArgAttrOfType<FloatAttr>(
                   idx, "tim::vx::input_scale"))
        s = sa.getValueAsDouble();
      if (auto za = f.getArgAttrOfType<IntegerAttr>(idx, "timvx.output_zp"))
        z = za.getInt();
      else if (auto za = f.getArgAttrOfType<IntegerAttr>(
                   idx, "tim::vx::input_zp"))
        z = za.getInt();
      map[a] = {s, z};
    }
  }

  // Read `timvx.output_scale` / `timvx.output_zp` from `func.func`'s
  // `res_attrs` (the result-side counterpart to `seedFuncArgs`'s
  // `arg_attrs`). For each return op in `f`, when the corresponding
  // res_attr declares a (scale, zp), pre-seed QM[return-operand].
  // The forward walk in `buildQuantInfoMap` checks the pre-seed
  // before falling through to chain recovery, so this is how a hand-
  // written atomic test (which has no downstream dequant chain to
  // pin So through) can declare its terminal output quant context.
  void seedFuncResults(func::FuncOp f) {
    f.walk([&](func::ReturnOp ret) {
      for (auto [idx, v] : llvm::enumerate(ret.getOperands())) {
        auto rt = dyn_cast<RankedTensorType>(v.getType());
        if (!rt) continue;
        auto it = dyn_cast<IntegerType>(rt.getElementType());
        if (!it || it.getWidth() > 8) continue;
        auto sa = f.getResultAttrOfType<FloatAttr>(idx, "timvx.output_scale");
        auto za = f.getResultAttrOfType<IntegerAttr>(idx, "timvx.output_zp");
        if (!sa || !za) continue;
        map[v] = {sa.getValueAsDouble(), za.getInt()};
      }
    });
  }

  std::optional<QuantInfo> lookup(Value v) const {
    auto it = map.find(v);
    if (it == map.end()) return std::nullopt;
    return it->second;
  }

  void set(Value v, QuantInfo qi) { map[v] = qi; }
};

template <typename Builder>
inline std::pair<FloatAttr, IntegerAttr> qmapAttrs(const QuantInfoMap &qm,
                                                    Value v, Builder &b) {
  auto qi = qm.lookup(v);
  if (!qi) return {FloatAttr{}, IntegerAttr{}};
  return {b.getF64FloatAttr(qi->scale), b.getI64IntegerAttr(qi->zp)};
}

// Walk forward through layout-/range-preserving ops looking for a consumer
// whose attributes pin the expected zero-point of `v`. Returns the first
// match (e.g. `tosa.conv2d.input_zp` or `tosa.pad.pad_const`).
inline std::optional<int64_t> consumerExpectedZp(Value v) {
  llvm::SmallPtrSet<Operation *, 16> visited;
  llvm::SmallVector<Value, 8> stack{v};
  while (!stack.empty()) {
    Value cur = stack.pop_back_val();
    for (Operation *user : cur.getUsers()) {
      if (!visited.insert(user).second) continue;
      if (auto conv = dyn_cast<tosa::Conv2DOp>(user))
        if (auto z = matchConstScalarInt(conv.getInputZp()))
          return *z;
      if (auto pad = dyn_cast<tosa::PadOp>(user)) {
        if (auto pc = pad.getPadConst().getDefiningOp<tosa::ConstOp>()) {
          if (auto attr = dyn_cast<DenseIntElementsAttr>(pc.getValuesAttr())) {
            if (attr.getNumElements() >= 1)
              return (*attr.getValues<APInt>().begin()).getSExtValue();
          }
        }
      }
      if (isa<tosa::TransposeOp, tosa::ReshapeOp, tosa::SliceOp>(user))
        if (user->getNumResults() == 1)
          stack.push_back(user->getResult(0));
    }
  }
  return std::nullopt;
}

// Walk forward from a rescale's int output through the standard tflite-
// emitted dequant chain `cast(int->f32) -> sub(_, zp) -> mul(_, scale)`.
// Returns the (scale, zp) the chain reconstructs.
//
// Skips zero-or-more quant-preserving ops (pad / pool / transpose /
// reshape / slice) between the rescale and the cast — TFLite often
// places one of these between the conv-rescale and the dequant chain
// (e.g. the first stem in resnet: conv -> rescale -> pad -> maxpool ->
// cast -> sub -> mul). The intermediates carry the same (S, Z) by the
// QM propagation rule, so naming `So` from the eventual `mul` const is
// still valid.
inline std::optional<std::pair<double, int64_t>>
tryRecoverChainQuant(Value rescaleOut) {
  Value cur = rescaleOut;
  while (cur.hasOneUse()) {
    Operation *user = *cur.getUsers().begin();
    if (isa<tosa::MaxPool2dOp, tosa::AvgPool2dOp, tosa::PadOp,
            tosa::TransposeOp, tosa::ReshapeOp, tosa::SliceOp>(user) &&
        user->getNumResults() == 1) {
      cur = user->getResult(0);
      continue;
    }
    break;
  }
  if (!cur.hasOneUse()) return std::nullopt;
  auto cast = dyn_cast<tosa::CastOp>(*cur.getUsers().begin());
  if (!cast || !cast->hasOneUse()) return std::nullopt;
  auto castOutTy = dyn_cast<RankedTensorType>(cast.getType());
  if (!castOutTy || !castOutTy.getElementType().isF32())
    return std::nullopt;

  auto sub = dyn_cast<tosa::SubOp>(*cast->getUsers().begin());
  if (!sub || !sub->hasOneUse()) return std::nullopt;
  if (sub.getInput1() != cast.getResult()) return std::nullopt;
  auto zpVal = matchConstScalarFloat(sub.getInput2());
  if (!zpVal) return std::nullopt;

  auto mul = dyn_cast<tosa::MulOp>(*sub->getUsers().begin());
  if (!mul) return std::nullopt;
  Value subOut = sub.getResult();
  Value scaleOperand;
  if (mul.getInput1() == subOut) scaleOperand = mul.getInput2();
  else if (mul.getInput2() == subOut) scaleOperand = mul.getInput1();
  else return std::nullopt;
  auto scaleVal = matchConstScalarFloat(scaleOperand);
  if (!scaleVal) return std::nullopt;

  return std::make_pair(*scaleVal,
                         static_cast<int64_t>(std::lround(*zpVal)));
}

// Forward decl: legacy derivation, called only when the anchor pass
// hasn't run (no tensor types carry !quant.uniform).
inline void legacyBuildQuantInfoMap(func::FuncOp f, QuantInfoMap &qm,
                                     bool *strictFailed);

// Read absolute (S, Z) for every quantized i8 SSA value off the
// `!quant.uniform<i8:f32, S:Z>` element type that `tosa-quant-anchor`
// stamped during the pre-walk. With anchoring centralized in that pass,
// this function is now a thin type-decoder: walk every BlockArg and
// every op result whose tensor element type is a UniformQuantizedType
// and copy (S, Z) into the QM.
//
// Legacy fallback paths (for IR not anchored by `tosa-quant-anchor` —
// e.g. tests that invoke `--tosa-to-timvx` directly): fall through to
// the original derivation logic so existing callers don't break in a
// transition window. Once everything goes through the anchor pass, the
// fallback walk can be deleted.
//
// `strictFailed` (out): preserved for ABI; populated only by the
// fallback walk (when it exhausts every derivation path) — type-stamped
// IR cannot trip this because the IR's types ARE the QM.
inline void buildQuantInfoMap(func::FuncOp f, QuantInfoMap &qm,
                               bool *strictFailed) {
  // Fast path: read (S, Z) off `!quant.uniform` element types.
  bool anchorPassRan = false;
  auto recordIfQuant = [&](Value v) {
    auto rt = dyn_cast<RankedTensorType>(v.getType());
    if (!rt) return;
    auto qty = dyn_cast<quant::UniformQuantizedType>(rt.getElementType());
    if (!qty) return;
    qm.set(v, {qty.getScale(), qty.getZeroPoint()});
    anchorPassRan = true;
  };
  for (BlockArgument a : f.getArguments()) recordIfQuant(a);
  f.walk([&](Operation *op) {
    for (Value r : op->getResults()) recordIfQuant(r);
  });
  // Even when the anchor pass ran we still want the cast-attr scan
  // below — RequantI32SkipFold tags the requant tail's `tosa.cast`
  // with `timvx.output_scale`/`zp` discardable attrs, but the cast's
  // result type stays plain i8 (a tosa.cast can't be the producer of
  // a quantized tensor without the anchor pass touching it). Read
  // those attrs so `CastConversion` can stamp Quantization() on the
  // downstream tensor spec.
  f.walk([&](tosa::CastOp cast) {
    if (qm.lookup(cast.getResult())) return;
    auto scaleAttr = cast->getAttrOfType<FloatAttr>("timvx.output_scale");
    auto zpAttr = cast->getAttrOfType<IntegerAttr>("timvx.output_zp");
    if (scaleAttr && zpAttr)
      qm.set(cast.getResult(),
             {scaleAttr.getValueAsDouble(), zpAttr.getInt()});
  });
  if (anchorPassRan) return;

  // Legacy fallback (anchor pass didn't run): the original derivation.
  qm.seedFuncArgs(f);
  qm.seedFuncResults(f);

  // After seeding, refine each func arg's zp using the first consumer's
  // expectation. Without this we'd anchor at zp=0, which mismatches models
  // whose tflite quantization assigns a non-zero input_zp.
  for (BlockArgument a : f.getArguments()) {
    auto qi = qm.lookup(a);
    if (!qi) continue;
    if (auto z = consumerExpectedZp(a)) qm.set(a, {qi->scale, *z});
  }

  legacyBuildQuantInfoMap(f, qm, strictFailed);
}

// Original (pre-anchor-pass) derivation logic. Kept as a fallback so
// IR that hasn't been through `tosa-quant-anchor` still lowers.
inline void legacyBuildQuantInfoMap(func::FuncOp f, QuantInfoMap &qm,
                                     bool *strictFailed) {
  qm.seedFuncArgs(f);
  qm.seedFuncResults(f);

  // After seeding, refine each func arg's zp using the first consumer's
  // expectation. Without this we'd anchor at zp=0, which mismatches models
  // whose tflite quantization assigns a non-zero input_zp.
  for (BlockArgument a : f.getArguments()) {
    auto qi = qm.lookup(a);
    if (!qi) continue;
    if (auto z = consumerExpectedZp(a)) qm.set(a, {qi->scale, *z});
  }

  f.walk([&](Operation *op) {
    if (auto resc = dyn_cast<tosa::RescaleOp>(op)) {
      auto mul = matchConstScalarInt(resc.getMultiplier());
      auto shift = matchConstScalarInt(resc.getShift());
      auto outZp = matchConstScalarInt(resc.getOutputZp());
      if (!mul || !shift || !outZp) return;
      double M = static_cast<double>(*mul) * std::pow(2.0, -double(*shift));

      auto conv = resc.getInput().getDefiningOp<tosa::Conv2DOp>();
      if (conv) {
        // Conv-rescale fusion target. M = (Si * Sw) / So. Si comes from
        // the conv's input (a func arg or a previous rescale's result —
        // both are guaranteed to be in QM by construction; the lookup
        // is strict-checked here).
        //
        // So is resolved in priority order:
        //   (1) `seedFuncResults` pre-seeded QM[resc.result] from the
        //       func's `timvx.output_scale`/`zp` res_attrs (most
        //       explicit; the right way for hand-written tests).
        //   (2) the dequant chain consuming resc.result names So
        //       directly (TFLite residual chains where the conv's
        //       output goes through cast→sub→mul before the next op).
        //   (3) Sw = 1/128 convention (TFLite-canonical symmetric
        //       int8 weight; back-compute So from M and Si).
        //
        // Why path (3) is non-strict: TOSA's RescaleOp encodes only M
        // (the int multiplier/shift ratio), not absolute Si/Sw/So.
        // Standard TFLite-quantized models export with M but no
        // weight-side scale, so an int8-inference graph (rn18/rn50)
        // with no fp32 dequant chain has no in-IR signal to pin Sw.
        // The Sw=1/128 convention is byte-output equivalent (the
        // conv's TIM-VX scale_factor stays exactly M regardless of
        // which Sw we pick); only the dequant printout differs.
        // Real bugs in the chain are caught upstream by
        // QuantResidualFuse's strict cross-check, which is the
        // intended bug-catcher rather than this convention.
        auto siOpt = qm.lookup(conv.getInput());
        if (!siOpt) {
          conv->emitError()
              << "buildQuantInfoMap: conv input has no tracked quant "
                 "info — every conv input must be a func arg (seeded "
                 "via `timvx.output_scale`/`zp` arg-attrs or the (1.0, 0) "
                 "anchor) or a previously-handled rescale's result; "
                 "this op's input is neither";
          *strictFailed = true;
          return;
        }
        QuantInfo si = *siOpt;
        double So;
        int64_t Zo = *outZp;
        if (auto preset = qm.lookup(resc.getResult())) {
          // Path (1): pre-seeded by seedFuncResults from func res_attrs.
          So = preset->scale;
          Zo = preset->zp;
        } else if (auto cq = tryRecoverChainQuant(resc.getResult())) {
          // Path (2): downstream dequant chain names So.
          So = cq->first;
          Zo = cq->second;
        } else {
          // Path (3): Sw=1/128 convention. See block comment above.
          double Sw = 1.0 / 128.0;
          So = si.scale * Sw / M;
        }
        qm.set(resc.getResult(), {So, Zo});
      } else {
        // Standalone rescale (e.g. tflite emits an i8->i8 requantize after
        // pad to align scales between conv-out and pool-in). Math in real-
        // value space: out_real = in_real, so M = Si / So, hence So = Si/M.
        auto siOpt = qm.lookup(resc.getInput());
        if (!siOpt) {
          resc->emitError()
              << "buildQuantInfoMap: standalone rescale input has no "
                 "tracked quant info — its producer must be a func arg "
                 "or a previously-handled rescale";
          *strictFailed = true;
          return;
        }
        QuantInfo si = *siOpt;
        double So = (M != 0.0) ? (si.scale / M) : si.scale;
        qm.set(resc.getResult(), {So, *outZp});
      }
      return;
    }
    if (isa<tosa::MaxPool2dOp, tosa::AvgPool2dOp, tosa::PadOp,
            tosa::TransposeOp, tosa::ReshapeOp, tosa::SliceOp>(op)) {
      if (op->getNumOperands() < 1 || op->getNumResults() != 1) return;
      auto qi = qm.lookup(op->getOperand(0));
      if (qi) qm.set(op->getResult(0), *qi);
      return;
    }
    // tosa.cast: drops quant in general. Exception: the *final* requantize
    // cast — `tosa.cast f32 -> narrow-int` produced by `RequantI32SkipFold`
    // — is tagged with `timvx.output_scale` / `timvx.output_zp` discardable
    // attrs. Read those here so `CastConversion` can attach the right
    // Quantization() to the downstream tensor spec.
    if (auto cast = dyn_cast<tosa::CastOp>(op)) {
      auto scaleAttr = cast->getAttrOfType<FloatAttr>("timvx.output_scale");
      auto zpAttr = cast->getAttrOfType<IntegerAttr>("timvx.output_zp");
      if (scaleAttr && zpAttr)
        qm.set(cast.getResult(),
               {scaleAttr.getValueAsDouble(), zpAttr.getInt()});
      return;
    }
  });
}

// Recover (scale, zp) from a producer op via its standardized
// `output_scale` / `output_zp` discardable attrs (every quantized timvx op
// carries them). Used by `QuantResidualFuse` / `matchDequantChain`.
//
// Function-arg case: a tensor that's a `BlockArgument` of the enclosing
// `func.func` has no defining op, so the standard producer-quant lookup
// would fail. The quant info instead lives on the func's `arg_attrs` —
// `output_scale` / `output_zp` keyed by the arg index.
inline std::optional<std::pair<double, int64_t>>
getProducerQuant(Value v) {
  if (auto bArg = dyn_cast<BlockArgument>(v)) {
    auto func = dyn_cast_or_null<func::FuncOp>(
        bArg.getOwner()->getParentOp());
    if (!func) return std::nullopt;
    unsigned idx = bArg.getArgNumber();
    auto sAttr = func.getArgAttrOfType<FloatAttr>(idx, "timvx.output_scale");
    auto zAttr = func.getArgAttrOfType<IntegerAttr>(idx, "timvx.output_zp");
    if (!sAttr || !zAttr) return std::nullopt;
    return std::make_pair(sAttr.getValueAsDouble(), zAttr.getInt());
  }
  auto *def = v.getDefiningOp();
  if (!def) return std::nullopt;
  auto sAttr = def->getAttrOfType<FloatAttr>("output_scale");
  auto zAttr = def->getAttrOfType<IntegerAttr>("output_zp");
  if (!sAttr || !zAttr) return std::nullopt;
  return std::make_pair(sAttr.getValueAsDouble(), zAttr.getInt());
}

} // namespace detail
} // namespace timvx
} // namespace mlir

#endif // TIMVX_PASSES_COMMON_H
