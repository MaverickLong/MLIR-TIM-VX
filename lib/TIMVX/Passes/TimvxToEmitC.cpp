//===- TimvxToEmitC.cpp - timvx -> emitc lowering ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Per-op lowerings emit one `emitc.call_opaque` against a runtime helper
// in `timvx_runtime::`. The helper header is supplied out-of-band;
// signatures follow the pattern
//
//   TensorPtr <op>(GraphPtr graph,
//                  <ssa input tensors...>,
//                  <constant attrs...>,
//                  tim::vx::TensorSpec output_spec);
//
// where `TensorPtr = std::shared_ptr<tim::vx::Tensor>` and
//       `GraphPtr  = std::shared_ptr<tim::vx::Graph>`.
//
// The graph itself is referenced as a free identifier `graph`; the
// surrounding function is expected to declare it (typically as the first
// parameter — see `prependGraphParam`). Output spec / per-op constants
// are serialized as opaque C++ text and inlined at call sites. Constant
// data for `timvx.const` is reified as a `static const T name[N] = {…};`
// declaration emitted before the call.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace timvx {

#define GEN_PASS_DEF_TIMVXTOEMITCPASS
#include "TIMVX/TIMVXPasses.h.inc"

namespace {

constexpr StringRef kTensorCxx = "std::shared_ptr<tim::vx::Tensor>";
constexpr StringRef kShapeCxx = "std::vector<uint32_t>";
constexpr StringRef kGraphCxx = "std::shared_ptr<tim::vx::Graph>";
constexpr StringRef kRuntimeNs = "timvx_runtime::";

emitc::OpaqueType tensorOpaqueTy(MLIRContext *c) {
  return emitc::OpaqueType::get(c, kTensorCxx);
}
emitc::OpaqueType shapeOpaqueTy(MLIRContext *c) {
  return emitc::OpaqueType::get(c, kShapeCxx);
}
emitc::OpaqueType graphOpaqueTy(MLIRContext *c) {
  return emitc::OpaqueType::get(c, kGraphCxx);
}

// Decide whether an asymmetric int8 tensor should be emitted as u8 with
// zp shifted +128. On VIP9000Nano-DI the runtime auto-promotes every
// asym int8 IO tensor to u8|asym(zp+128) and inserts a DataConvert at
// the boundary; that auto-DataConvert COMPILE_FAILs at graph IO for
// non-trivial shapes (the `vivante.nn.tensorcopy` kernel rejects e.g.
// rank-2 1×1000 outputs, even though same-shape `u8 <-> i8` passes the
// per-pair probe). Pre-emitting all asym int8 specs as u8 matches the
// form TIM-VX wants internally — internally consistent throughout, so
// the auto-rewriter sees no candidates and skips inserting the boundary
// DataConvert. Constants need their bytes XOR'd with 0x80 to compensate
// for the storage relabel.
bool shouldPromoteI8AsymToU8(Type elem, FloatAttr scaleOverride,
                              IntegerAttr zpOverride) {
  if (auto i = dyn_cast<IntegerType>(elem))
    if (i.getWidth() == 8 && i.isSignless())
      return scaleOverride && zpOverride;
  if (auto quni = dyn_cast<quant::UniformQuantizedType>(elem))
    if (auto sty = dyn_cast<IntegerType>(quni.getStorageType()))
      return sty.getWidth() == 8 && quni.isSigned();
  return false;
}

// Render an MLIR element type as the matching tim::vx::DataType enum literal.
std::string tvxDataType(Type t) {
  if (auto qt = dyn_cast<quant::QuantizedType>(t))
    return tvxDataType(qt.getStorageType());
  if (t.isF32())       return "tim::vx::DataType::FLOAT32";
  if (t.isF16())       return "tim::vx::DataType::FLOAT16";
  if (t.isInteger(1))  return "tim::vx::DataType::BOOL8";
  if (auto it = dyn_cast<IntegerType>(t)) {
    bool u = it.isUnsigned();
    switch (it.getWidth()) {
    case 8:  return u ? "tim::vx::DataType::UINT8"  : "tim::vx::DataType::INT8";
    case 16: return u ? "tim::vx::DataType::UINT16" : "tim::vx::DataType::INT16";
    case 32: return u ? "tim::vx::DataType::UINT32" : "tim::vx::DataType::INT32";
    case 64: return "tim::vx::DataType::INT64";
    }
  }
  return "tim::vx::DataType::UNKNOWN";
}

template <typename Range> std::string fmtBraceList(Range &&r) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "{";
  llvm::interleaveComma(r, os);
  os << "}";
  return os.str();
}

std::string fmtArray(ArrayRef<int64_t> v) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "std::array<uint32_t, " << v.size() << ">"
     << fmtBraceList(llvm::map_range(v, [](int64_t x) { return uint64_t(x); }));
  return os.str();
}

bool isFuncReturnedOp(Operation *op) {
  for (Value r : op->getResults())
    for (Operation *u : r.getUsers())
      if (isa<func::ReturnOp>(u))
        return true;
  return false;
}

// Map an MLIR scalar element type to its C++ name (used for the static
// array decl that backs each timvx.const).
StringRef cxxScalarType(Type t) {
  if (t.isF32())       return "float";
  if (t.isF64())       return "double";
  if (t.isF16())       return "uint16_t"; // raw FP16 storage
  if (t.isInteger(1))  return "uint8_t";
  if (auto it = dyn_cast<IntegerType>(t)) {
    bool u = it.isUnsigned();
    switch (it.getWidth()) {
    case 8:  return u ? "uint8_t"  : "int8_t";
    case 16: return u ? "uint16_t" : "int16_t";
    case 32: return u ? "uint32_t" : "int32_t";
    case 64: return "int64_t";
    }
  }
  return "char";
}

// Compute a permutation `m` such that `m[tvx_off] = mlir_off` —
// i.e. the index of the MLIR-row-major value that lands at TIM-VX
// innermost-first byte position `tvx_off`. For rank<=1 or palindromic
// shapes this is the identity.
inline SmallVector<size_t> tvxToMlirIndexMap(ArrayRef<int64_t> shape) {
  size_t rank = shape.size();
  size_t numel = 1;
  for (auto d : shape) numel *= d;
  SmallVector<size_t> m(numel);
  if (rank <= 1) {
    for (size_t e = 0; e < numel; ++e) m[e] = e;
    return m;
  }
  SmallVector<size_t> mlir_strides(rank), tvx_strides(rank);
  mlir_strides[rank - 1] = 1;
  for (int k = static_cast<int>(rank) - 2; k >= 0; --k)
    mlir_strides[k] = mlir_strides[k + 1] * shape[k + 1];
  tvx_strides[0] = 1;
  for (size_t k = 1; k < rank; ++k)
    tvx_strides[k] = tvx_strides[k - 1] * shape[k - 1];
  SmallVector<size_t> idx(rank, 0);
  for (size_t e = 0; e < numel; ++e) {
    size_t mlir_off = 0, tvx_off = 0;
    for (size_t k = 0; k < rank; ++k) {
      mlir_off += idx[k] * mlir_strides[k];
      tvx_off  += idx[k] * tvx_strides[k];
    }
    m[tvx_off] = mlir_off;
    for (int k = static_cast<int>(rank) - 1; k >= 0; --k) {
      if (++idx[k] < static_cast<size_t>(shape[k])) break;
      idx[k] = 0;
    }
  }
  return m;
}

// Reorder a flat array from MLIR row-major to TIM-VX innermost-first.
// Built via push_back so it works for non-default-constructible element
// types like APFloat (which only has a copy ctor, not a default one).
//
// Done once at lowering so the emitted `static const T[]` arrays bind
// 1:1 to the TIM-VX tensor — no runtime layout fixup.
template <typename T>
SmallVector<T> reorderMlirToTvx(ArrayRef<T> values, ArrayRef<int64_t> shape) {
  auto m = tvxToMlirIndexMap(shape);
  SmallVector<T> out;
  out.reserve(values.size());
  for (size_t i = 0; i < values.size(); ++i)
    out.push_back(values[m[i]]);
  return out;
}

template <typename T>
void writePodArray(llvm::raw_string_ostream &os, ArrayRef<T> values,
                   StringRef floatSuffix) {
  bool first = true;
  if constexpr (std::is_floating_point_v<T>) {
    char buf[32];
    for (T v : values) {
      os << (first ? "" : ", ");
      first = false;
      std::snprintf(buf, sizeof(buf), "%a", static_cast<double>(v));
      os << buf << floatSuffix;
    }
  } else {
    for (T v : values) {
      os << (first ? "" : ", ");
      first = false;
      os << static_cast<int64_t>(v);
    }
  }
}

// Render an ElementsAttr as `static const T name[N] = { v0, v1, ... };`.
// Returns "" if the storage or element type isn't one we handle.
//
// When `promoteI8ToU8` is true, the array is emitted as `uint8_t[]` with
// each byte XOR'd 0x80, so reading the bytes as u8 with zp shifted +128
// gives the same real values as reading the original bytes as i8 with
// the original zp.
std::string fmtStaticArrayDecl(ElementsAttr values, StringRef name,
                                bool promoteI8ToU8 = false) {
  auto rt = cast<RankedTensorType>(values.getType());
  Type elem = rt.getElementType();
  ArrayRef<int64_t> shape = rt.getShape();
  uint64_t numel = std::max<uint64_t>(rt.getNumElements(), 1);

  std::string out;
  llvm::raw_string_ostream os(out);

  if (promoteI8ToU8) {
    // Gather as int8 (MLIR row-major), reorder to TIM-VX innermost-first,
    // then XOR 0x80 to land on UINT8 semantics with zp+128.
    SmallVector<int8_t> mlir_bytes;
    mlir_bytes.reserve(numel);
    if (auto ints = values.tryGetValues<APInt>()) {
      for (APInt v : *ints)
        mlir_bytes.push_back(static_cast<int8_t>(v.getZExtValue()));
    } else if (auto r = dyn_cast<DenseI8ResourceElementsAttr>(values)) {
      auto data = r.tryGetAsArrayRef();
      if (!data) return std::string();
      mlir_bytes.assign(data->begin(), data->end());
    } else {
      return std::string();
    }
    auto reordered = reorderMlirToTvx<int8_t>(mlir_bytes, shape);
    os << "static const uint8_t " << name << "[" << numel << "] = {";
    bool first = true;
    for (int8_t v : reordered) {
      os << (first ? "" : ", ")
         << static_cast<int>(static_cast<uint8_t>(v) ^ 0x80);
      first = false;
    }
    os << "};";
    return out;
  }

  os << "static const " << cxxScalarType(elem) << " " << name << "[" << numel
     << "] = {";

  StringRef floatSuffix = elem.isF32() ? "f" : "";

  if (auto floats = values.tryGetValues<APFloat>()) {
    SmallVector<APFloat> mlir_vals;
    mlir_vals.reserve(numel);
    for (APFloat v : *floats) mlir_vals.push_back(v);
    auto reordered = reorderMlirToTvx<APFloat>(mlir_vals, shape);
    char buf[32];
    bool first = true;
    for (APFloat v : reordered) {
      os << (first ? "" : ", ");
      first = false;
      std::snprintf(buf, sizeof(buf), "%a", v.convertToDouble());
      os << buf << floatSuffix;
    }
  } else if (auto ints = values.tryGetValues<APInt>()) {
    SmallVector<APInt> mlir_vals;
    mlir_vals.reserve(numel);
    for (APInt v : *ints) mlir_vals.push_back(v);
    auto reordered = reorderMlirToTvx<APInt>(mlir_vals, shape);
    SmallString<32> s;
    bool first = true;
    bool isSigned = !cxxScalarType(elem).starts_with("u");
    for (APInt v : reordered) {
      os << (first ? "" : ", ");
      first = false;
      s.clear();
      v.toString(s, /*radix=*/10, isSigned);
      os << s;
    }
  } else if (auto r = dyn_cast<DenseF32ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return std::string();
    auto reordered = reorderMlirToTvx<float>(*data, shape);
    writePodArray<float>(os, reordered, floatSuffix);
  } else if (auto r = dyn_cast<DenseF64ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return std::string();
    auto reordered = reorderMlirToTvx<double>(*data, shape);
    writePodArray<double>(os, reordered, "");
  } else if (auto r = dyn_cast<DenseI8ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return std::string();
    auto reordered = reorderMlirToTvx<int8_t>(*data, shape);
    writePodArray<int8_t>(os, reordered, "");
  } else if (auto r = dyn_cast<DenseI16ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return std::string();
    auto reordered = reorderMlirToTvx<int16_t>(*data, shape);
    writePodArray<int16_t>(os, reordered, "");
  } else if (auto r = dyn_cast<DenseI32ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return std::string();
    auto reordered = reorderMlirToTvx<int32_t>(*data, shape);
    writePodArray<int32_t>(os, reordered, "");
  } else if (auto r = dyn_cast<DenseI64ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return std::string();
    auto reordered = reorderMlirToTvx<int64_t>(*data, shape);
    writePodArray<int64_t>(os, reordered, "");
  } else if (auto r = dyn_cast<DenseUI8ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return std::string();
    auto reordered = reorderMlirToTvx<uint8_t>(*data, shape);
    writePodArray<uint8_t>(os, reordered, "");
  } else if (auto r = dyn_cast<DenseUI16ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return std::string();
    auto reordered = reorderMlirToTvx<uint16_t>(*data, shape);
    writePodArray<uint16_t>(os, reordered, "");
  } else if (auto r = dyn_cast<DenseUI32ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return std::string();
    auto reordered = reorderMlirToTvx<uint32_t>(*data, shape);
    writePodArray<uint32_t>(os, reordered, "");
  } else if (auto r = dyn_cast<DenseUI64ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return std::string();
    auto reordered = reorderMlirToTvx<uint64_t>(*data, shape);
    writePodArray<uint64_t>(os, reordered, "");
  } else {
    return std::string();
  }

  os << "};";
  return out;
}

// Format a double as a float literal that's always parseable by C++:
// `%.10g` alone can drop the decimal point (e.g. `1` for 1.0), which
// chains with the `f` suffix to an invalid token `1f`.
std::string fmtFloatLiteral(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.10g", v);
  std::string s(buf);
  bool hasDot = s.find_first_of(".eEpP") != std::string::npos;
  if (!hasDot) s += ".0";
  return s + "f";
}

// Format a tensor's MLIR type as a `tim::vx::TensorSpec(...)` constructor
// expression. `tensorAttr` defaults to TRANSIENT (intermediate); callers
// can override (e.g. for constants).
//
// `scaleOverride` / `zpOverride`: when both set, the emitted spec carries
// `Quantization(ASYMMETRIC, scale, zp)`. Otherwise we fall back to the
// element type — `!quant.uniform<...>` types still emit Quantization()
// automatically. Plain integer/float types emit no quantization arg.
std::string fmtTensorSpec(Type ty, StringRef tensorAttr = "TRANSIENT",
                           FloatAttr scaleOverride = {},
                           IntegerAttr zpOverride = {}) {
  auto rt = cast<RankedTensorType>(ty);
  Type elem = rt.getElementType();
  std::string s;
  llvm::raw_string_ostream os(s);

  if (shouldPromoteI8AsymToU8(elem, scaleOverride, zpOverride)) {
    double scale;
    int64_t zp;
    if (scaleOverride && zpOverride) {
      scale = scaleOverride.getValueAsDouble();
      zp = zpOverride.getInt();
    } else {
      auto quni = cast<quant::UniformQuantizedType>(elem);
      scale = quni.getScale();
      zp = quni.getZeroPoint();
    }
    os << "tim::vx::TensorSpec(tim::vx::DataType::UINT8, " << kShapeCxx
       << fmtBraceList(rt.getShape())
       << ", tim::vx::TensorAttribute::" << tensorAttr
       << ", tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC, "
       << fmtFloatLiteral(scale) << ", " << (zp + 128) << "))";
    return os.str();
  }

  os << "tim::vx::TensorSpec(" << tvxDataType(elem) << ", "
     << kShapeCxx << fmtBraceList(rt.getShape())
     << ", tim::vx::TensorAttribute::" << tensorAttr;

  if (scaleOverride && zpOverride) {
    os << ", tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC, "
       << fmtFloatLiteral(scaleOverride.getValueAsDouble()) << ", "
       << zpOverride.getInt() << ")";
  } else if (auto quni = dyn_cast<quant::UniformQuantizedType>(elem)) {
    os << ", tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC, "
       << fmtFloatLiteral(quni.getScale()) << ", "
       << quni.getZeroPoint() << ")";
  } else if (auto qpa = dyn_cast<quant::UniformQuantizedPerAxisType>(elem)) {
    os << ", tim::vx::Quantization(tim::vx::QuantType::"
       << (llvm::all_of(qpa.getZeroPoints(), [](int64_t z) { return z == 0; })
               ? "SYMMETRIC_PER_CHANNEL"
               : "ASYMMETRIC_PER_CHANNEL")
       << ", " << qpa.getQuantizedDimension() << ", std::vector<float>{";
    llvm::interleaveComma(qpa.getScales(), os, [&](double s) {
      os << fmtFloatLiteral(s);
    });
    os << "}, std::vector<int32_t>{";
    llvm::interleaveComma(qpa.getZeroPoints(), os);
    os << "})";
  }

  os << ")";
  return os.str();
}

emitc::OpaqueAttr opq(MLIRContext *c, StringRef s) {
  return emitc::OpaqueAttr::get(c, s);
}

// Look up the `graph` SSA value — by convention the first argument of the
// enclosing func.func, prepended in the pre-pass below.
Value getGraphArg(Operation *op) {
  auto func = op->getParentOfType<func::FuncOp>();
  if (!func || func.getNumArguments() == 0)
    return {};
  Value first = func.getArgument(0);
  auto opaque = dyn_cast<emitc::OpaqueType>(first.getType());
  if (!opaque || opaque.getValue() != kGraphCxx)
    return {};
  return first;
}

LogicalResult emitRuntimeCall(ConversionPatternRewriter &rewriter,
                              Operation *op, StringRef helperName,
                              ValueRange operands,
                              ArrayRef<Attribute> trailing,
                              Type resultType) {
  Value graph = getGraphArg(op);
  if (!graph)
    return rewriter.notifyMatchFailure(
        op, "enclosing func has no graph argument; pre-pass missed it");

  SmallVector<Value, 8> allOperands{graph};
  allOperands.append(operands.begin(), operands.end());

  SmallVector<Attribute, 8> args;
  for (size_t i = 0; i < allOperands.size(); ++i)
    args.push_back(rewriter.getIndexAttr(i));
  args.append(trailing.begin(), trailing.end());

  std::string callee = (kRuntimeNs + helperName).str();
  rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
      op, TypeRange{resultType}, callee, allOperands,
      rewriter.getArrayAttr(args), ArrayAttr{});
  return success();
}

//===----------------------------------------------------------------------===//
// Type converter
//===----------------------------------------------------------------------===//
//
// Maps:
//   tensor<Nxindex>      -> opaque<"std::vector<uint32_t>">  (shape vectors)
//   tensor<...>          -> opaque<"std::shared_ptr<tim::vx::Tensor>">
//   !timvx.graph         -> opaque<"std::shared_ptr<tim::vx::Graph>">
class TIMVXToEmitCTypeConverter : public TypeConverter {
public:
  TIMVXToEmitCTypeConverter(MLIRContext *ctx) {
    addConversion([](Type t) { return t; }); // identity fallback
    addConversion([ctx](RankedTensorType t) -> Type {
      if (isa<IndexType>(t.getElementType()))
        return shapeOpaqueTy(ctx);
      return tensorOpaqueTy(ctx);
    });
    addConversion(
        [ctx](GraphType) -> Type { return graphOpaqueTy(ctx); });

    auto addCast = [](OpBuilder &b, Type t, ValueRange vs, Location loc) {
      return UnrealizedConversionCastOp::create(b, loc, t, vs).getResult(0);
    };
    addSourceMaterialization(addCast);
    addTargetMaterialization(addCast);
  }
};

//===----------------------------------------------------------------------===//
// Per-op patterns
//===----------------------------------------------------------------------===//

// timvx.const_shape -> emitc.constant {value = #emitc.opaque<"{...}">}
// We don't go through a runtime helper — a brace-init literal is enough.
struct ConstShapeToEmitC : public OpConversionPattern<ConstShapeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ConstShapeOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto attr = cast<DenseIntElementsAttr>(op.getValuesAttr());
    SmallVector<int64_t> vals;
    for (APInt v : attr.getValues<APInt>())
      vals.push_back(static_cast<int64_t>(v.getZExtValue()));
    std::string lit = (kShapeCxx + fmtBraceList(vals)).str();
    rewriter.replaceOpWithNewOp<emitc::ConstantOp>(
        op, shapeOpaqueTy(rewriter.getContext()),
        opq(rewriter.getContext(), lit));
    return success();
  }
};

// timvx.const -> timvx_runtime::const_tensor(graph, spec, &<data>[0]).
//
// The const op's data is reified into a function-local
// `static const T <name>[N] = { … };` declaration emitted before the
// call, and the call passes the array name (which decays to a const T*).
// Weights are baked into the compiled binary — no separate weights file.
struct ConstToEmitC : public OpConversionPattern<ConstOp> {
  unsigned *counter;
  ConstToEmitC(const TypeConverter &tc, MLIRContext *ctx, unsigned *c)
      : OpConversionPattern<ConstOp>(tc, ctx), counter(c) {}
  LogicalResult
  matchAndRewrite(ConstOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto values = cast<ElementsAttr>(op.getValuesAttr());

    // i8|asym consts get bit-flipped (XOR 0x80) and emitted as uint8_t,
    // matching the u8|asym(zp+128) spec the matching fmtTensorSpec path
    // emits. See shouldPromoteI8AsymToU8 for the rationale.
    bool promote = shouldPromoteI8AsymToU8(
        cast<RankedTensorType>(op.getType()).getElementType(),
        op.getQuantScaleAttr(), op.getQuantZpAttr());

    std::string name = "_timvx_const_" + std::to_string((*counter)++);
    std::string decl = fmtStaticArrayDecl(values, name, promote);
    if (decl.empty())
      return rewriter.notifyMatchFailure(
          op, "unsupported element type / storage for constant data");

    emitc::VerbatimOp::create(rewriter, op.getLoc(),
                              rewriter.getStringAttr(decl));

    MLIRContext *ctx = rewriter.getContext();
    SmallVector<Attribute, 2> trailing{
        opq(ctx, fmtTensorSpec(op.getType(), "CONSTANT",
                                op.getQuantScaleAttr(),
                                op.getQuantZpAttr())),
        opq(ctx, name),
    };
    return emitRuntimeCall(rewriter, op, "const_tensor", /*operands=*/{},
                           trailing, tensorOpaqueTy(ctx));
  }
};

// Helper: pick OUTPUT vs TRANSIENT for the op's output_spec text.
inline StringRef outAttr(Operation *op) {
  return isFuncReturnedOp(op) ? "OUTPUT" : "TRANSIENT";
}

// Helper for ops whose only trailing arg is the output TensorSpec. Picks
// up optional `output_scale` / `output_zp` discardable attrs by name —
// that's how `timvx-quant-residual-fuse` parks the (Sout, Zout) it needs
// to land on the new quant `timvx.add` (and friends).
template <typename TIMVXOp>
struct SimpleRuntimeCall : public OpConversionPattern<TIMVXOp> {
  using OpConversionPattern<TIMVXOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<TIMVXOp>::OpAdaptor;
  StringRef helperName;
  SimpleRuntimeCall(const TypeConverter &tc, MLIRContext *ctx, StringRef name)
      : OpConversionPattern<TIMVXOp>(tc, ctx), helperName(name) {}
  LogicalResult
  matchAndRewrite(TIMVXOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *c = rewriter.getContext();
    auto sAttr = op->template getAttrOfType<FloatAttr>("output_scale");
    auto zAttr = op->template getAttrOfType<IntegerAttr>("output_zp");
    SmallVector<Attribute, 1> trailing{
        opq(c, fmtTensorSpec(op.getType(), outAttr(op), sAttr, zAttr))};
    return emitRuntimeCall(rewriter, op, helperName, adaptor.getOperands(),
                           trailing, tensorOpaqueTy(c));
  }
};

struct ClipToEmitC : public OpConversionPattern<ClipOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ClipOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *c = rewriter.getContext();
    auto fmtFloat = [](float v) {
      std::string s;
      llvm::raw_string_ostream(s) << v << "f";
      return s;
    };
    auto sAttr = op->getAttrOfType<FloatAttr>("output_scale");
    auto zAttr = op->getAttrOfType<IntegerAttr>("output_zp");
    SmallVector<Attribute, 3> trailing{
        opq(c, fmtFloat(op.getMinVal().convertToFloat())),
        opq(c, fmtFloat(op.getMaxVal().convertToFloat())),
        opq(c, fmtTensorSpec(op.getType(), outAttr(op), sAttr, zAttr)),
    };
    return emitRuntimeCall(rewriter, op, "clip",
                           ValueRange{adaptor.getInput()}, trailing,
                           tensorOpaqueTy(c));
  }
};

struct Conv2DToEmitC : public OpConversionPattern<Conv2DOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(Conv2DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *c = rewriter.getContext();
    SmallVector<Attribute, 4> trailing{
        opq(c, fmtArray(op.getPad())),
        opq(c, fmtArray(op.getStride())),
        opq(c, fmtArray(op.getDilation())),
        opq(c, fmtTensorSpec(op.getType(), outAttr(op),
                              op.getOutputScaleAttr(),
                              op.getOutputZpAttr())),
    };
    return emitRuntimeCall(rewriter, op, "conv2d",
                           ValueRange{adaptor.getInput(), adaptor.getWeight(),
                                      adaptor.getBias()},
                           trailing, tensorOpaqueTy(c));
  }
};

struct Pool2DToEmitC : public OpConversionPattern<Pool2DOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(Pool2DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *c = rewriter.getContext();
    std::string poolEnum =
        ("tim::vx::PoolType::" + stringifyPoolType(op.getPoolType())).str();
    SmallVector<Attribute, 5> trailing{
        opq(c, poolEnum),
        opq(c, fmtArray(op.getKernel())),
        opq(c, fmtArray(op.getStride())),
        opq(c, fmtArray(op.getPad())),
        opq(c, fmtTensorSpec(op.getType(), outAttr(op),
                              op.getOutputScaleAttr(),
                              op.getOutputZpAttr())),
    };
    return emitRuntimeCall(rewriter, op, "pool2d",
                           ValueRange{adaptor.getInput()}, trailing,
                           tensorOpaqueTy(c));
  }
};

struct ReshapeToEmitC : public OpConversionPattern<ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *c = rewriter.getContext();
    SmallVector<Attribute, 1> trailing{
        opq(c, fmtTensorSpec(op.getType(), outAttr(op),
                              op.getOutputScaleAttr(),
                              op.getOutputZpAttr())),
    };
    return emitRuntimeCall(rewriter, op, "reshape",
                           ValueRange{adaptor.getInput1(), adaptor.getShape()},
                           trailing, tensorOpaqueTy(c));
  }
};

struct SliceToEmitC : public OpConversionPattern<SliceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *c = rewriter.getContext();
    SmallVector<Attribute, 1> trailing{
        opq(c, fmtTensorSpec(op.getType(), outAttr(op),
                              op.getOutputScaleAttr(),
                              op.getOutputZpAttr())),
    };
    return emitRuntimeCall(rewriter, op, "slice",
                           ValueRange{adaptor.getInput1(),
                                      adaptor.getStart(), adaptor.getSize()},
                           trailing, tensorOpaqueTy(c));
  }
};

struct TransposeToEmitC : public OpConversionPattern<TransposeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *c = rewriter.getContext();
    SmallVector<int64_t> perms(op.getPerms().begin(), op.getPerms().end());
    SmallVector<Attribute, 2> trailing{
        opq(c, kShapeCxx.str() + fmtBraceList(perms)),
        opq(c, fmtTensorSpec(op.getType(), outAttr(op),
                              op.getOutputScaleAttr(),
                              op.getOutputZpAttr())),
    };
    return emitRuntimeCall(rewriter, op, "transpose",
                           ValueRange{adaptor.getInput1()}, trailing,
                           tensorOpaqueTy(c));
  }
};

struct ReduceSumToEmitC : public OpConversionPattern<ReduceSumOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReduceSumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *c = rewriter.getContext();
    SmallVector<int64_t> axes(op.getAxes().begin(), op.getAxes().end());
    SmallVector<Attribute, 3> trailing{
        opq(c, "std::vector<int32_t>" + fmtBraceList(axes)),
        opq(c, op.getKeepDims() ? "true" : "false"),
        opq(c, fmtTensorSpec(op.getType(), outAttr(op))),
    };
    return emitRuntimeCall(rewriter, op, "reduce_sum",
                           ValueRange{adaptor.getInput()}, trailing,
                           tensorOpaqueTy(c));
  }
};

struct FullyConnectedToEmitC : public OpConversionPattern<FullyConnectedOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(FullyConnectedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *c = rewriter.getContext();
    SmallVector<Attribute, 1> trailing{
        opq(c, fmtTensorSpec(op.getType(), outAttr(op),
                              op.getOutputScaleAttr(),
                              op.getOutputZpAttr())),
    };
    return emitRuntimeCall(rewriter, op, "fully_connected",
                           ValueRange{adaptor.getInput(), adaptor.getWeight(),
                                      adaptor.getBias()},
                           trailing, tensorOpaqueTy(c));
  }
};

struct CastToEmitC : public OpConversionPattern<CastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *c = rewriter.getContext();
    SmallVector<Attribute, 1> trailing{
        opq(c, fmtTensorSpec(op.getType(), outAttr(op),
                              op.getOutputScaleAttr(),
                              op.getOutputZpAttr())),
    };
    return emitRuntimeCall(rewriter, op, "cast",
                           ValueRange{adaptor.getInput()}, trailing,
                           tensorOpaqueTy(c));
  }
};

struct DataConvertToEmitC : public OpConversionPattern<DataConvertOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(DataConvertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *c = rewriter.getContext();
    SmallVector<Attribute, 1> trailing{
        opq(c, fmtTensorSpec(op.getType(), outAttr(op),
                              op.getOutputScaleAttr(),
                              op.getOutputZpAttr())),
    };
    return emitRuntimeCall(rewriter, op, "dataconvert",
                           ValueRange{adaptor.getInput()}, trailing,
                           tensorOpaqueTy(c));
  }
};

struct PadToEmitC : public OpConversionPattern<PadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *c = rewriter.getContext();
    auto fmtFloat = [](float v) {
      std::string s;
      llvm::raw_string_ostream(s) << v << "f";
      return s;
    };
    SmallVector<Attribute, 2> trailing{
        opq(c, fmtFloat(op.getPadConst().convertToFloat())),
        opq(c, fmtTensorSpec(op.getType(), outAttr(op),
                              op.getOutputScaleAttr(),
                              op.getOutputZpAttr())),
    };
    return emitRuntimeCall(rewriter, op, "pad",
                           ValueRange{adaptor.getInput(), adaptor.getPadding()},
                           trailing, tensorOpaqueTy(c));
  }
};

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

// Pre-pass: prepend a `graph` argument to every func.func, and rename any
// function called `main` (would clash with C++'s `int main()` once the
// emitter inlines it into the runner translation unit).
//
// Has to run as plain IR rewriting before applyPartialConversion, because
// the per-op patterns reference the parent func's argument-0 to source
// the graph SSA value.
void prependGraphParam(func::FuncOp func) {
  if (func.isExternal())
    return;
  MLIRContext *ctx = func.getContext();
  auto graphTy = emitc::OpaqueType::get(ctx, kGraphCxx);

  SmallVector<Type, 8> inputs{graphTy};
  for (Type t : func.getFunctionType().getInputs())
    inputs.push_back(t);
  func.setType(FunctionType::get(ctx, inputs,
                                 func.getFunctionType().getResults()));
  func.getBody().front().insertArgument(/*index=*/0u, graphTy, func.getLoc());

  if (func.getName() == "main")
    func.setName("timvx_main");
}

struct TIMVXToEmitCPass
    : public impl::TIMVXToEmitCPassBase<TIMVXToEmitCPass> {
  void runOnOperation() final {
    MLIRContext *ctx = &getContext();

    for (auto func : llvm::to_vector(getOperation().getOps<func::FuncOp>()))
      prependGraphParam(func);

    TIMVXToEmitCTypeConverter converter(ctx);

    ConversionTarget target(*ctx);
    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addIllegalDialect<TIMVXDialect>();

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp f) {
      return converter.isSignatureLegal(f.getFunctionType()) &&
             converter.isLegal(&f.getBody());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp r) { return converter.isLegal(r.getOperandTypes()); });
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp c) { return converter.isLegal(c); });

    RewritePatternSet patterns(ctx);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    populateReturnOpTypeConversionPattern(patterns, converter);
    populateCallOpTypeConversionPattern(patterns, converter);

    unsigned constCounter = 0;
    patterns.add<ConstToEmitC>(converter, ctx, &constCounter);

    patterns.add<ConstShapeToEmitC, ClipToEmitC, Conv2DToEmitC,
                 Pool2DToEmitC, TransposeToEmitC, CastToEmitC,
                 DataConvertToEmitC, PadToEmitC, FullyConnectedToEmitC,
                 ReduceSumToEmitC, ReshapeToEmitC, SliceToEmitC>(converter,
                                                                  ctx);

    auto addSimple = [&](StringRef name, auto opTag) {
      using T = decltype(opTag);
      patterns.add<SimpleRuntimeCall<T>>(converter, ctx, name);
    };
    addSimple("matmul", MatMulOp{});
    addSimple("multiply", MultiplyOp{});
    addSimple("add", AddOp{});
    addSimple("sub", SubOp{});
    addSimple("pow_op", PowOp{});
    addSimple("rcp", RcpOp{});
    addSimple("maximum", MaximumOp{});
    addSimple("minimum", MinimumOp{});

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createTIMVXToEmitCPass() {
  return std::make_unique<TIMVXToEmitCPass>();
}

} // namespace timvx
} // namespace mlir
