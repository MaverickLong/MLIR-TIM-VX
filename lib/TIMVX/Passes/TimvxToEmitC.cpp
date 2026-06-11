//===- TimvxToEmitC.cpp - timvx -> emitc lowering ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declaration: this file is mostly a directly passthrough of timvx dialect
// so is largely written by AI.
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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>

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

// Note: every i8↔u8 promotion concern is handled exclusively by
// `--timvx-promote-i8-to-u8`, which runs immediately after
// `--tosa-to-timvx`. By the time this file is reached, every quantized
// tensor is already u8 (storage type, zp, byte values, and the ±128
// compensation around f32↔u8 casts), so the EmitC pass just emits the
// types verbatim — no promotion logic, no override branches.

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
// Returns "" if the storage or element type isn't one we handle. The
// `--timvx-promote-i8-to-u8` pass has already rewritten any i8 const to
// its u8-byte-flipped equivalent, so this function never sees signed-i8
// quantized constants.
std::string fmtStaticArrayDecl(ElementsAttr values, StringRef name) {
  auto rt = cast<RankedTensorType>(values.getType());
  Type elem = rt.getElementType();
  ArrayRef<int64_t> shape = rt.getShape();
  uint64_t numel = std::max<uint64_t>(rt.getNumElements(), 1);

  std::string out;
  llvm::raw_string_ostream os(out);

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

// Pack a value range into raw machine-byte storage in TIM-VX innermost-
// first order. The output is a plain memcpy of the reordered scalar
// elements — exactly what `graph->CreateTensor(spec, data)` consumes.
template <typename T>
SmallVector<uint8_t> packReorderedRaw(ArrayRef<T> values,
                                       ArrayRef<int64_t> shape) {
  auto reordered = reorderMlirToTvx<T>(values, shape);
  SmallVector<uint8_t> out(reordered.size() * sizeof(T));
  if (!reordered.empty())
    std::memcpy(out.data(), reordered.data(), out.size());
  return out;
}

// Extract a `timvx.const`'s storage as raw machine bytes in TIM-VX
// innermost-first order. Mirrors `fmtStaticArrayDecl`'s element-type
// dispatch, but emits a flat byte vector instead of C++ literal text.
//
// On success, fills `out` and returns true. Returns false for element
// types/storage flavours we don't pack (sub-byte ints, non-1/8/16/32/64
// widths, unhandled resource attrs) — callers fall back to the inline
// array path, which renders or also fails the same way.
bool extractReorderedBytes(ElementsAttr values,
                            SmallVectorImpl<uint8_t> &out) {
  auto rt = cast<RankedTensorType>(values.getType());
  Type elem = rt.getElementType();
  ArrayRef<int64_t> shape = rt.getShape();
  uint64_t numel = rt.getNumElements();

  if (auto floats = values.tryGetValues<APFloat>()) {
    SmallVector<APFloat> mlir_vals;
    mlir_vals.reserve(numel);
    for (APFloat v : *floats) mlir_vals.push_back(v);
    auto reordered = reorderMlirToTvx<APFloat>(mlir_vals, shape);
    if (elem.isF32()) {
      out.resize(numel * sizeof(float));
      auto *p = reinterpret_cast<float *>(out.data());
      for (size_t i = 0; i < numel; ++i)
        p[i] = reordered[i].convertToFloat();
      return true;
    }
    if (elem.isF64()) {
      out.resize(numel * sizeof(double));
      auto *p = reinterpret_cast<double *>(out.data());
      for (size_t i = 0; i < numel; ++i)
        p[i] = reordered[i].convertToDouble();
      return true;
    }
    if (elem.isF16()) {
      // Stored as raw u16 bits (matches cxxScalarType's "uint16_t" mapping).
      out.resize(numel * sizeof(uint16_t));
      auto *p = reinterpret_cast<uint16_t *>(out.data());
      for (size_t i = 0; i < numel; ++i)
        p[i] = static_cast<uint16_t>(
            reordered[i].bitcastToAPInt().getZExtValue());
      return true;
    }
    return false;
  }

  if (auto ints = values.tryGetValues<APInt>()) {
    Type stored = elem;
    if (auto qt = dyn_cast<quant::QuantizedType>(elem))
      stored = qt.getStorageType();
    auto it = dyn_cast<IntegerType>(stored);
    if (!it) return false;
    unsigned width = it.getWidth();
    auto pack = [&](auto sample) {
      using U = decltype(sample);
      SmallVector<U> mlir_vals;
      mlir_vals.reserve(numel);
      for (APInt v : *ints) mlir_vals.push_back(static_cast<U>(v.getZExtValue()));
      out = packReorderedRaw<U>(mlir_vals, shape);
    };
    switch (width) {
    case 1:
    case 8:  pack(uint8_t{});  return true;
    case 16: pack(uint16_t{}); return true;
    case 32: pack(uint32_t{}); return true;
    case 64: pack(uint64_t{}); return true;
    default: return false;
    }
  }

  // Resource-attr fast paths: data is already a flat raw-byte buffer in
  // MLIR row-major order. Permute it into TIM-VX innermost-first.
  auto reorderResourceRaw = [&](const void *src, size_t elemSize) {
    auto m = tvxToMlirIndexMap(shape);
    out.resize(numel * elemSize);
    auto *bytes = reinterpret_cast<const uint8_t *>(src);
    for (size_t i = 0; i < numel; ++i)
      std::memcpy(out.data() + i * elemSize,
                  bytes + m[i] * elemSize, elemSize);
  };

  if (auto r = dyn_cast<DenseF32ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return false;
    reorderResourceRaw(data->data(), sizeof(float));
    return true;
  }
  if (auto r = dyn_cast<DenseF64ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return false;
    reorderResourceRaw(data->data(), sizeof(double));
    return true;
  }
  if (auto r = dyn_cast<DenseI8ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return false;
    reorderResourceRaw(data->data(), 1);
    return true;
  }
  if (auto r = dyn_cast<DenseI16ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return false;
    reorderResourceRaw(data->data(), 2);
    return true;
  }
  if (auto r = dyn_cast<DenseI32ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return false;
    reorderResourceRaw(data->data(), 4);
    return true;
  }
  if (auto r = dyn_cast<DenseI64ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return false;
    reorderResourceRaw(data->data(), 8);
    return true;
  }
  if (auto r = dyn_cast<DenseUI8ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return false;
    reorderResourceRaw(data->data(), 1);
    return true;
  }
  if (auto r = dyn_cast<DenseUI16ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return false;
    reorderResourceRaw(data->data(), 2);
    return true;
  }
  if (auto r = dyn_cast<DenseUI32ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return false;
    reorderResourceRaw(data->data(), 4);
    return true;
  }
  if (auto r = dyn_cast<DenseUI64ResourceElementsAttr>(values)) {
    auto data = r.tryGetAsArrayRef();
    if (!data) return false;
    reorderResourceRaw(data->data(), 8);
    return true;
  }
  return false;
}

// Write `bytes` to `<dir>/<filename>`. Returns failure() and emits to
// llvm::errs on I/O error. Used by the const-externalization pre-walk.
LogicalResult writeBytesToFile(StringRef dir, StringRef filename,
                                ArrayRef<uint8_t> bytes) {
  SmallString<256> path(dir);
  llvm::sys::path::append(path, filename);
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "[timvx-to-emitc] cannot open " << path << " for write: "
                 << ec.message() << "\n";
    return failure();
  }
  os.write(reinterpret_cast<const char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  os.close();
  if (os.has_error()) {
    llvm::errs() << "[timvx-to-emitc] write to " << path << " failed\n";
    return failure();
  }
  return success();
}

// Per-const externalization decision filled by the pre-walk in the pass
// driver and consumed by `ConstToEmitC`. Every `timvx.const` lives in
// the map; `externalized=false` means "fall through to the inline-array
// path" (either too small, no extern-dir set, or unsupported storage).
struct ExternConstInfo {
  unsigned id = 0;
  bool externalized = false;
  std::string filename;  // basename, no directory prefix
  size_t byteSize = 0;
};

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

// timvx.const -> timvx_runtime::const_tensor(graph, spec, <data>).
//
// One of two C++ forms is emitted just before the call site, both
// resolving to a `const T*` named `_timvx_const_<id>`:
//
//   inline:    static const T _timvx_const_<id>[N] = { v0, v1, ... };
//   external:  static const T *_timvx_const_<id> =
//                  static_cast<const T *>(::timvx_runtime::mmap_const(
//                      "_timvx_const_<id>.bin", N_bytes));
//
// The choice is made up-front by the pass's pre-walk (see
// `runOnOperation`) and recorded in `externMap` so this pattern just
// reads it back. Both forms decay to `const void*` at the call site.
struct ConstToEmitC : public OpConversionPattern<ConstOp> {
  const llvm::DenseMap<Operation *, ExternConstInfo> *externMap;
  ConstToEmitC(const TypeConverter &tc, MLIRContext *ctx,
                const llvm::DenseMap<Operation *, ExternConstInfo> *m)
      : OpConversionPattern<ConstOp>(tc, ctx), externMap(m) {}
  LogicalResult
  matchAndRewrite(ConstOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto values = cast<ElementsAttr>(op.getValuesAttr());
    auto rt = cast<RankedTensorType>(values.getType());
    StringRef cxxType = cxxScalarType(rt.getElementType());

    auto it = externMap->find(op.getOperation());
    if (it == externMap->end())
      return rewriter.notifyMatchFailure(
          op, "missing externMap entry (pre-walk skipped this op)");
    const ExternConstInfo &info = it->second;
    std::string name = "_timvx_const_" + std::to_string(info.id);

    std::string decl;
    if (info.externalized) {
      llvm::raw_string_ostream os(decl);
      os << "static const " << cxxType << " *" << name
         << " = static_cast<const " << cxxType << " *>("
         << "::timvx_runtime::mmap_const(\"" << info.filename << "\", "
         << info.byteSize << "));";
    } else {
      decl = fmtStaticArrayDecl(values, name);
      if (decl.empty())
        return rewriter.notifyMatchFailure(
            op, "unsupported element type / storage for constant data");
    }
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
  using impl::TIMVXToEmitCPassBase<TIMVXToEmitCPass>::TIMVXToEmitCPassBase;

  void runOnOperation() final {
    MLIRContext *ctx = &getContext();

    for (auto func : llvm::to_vector(getOperation().getOps<func::FuncOp>()))
      prependGraphParam(func);

    // Pre-walk: assign each `timvx.const` a stable id (in walk order) and,
    // when the option is enabled and the const is large enough, materialize
    // its raw bytes into an `_timvx_const_<id>.bin` file under
    // `extern-const-dir`. The resulting per-op decision is consumed by
    // `ConstToEmitC` below — keeping the file I/O outside the pattern
    // avoids re-writing the same file if the pattern is re-applied during
    // partial conversion's match/rewrite churn.
    llvm::DenseMap<Operation *, ExternConstInfo> externMap;
    StringRef externDir = this->externConstDir;
    uint64_t externThreshold = this->externConstThreshold;
    size_t externCount = 0, externBytes = 0;
    {
      unsigned id = 0;
      auto walkResult = getOperation().walk([&](ConstOp op) -> WalkResult {
        ExternConstInfo info;
        info.id = id++;
        auto values = cast<ElementsAttr>(op.getValuesAttr());
        uint64_t numel =
            cast<RankedTensorType>(values.getType()).getNumElements();
        if (!externDir.empty() && numel > externThreshold) {
          SmallVector<uint8_t> bytes;
          if (extractReorderedBytes(values, bytes)) {
            info.filename =
                "_timvx_const_" + std::to_string(info.id) + ".bin";
            info.byteSize = bytes.size();
            if (failed(writeBytesToFile(externDir, info.filename, bytes)))
              return WalkResult::interrupt();
            info.externalized = true;
            ++externCount;
            externBytes += info.byteSize;
          }
          // If extractReorderedBytes failed (unsupported storage), fall
          // through to the inline path; it will either render or also
          // fail, with the same diagnostic surface as before.
        }
        externMap[op.getOperation()] = info;
        return WalkResult::advance();
      });
      if (walkResult.wasInterrupted()) {
        signalPassFailure();
        return;
      }
    }
    if (externCount > 0) {
      llvm::errs() << "[timvx-to-emitc] externalized " << externCount
                   << " timvx.const op(s) totalling " << externBytes
                   << " bytes to " << externDir << "\n";
    }

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

    patterns.add<ConstToEmitC>(converter, ctx, &externMap);

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
