//===- TIMVXTypes.td - TIMVX dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//                          *** NHWC ONLY ***
//
// The TIM-VX dialect is NHWC. Tensors are channel-innermost; spatial rank is
// inferred from the tensor rank:
//
//   spatial rank 1 (conv1d / pool1d)   activations: NWC      weights: OWI
//   spatial rank 2 (conv2d / pool2d)   activations: NHWC     weights: OHWI
//   spatial rank 3 (conv3d / pool3d)   activations: NDHWC    weights: ODHWI
//
// Equivalent in TIM-VX's own innermost-first notation: CWN / CWHN / CWHDN for
// activations; WIcOc / IcWHOc / OcIcWHD for weights. TIM-VX's default
// weight layout is OIHW (= WHIcOc) but it accepts the channel-innermost form
// natively — we pick channel-innermost end-to-end.
//
//===----------------------------------------------------------------------===//
//
// Type-system shape:
//
//   - Tensors use builtin `tensor<...>`. Shape, layout, and quant
//     parameters are all expressible via the upstream type system; there is no
//     custom tensor type.
//
//   - Element-type constraints are expressed at op operand/result sites via
//     the TensorOf<[...]> constraint aliases below — see uses in TIMVXOps.td.
//
//   - The only custom type defined here is !timvx.graph, an opaque handle
//     returned by timvx.graph_compile and consumed by timvx.graph_run.
//
//===----------------------------------------------------------------------===//
//
// TIM-VX <-> MLIR dtype mapping (from tim::vx::DataType in
// TIM-VX/include/tim/vx/types.h):
//
//   tim::vx::INT4  / UINT4    →  !quant.uniform<i*<storage:4>:f32, ...>
//                                (no native MLIR i4; routed via quant)
//   tim::vx::INT8  / UINT8    →  i8 / ui8, or !quant.uniform<i8|u8:f32, ...>
//   tim::vx::INT16 / UINT16   →  i16 / ui16
//   tim::vx::INT32 / UINT32   →  i32 / ui32
//   tim::vx::INT64            →  i64
//   tim::vx::FLOAT16          →  f16        (NPU NN-core native)
//   tim::vx::FLOAT32          →  f32        (API-accepted, auto-converted to
//                                            FP16 inside; excluded at
//                                            partition boundaries — see below)
//   tim::vx::BOOL8            →  i1         (widened to i8 at the runtime
//                                            walker when emitting tim::vx)
//
// TIM-VX <-> MLIR quant mapping (tim::vx::QuantType):
//
//   ASYMMETRIC                    !quant.uniform<i8:f32, scale:zp>
//   SYMMETRIC_PER_CHANNEL         !quant.uniform<i8<-127:127>:f32:axis, ...>
//   ASYMMETRIC_PER_CHANNEL        !quant.uniform<i8:f32:axis, ...>
//   DYNAMIC_FIXED_POINT           Uncommon so leave it for now.
//
//===----------------------------------------------------------------------===//

#ifndef TIMVX_TYPES
#define TIMVX_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/Quant/IR/QuantBase.td"
include "TIMVX/TIMVXDialect.td"

//===----------------------------------------------------------------------===//
// Type base
//===----------------------------------------------------------------------===//

class TIMVX_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TIMVX_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Tensor
//===----------------------------------------------------------------------===//

// All element types TIM-VX accepts at its C++ API level.
// Native i4 is omitted (no upstream MLIR i4); INT4/UINT4 reach TIM-VX via
// quant_QuantizedType with storage width 4.
//
// Note: the TableGen-side predicate is `quant_QuantizedType` (defined in
// upstream's QuantBase.td); the C++ class `mlir::quant::AnyQuantizedType` is
// a concrete subclass of QuantizedType and isn't directly usable as an ODS
// type constraint.
def TIMVX_Tensor : TensorOf<[
    F16, F32,
    I1,                                                  // == tim::vx::BOOL8
    I8, UI8, I16, UI16, I32, UI32, I64,
    quant_QuantizedType
]>;

//===----------------------------------------------------------------------===//
// !timvx.graph
//===----------------------------------------------------------------------===//
// Opaque handle for a compiled tim::vx::Graph. Produced by timvx.graph_compile
// and consumed by timvx.graph_run / timvx.graph_query.

def TIMVX_GraphType : TIMVX_Type<"Graph", "graph"> {
  let summary = "Opaque handle to a compiled tim::vx::Graph";
}

#endif // TIMVX_TYPES
