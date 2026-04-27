//===- TIMVXExtensionNanobind.cpp - Extension module ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TIMVX-c/Dialects.h"
#include "mlir-c/Dialect/Arith.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/IRTypes.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

NB_MODULE(_timvxDialectsNanobind, m) {
  //===--------------------------------------------------------------------===//
  // timvx dialect
  //===--------------------------------------------------------------------===//
  auto timvxM = m.def_submodule("timvx");

  // TODO: re-add Python bindings for TIM-VX-specific types (GraphType, etc.)
  // alongside the matching CAPI entry points.

  timvxM.def(
      "register_dialects",
      [](mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext
             context,
         bool load) {
        MlirDialectHandle arithHandle = mlirGetDialectHandle__arith__();
        MlirDialectHandle timvxHandle = mlirGetDialectHandle__timvx__();
        MlirContext context_ = context.get()->get();
        mlirDialectHandleRegisterDialect(arithHandle, context_);
        mlirDialectHandleRegisterDialect(timvxHandle, context_);
        if (load) {
          mlirDialectHandleLoadDialect(arithHandle, context_);
          mlirDialectHandleLoadDialect(timvxHandle, context_);
        }
      },
      nb::arg("context").none() = nb::none(), nb::arg("load") = true);
}
