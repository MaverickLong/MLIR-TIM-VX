//===- TIMVXPasses.td - TIMVX dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TIMVX_PASS
#define TIMVX_PASS

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// tosa-to-timvx
//===----------------------------------------------------------------------===//

def TosaToTIMVXPass : Pass<"tosa-to-timvx", "::mlir::ModuleOp"> {
  let summary = "Lower TOSA dialect ops to TIM-VX dialect ops.";

  let description = [{
    Performs a (almost) one-to-one structural conversion of the supported
    TOSA op set to the matching TIM-VX op. The pass does not register a
    `TypeConverter`; it is a pure op-rewrite pass.
  }];

  let constructor = "::mlir::timvx::createTosaToTIMVXPass()";

  let dependentDialects = [
      "::mlir::timvx::TIMVXDialect",
      "::mlir::tosa::TosaDialect"
  ];
}

#endif // TIMVX_PASS
