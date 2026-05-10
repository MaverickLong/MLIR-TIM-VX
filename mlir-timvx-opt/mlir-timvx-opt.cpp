//===- mlir-timvx-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "TIMVX/TIMVXDialect.h"
#include "TIMVX/TIMVXPasses.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::timvx::registerPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::timvx::TIMVXDialect, mlir::tosa::TosaDialect,
                  mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::emitc::EmitCDialect, mlir::quant::QuantDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "TIM-VX Bridge Driver\n", registry));
}
