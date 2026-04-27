//===- TIMVXTypes.cpp - TIMVX dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TIMVX/TIMVXTypes.h"

#include "TIMVX/TIMVXDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::timvx;

#define GET_TYPEDEF_CLASSES
#include "TIMVX/TIMVXOpsTypes.cpp.inc"

void TIMVXDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "TIMVX/TIMVXOpsTypes.cpp.inc"
      >();
}
