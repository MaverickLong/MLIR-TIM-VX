//===- TIMVXDialect.cpp - TIMVX dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TIMVX/TIMVXDialect.h"
#include "TIMVX/TIMVXOps.h"
#include "TIMVX/TIMVXTypes.h"

using namespace mlir;
using namespace mlir::timvx;

#include "TIMVX/TIMVXEnums.cpp.inc"
#include "TIMVX/TIMVXOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// TIMVX dialect.
//===----------------------------------------------------------------------===//

void TIMVXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TIMVX/TIMVXOps.cpp.inc"
      >();
  registerTypes();
}
