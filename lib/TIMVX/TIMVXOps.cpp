//===- TIMVXOps.cpp - TIMVX dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TIMVX/TIMVXOps.h"
#include "TIMVX/TIMVXDialect.h"
#include "llvm/Support/LogicalResult.h"

#define GET_OP_CLASSES
#include "TIMVX/TIMVXOps.cpp.inc"

namespace mlir {
namespace timvx {

// Folders for ConstantLike ops.
OpFoldResult ConstOp::fold(FoldAdaptor) { return getValuesAttr(); }
OpFoldResult ConstShapeOp::fold(FoldAdaptor) { return getValuesAttr(); }

} // namespace timvx
} // namespace mlir