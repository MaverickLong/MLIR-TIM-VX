//===- TIMVXPasses.h - TIMVX passes  ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef TIMVX_TIMVXPASSES_H
#define TIMVX_TIMVXPASSES_H

#include "TIMVX/TIMVXDialect.h"
#include "TIMVX/TIMVXOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace timvx {

// Forward-declared so the .td-generated `Base<>` can reference it as the
// pass's constructor (see `let constructor = ...` in TIMVXPasses.td).
std::unique_ptr<Pass> createTosaToTIMVXPass();

#define GEN_PASS_DECL
#include "TIMVX/TIMVXPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "TIMVX/TIMVXPasses.h.inc"

} // namespace timvx
} // namespace mlir

#endif
