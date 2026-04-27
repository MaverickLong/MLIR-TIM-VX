//===- Dialects.cpp - CAPI for dialects -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TIMVX-c/Dialects.h"

#include "TIMVX/TIMVXDialect.h"
#include "TIMVX/TIMVXOps.h"
#include "TIMVX/TIMVXTypes.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TIMVX, timvx, mlir::timvx::TIMVXDialect)

// TODO: re-add type-specific CAPI entry points (GraphType, etc.) as they
// stabilize.
