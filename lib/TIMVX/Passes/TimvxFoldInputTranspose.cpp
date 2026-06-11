//===- TimvxFoldInputTranspose.cpp - timvx-fold-input-transpose -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fold a leading `timvx.transpose` that reads a function argument directly
// into the argument's type, and mark the argument TIM-VX-native layout.
//
// See the pass description in `TIMVXPasses.td` for the rationale (the
// entry NCHW→WHCN transpose is the only one `timvx-canonicalize-transpose`
// can't retire, and it's mutually inverse with the runner's host-side
// layout-convert; dropping both lets the preprocessor feed conv2d directly).
//
//===----------------------------------------------------------------------===//

#include "Common.h"

namespace mlir {
namespace timvx {

#define GEN_PASS_DEF_TIMVXFOLDINPUTTRANSPOSEPASS
#include "TIMVX/TIMVXPasses.h.inc"

namespace {

// Attribute key stamped on a folded argument. `parse_input_specs` in
// `example/lower_to_timvx.py` reads this to flip the runner's
// preprocessor into raw-innermost-first (WHCN) byte emission.
static constexpr StringLiteral kInputLayoutAttr = "timvx.input_layout";

struct TIMVXFoldInputTransposePass
    : public impl::TIMVXFoldInputTransposePassBase<
          TIMVXFoldInputTransposePass> {
  void runOnOperation() final {
    ModuleOp module = getOperation();
    auto *ctx = &getContext();

    module.walk([&](func::FuncOp fn) {
      if (fn.isExternal() || fn.getBody().empty())
        return;
      Block &entry = fn.front();

      bool changed = false;
      SmallVector<Type> argTys(fn.getFunctionType().getInputs().begin(),
                               fn.getFunctionType().getInputs().end());

      for (unsigned i = 0, e = entry.getNumArguments(); i < e; ++i) {
        BlockArgument arg = entry.getArgument(i);
        // Single use only: a fan-out arg feeds other consumers whose view
        // of the data would be wrong if we silently re-typed the arg.
        if (!arg.hasOneUse())
          continue;
        auto xpose = dyn_cast<TransposeOp>(*arg.user_begin());
        if (!xpose || xpose.getInput1() != arg)
          continue;

        // Retype the argument to the transpose's WHCN result type, drop
        // the transpose, and record the new arg type for the signature.
        Type newTy = xpose.getResult().getType();
        arg.setType(newTy);
        xpose.getResult().replaceAllUsesWith(arg);
        xpose.erase();
        argTys[i] = newTy;
        fn.setArgAttr(i, kInputLayoutAttr, StringAttr::get(ctx, "whcn"));
        changed = true;
      }

      if (changed) {
        fn.setType(FunctionType::get(ctx, argTys,
                                     fn.getFunctionType().getResults()));
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> createTIMVXFoldInputTransposePass() {
  return std::make_unique<TIMVXFoldInputTransposePass>();
}

} // namespace timvx
} // namespace mlir
