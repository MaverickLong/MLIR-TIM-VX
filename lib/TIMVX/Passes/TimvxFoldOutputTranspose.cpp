//===- TimvxFoldOutputTranspose.cpp - timvx-fold-output-transpose -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Collapse the trailing byte-preserving `reshape`/`transpose` chain feeding
// `func.return` into the function result type — the output mirror of
// `timvx-fold-input-transpose`. See the pass description in
// `TIMVXPasses.td` for the rationale (a classifier tail's
// reshape→transpose→reshape→transpose only re-slots size-1 dims around the
// single class dim, so it's a byte-level identity and the device transposes
// are pure overhead).
//
//===----------------------------------------------------------------------===//

#include "Common.h"

namespace mlir {
namespace timvx {

#define GEN_PASS_DEF_TIMVXFOLDOUTPUTTRANSPOSEPASS
#include "TIMVX/TIMVXPasses.h.inc"

namespace {

// A transpose is byte-preserving iff it keeps every NON-unit dim in its
// original innermost→outermost order — i.e. it only re-slots size-1 axes.
// (Output dim i reads input dim perms[i]; walking perms innermost→outermost,
// the non-unit input dims must appear in strictly increasing index order,
// which is their original order.)
static bool isBytePreservingTranspose(TransposeOp op) {
  auto inTy = dyn_cast<RankedTensorType>(op.getInput1().getType());
  if (!inTy || !inTy.hasStaticShape())
    return false;
  ArrayRef<int64_t> shape = inTy.getShape();
  int prev = -1;
  for (int32_t p : op.getPerms()) {
    if (p < 0 || p >= static_cast<int32_t>(shape.size()))
      return false;
    if (shape[p] == 1)
      continue;  // size-1 dim can move anywhere without touching bytes
    if (p <= prev)
      return false;  // a non-unit dim jumped ahead of another → real reorder
    prev = p;
  }
  return true;
}

struct TIMVXFoldOutputTransposePass
    : public impl::TIMVXFoldOutputTransposePassBase<
          TIMVXFoldOutputTransposePass> {
  void runOnOperation() final {
    ModuleOp module = getOperation();
    auto *ctx = &getContext();

    module.walk([&](func::FuncOp fn) {
      if (fn.isExternal() || fn.getBody().empty())
        return;
      auto ret = dyn_cast<func::ReturnOp>(fn.front().getTerminator());
      if (!ret)
        return;

      bool changed = false;
      SmallVector<Type> resTys(fn.getFunctionType().getResults().begin(),
                               fn.getFunctionType().getResults().end());

      for (unsigned r = 0, e = ret.getNumOperands(); r < e; ++r) {
        Value cur = ret.getOperand(r);
        SmallVector<Operation *> toErase;

        // Walk backward through single-use byte-preserving producers.
        while (Operation *def = cur.getDefiningOp()) {
          if (!def->hasOneUse())
            break;  // an intermediate result is read elsewhere — stop
          Value next;
          if (auto rs = dyn_cast<ReshapeOp>(def)) {
            next = rs.getInput1();  // reshape never moves bytes
          } else if (auto tp = dyn_cast<TransposeOp>(def)) {
            if (!isBytePreservingTranspose(tp))
              break;
            next = tp.getInput1();
          } else {
            break;
          }
          toErase.push_back(def);
          cur = next;
        }

        if (toErase.empty())
          continue;
        ret.setOperand(r, cur);
        resTys[r] = cur.getType();
        // toErase is collected return→producer order; each becomes dead as
        // the one nearer the return is dropped, so erase in collection
        // order. Their `shape` const operands fall to DCE in the following
        // --canonicalize.
        for (Operation *op : toErase)
          op->erase();
        changed = true;
      }

      if (changed) {
        fn.setType(FunctionType::get(
            ctx, fn.getFunctionType().getInputs(), resTys));
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> createTIMVXFoldOutputTransposePass() {
  return std::make_unique<TIMVXFoldOutputTransposePass>();
}

} // namespace timvx
} // namespace mlir
