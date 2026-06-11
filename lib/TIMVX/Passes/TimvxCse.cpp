//===- TimvxCse.cpp - timvx-cse ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Common subexpression elimination scoped to memory-effect-free ops in the
// TIM-VX dialect. Distinct from upstream `--cse` only in that it's part of
// the timvx pipeline (so the pass list stays self-documenting) and it
// runs after `timvx-canonicalize-transpose`'s fan-out duplication step
// has put the duplicates in place.
//
// Why this is load-bearing for the runtime path: my canonicalize-
// transpose pattern pushes a transpose past a fan-out producer by
// rebuilding the producer (one copy per propagating path), which after
// two or three residual blocks materialises the *same* upstream
// `dataconvert(u8 → f32)` three to seven times — once per downstream
// residual that re-reads the skip operand. The same applies to chain
// constants and the `clip` / `add` ops downstream of each fan-out: any
// op whose operands and attributes match an earlier op is doing
// duplicated work at runtime. On resnet50 with the transpose
// canonicalization on, 47 of 57 `timvx.dataconvert` ops are
// duplicates of an earlier op (e.g. 7 calls of `dataconvert(%277)`),
// and similar ratios apply to the surrounding `clip` / `add` chain.
// Each duplicate is a separate kernel dispatch on the NN/PPU path; the
// per-dispatch tax on OVXLIB is ~1ms regardless of tensor size, so a
// pure-op duplicate is a pure perf loss.
//
// Implementation: standard SSA CSE driven by
// `mlir::OperationEquivalence` for structural hashing and equality.
// Walk the function in pre-order (= dominance order for the straight-
// line, branchless IR our lowering emits); for each pure op, look it
// up by structural hash in a `DenseMap`. On a hit, RAUW the duplicate
// onto the canonical (earlier) op's results and erase the duplicate.
// On a miss, insert it as the canonical version.
//
// Why a custom pass instead of upstream `--cse`:
//   * Scoped to the timvx dialect — `OperationEquivalence` would also
//     dedupe ops in other dialects we don't want touched in this stage
//     (e.g. `func`, `tosa` if any leaked through).
//   * Keeps the pipeline definition self-contained: every transform
//     that shapes the timvx IR is a `timvx-*` pass.
//   * Easier to debug in isolation: a pass with a single, tightly-
//     scoped job leaves a smaller blast radius if we ever need to
//     bisect a regression.
//
// Correctness notes:
//   * `OperationEquivalence` compares declared attributes (including
//     `quant_scale` / `quant_zp` on `timvx.const`, `output_scale` /
//     `output_zp` on transpose/cast/dataconvert, kernel/stride/pad on
//     conv2d/pool2d, etc.) AND result types — including the
//     `!quant.uniform<...>` element-type stamp. So two consts with
//     identical `values` but different `(S, Z)` won't be merged, and
//     two casts whose outputs land at different `quant.uniform`
//     stamps won't be merged either. This is the right behaviour for
//     `timvx.const` (the comment on `TIMVX_ConstOp` in `TIMVXOps.td`
//     specifically calls out the canonicalize-by-`values`-alone hazard
//     that we're avoiding here).
//   * The candidate filter `isMemoryEffectFree` excludes ops that
//     have any side effects. Every timvx op of interest (const,
//     transpose, cast, dataconvert, clip, add, sub, multiply, slice,
//     reshape, conv2d, pool2d, fully_connected) is `Pure` in its
//     TableGen definition; the trait expansion adds the
//     `NoMemoryEffect` interface, which is what this check reads.
//
// Run order: `... → timvx-canonicalize-transpose → canonicalize →
//             **timvx-cse** → timvx-conv1x1-to-fc → timvx-to-emitc`.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace timvx {

#define GEN_PASS_DEF_TIMVXCSEPASS
#include "TIMVX/TIMVXPasses.h.inc"

namespace {

// DenseMap hashing/equality keyed on structural equivalence. Two ops
// hash equal iff `OperationEquivalence` says they have the same shape:
// same op name, same operand SSA values, same declared attributes,
// same result types. RegionEquivalenceCache is unused (timvx ops are
// region-less); locations are ignored so two ops created at different
// source positions can still dedupe.
struct EquivalentOpInfo : llvm::DenseMapInfo<Operation *> {
  static unsigned getHashValue(Operation *op) {
    return OperationEquivalence::computeHash(
        op, /*hashOperands=*/OperationEquivalence::directHashValue,
        /*hashResults=*/OperationEquivalence::ignoreHashValue,
        OperationEquivalence::IgnoreLocations);
  }
  static bool isEqual(Operation *lhs, Operation *rhs) {
    if (lhs == rhs) return true;
    if (lhs == getEmptyKey() || lhs == getTombstoneKey() ||
        rhs == getEmptyKey() || rhs == getTombstoneKey())
      return false;
    return OperationEquivalence::isEquivalentTo(
        lhs, rhs, OperationEquivalence::IgnoreLocations);
  }
};

struct TIMVXCsePass : public impl::TIMVXCsePassBase<TIMVXCsePass> {
  void runOnOperation() final {
    llvm::DenseMap<Operation *, Operation *, EquivalentOpInfo> canonical;
    SmallVector<Operation *> redundant;

    getOperation().walk<WalkOrder::PreOrder>([&](Operation *op) {
      // Skip the module/function shells; they aren't candidates and
      // walking them would just descend into the body anyway.
      if (op == getOperation().getOperation()) return WalkResult::advance();
      // Only consider ops from the TIM-VX dialect. Other dialects
      // shouldn't normally appear at this pipeline stage, but a stray
      // `func.return` or similar would be matched as "no operands, no
      // attrs" and we'd risk merging multiple returns — bail.
      if (!isa<TIMVXDialect>(op->getDialect()))
        return WalkResult::advance();
      // Pure ops only (`isMemoryEffectFree` reads the `NoMemoryEffect`
      // interface added by the `Pure` TableGen trait). A non-pure op
      // can't be CSE'd because its semantic effect — not just its
      // returned value — could differ from a "structurally equal"
      // sibling.
      if (!isMemoryEffectFree(op)) return WalkResult::advance();
      // Ops with regions or no results are not interesting CSE
      // candidates in this dialect. Bailing on them keeps the hash
      // map's value type small (each entry is a single Operation*).
      if (op->getNumRegions() != 0 || op->getNumResults() == 0)
        return WalkResult::advance();

      auto [it, inserted] = canonical.try_emplace(op, op);
      if (!inserted) {
        // Found an earlier op with identical structure. Redirect every
        // SSA use of `op`'s results to the canonical op's matching
        // results, then queue this op for erasure (can't erase mid-walk).
        Operation *keep = it->second;
        assert(keep->getNumResults() == op->getNumResults() &&
                "OperationEquivalence said equal but result counts differ");
        for (auto [oldRes, newRes] :
              llvm::zip(op->getResults(), keep->getResults()))
          oldRes.replaceAllUsesWith(newRes);
        redundant.push_back(op);
      }
      return WalkResult::advance();
    });

    for (Operation *op : redundant) op->erase();
  }
};

} // namespace

std::unique_ptr<Pass> createTIMVXCsePass() {
  return std::make_unique<TIMVXCsePass>();
}

} // namespace timvx
} // namespace mlir
