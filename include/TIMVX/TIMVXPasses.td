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
// tosa-layout-tag
//===----------------------------------------------------------------------===//

def TosaLayoutTagPass : Pass<"tosa-layout-tag", "::mlir::ModuleOp"> {
  let summary = "Annotate per-axis tensor layouts (NHWC, OHWI, broadcast, ...).";

  let description = [{
    Forward dataflow analysis that anchors at TOSA's spatial ops — where the
    spec contractually fixes operand layouts (conv2d input NHWC + weight OHWI;
    pool2d input NHWC) — and propagates the layout assignment through layout-
    passthrough ops (binary / unary elementwise, with broadcast handling).

    Each tagged op result gets a `timvx.layout` StringAttr like `"N,H,W,C"` or
    `"1,1,1,C"`. Tagged func args get the same attribute via `arg_attrs`.
    Comma-separated axis roles (`N`, `H`, `W`, `C`, `D`, `Oc`, `Ic`, `1` for
    broadcast).

    Propagation halts at ops whose layout transfer function isn't defined here
    — `tosa.matmul`, `tosa.fully_connected`, `tosa.reshape`, `tosa.transpose`
    (per request, transposes are left to `--tosa-reduce-transposes`). Untagged
    values are treated as the boundary by the downstream WHCN-canonicalize
    pass.

    Errors on conflict: a value tagged with two different layouts (i.e. used
    as a weight by one consumer and a feature map by another) — diagnostic
    points at the value with the existing and new layouts.
  }];

  let constructor = "::mlir::timvx::createTosaLayoutTagPass()";

  let dependentDialects = [
      "::mlir::tosa::TosaDialect",
      "::mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// tosa-layout-to-whcn
//===----------------------------------------------------------------------===//

def TosaLayoutToWhcnPass : Pass<"tosa-layout-to-whcn", "::mlir::ModuleOp"> {
  let summary = "Permute layout-tagged TOSA tensors from NHWC/OHWI to WHCN.";

  let description = [{
    Consumes the `timvx.layout` attributes set by `--tosa-layout-tag` and
    rewrites the IR so that every tagged tensor is in TIM-VX's native
    innermost-first WHCN convention.

    Concretely, for each tagged value:
      * Tagged `tosa.const` → emit a new const with permuted dense data
        and the WHCN-shaped tensor type. Compile-time only.
      * Tagged op result (non-const) → update the result's tensor type
        in place to the WHCN-permuted shape. The op semantics are
        layout-agnostic (broadcast / pointwise) so this is sound; for
        spatial ops (`tosa.conv2d` / `tosa.max_pool2d` / `tosa.avg_pool2d`),
        the kernel / stride / pad / dilation arrays are also swapped from
        TOSA's `[H, W]` / `[top, bot, left, right]` to WHCN's
        `[W, H]` / `[left, right, top, bot]`.

    At each tagged ↔ untagged boundary, a single `tosa.transpose` is
    inserted to bridge the layouts (NCHW or NHWC ↔ WHCN). The entry-side
    transpose typically composes with the front-end's NCHW→NHWC
    transpose; `--canonicalize` then folds the consecutive pair into one.

    After this pass, the IR carries no `timvx.layout` attribute
    (intentionally stripped — the layout is now implicit in the shapes).
    Designed to run immediately before `--tosa-to-timvx`.
  }];

  let constructor = "::mlir::timvx::createTosaLayoutToWhcnPass()";

  let dependentDialects = [
      "::mlir::tosa::TosaDialect",
      "::mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// tosa-const-fold
//===----------------------------------------------------------------------===//

def TosaConstFoldPass : Pass<"tosa-const-fold", "::mlir::ModuleOp"> {
  let summary = "Fold elementwise / reshape ops on constant TOSA tensors.";

  let description = [{
    Greedy tosa→tosa rewrite that constant-folds `add`, `sub`, `mul`, `pow`,
    `reciprocal`, and `reshape` when all operands trace to `tosa.const`.
    Iterates to fixed point so chains (e.g. BatchNorm
    `var → +eps → ^0.5 → 1/x → *gamma`) collapse to a single const each.

    Extends upstream's `--tosa-layerwise-constant-fold` (which covers
    `transpose`, `reciprocal`, and the `reduce*` family) with the binary
    elementwise ops and `reshape`. FP32 only.

    Motivated by TIM-VX's eltwise kernels rejecting same-rank size-1-vs-N
    broadcast (`tensor<1xf32> + tensor<64xf32>`), which is the exact shape
    every BatchNorm scalar add produces.
  }];

  let constructor = "::mlir::timvx::createTosaConstFoldPass()";

  let dependentDialects = [
      "::mlir::tosa::TosaDialect"
  ];
}

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

//===----------------------------------------------------------------------===//
// timvx-to-emitc
//===----------------------------------------------------------------------===//

def TIMVXToEmitCPass : Pass<"timvx-to-emitc", "::mlir::ModuleOp"> {
  let summary = "Lower TIM-VX dialect ops to EmitC for C++ source emission.";

  let description = [{
    Lowers the TIM-VX op set onto `emitc.call_opaque` invocations of helpers
    in a `timvx_runtime::` namespace (header supplied out-of-band). Tensors
    become `!emitc.opaque<"std::shared_ptr<tim::vx::Tensor>">`, shape vectors
    become `!emitc.opaque<"std::vector<uint32_t>">`.

    Per-op constants (pad / stride / dilation / kernel / perms / output spec)
    are formatted in-place as opaque C++ text. The `graph` argument is
    referenced as a free identifier — the surrounding function is expected to
    have one in scope.
  }];

  let constructor = "::mlir::timvx::createTIMVXToEmitCPass()";

  let dependentDialects = [
      "::mlir::timvx::TIMVXDialect",
      "::mlir::emitc::EmitCDialect",
      "::mlir::func::FuncDialect"
  ];
}

#endif // TIMVX_PASS
