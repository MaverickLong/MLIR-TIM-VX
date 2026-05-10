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
// tosa-quant-anchor
//===----------------------------------------------------------------------===//

def TosaQuantAnchorPass : Pass<"tosa-quant-anchor", "::mlir::ModuleOp"> {
  let summary = "Anchor (scale, zp) on every TOSA tensor type as "
                "!quant.uniform<i8:f32, S:Z>.";

  let description = [{
    Single-pass entry point that materialises absolute per-tensor
    quantization context as `!quant.uniform<i8:f32, S:Z>` element types
    on every quantized SSA value in the function. After this pass runs,
    downstream lowerings (`tosa-to-timvx`, `RescaleConvFusion`,
    `QuantResidualFuse`) read (S, Z) directly off the operand types
    instead of re-deriving them.

    Why a separate pass:

    The IREE TFLite frontend strips the `!quant.uniform<...>` element
    types its TFLite source carried — TOSA `RescaleOp` only encodes the
    scale ratio `M = (Si × Sw) / So` as an integer multiplier+shift,
    not absolute Si / Sw / So. Some tensors recover absolute Si from
    the dequant-chain `mul` constants TFLite emits between residual
    blocks (12 anchor sites in resnet18); the rest need the canonical
    `Sw = 1/128` symmetric-int8 weight convention to back-compute So
    through the rescale chain. Applying that convention scattered
    across `buildQuantInfoMap` / `getProducerQuant` / `RescaleConvFusion`
    made it invisible: a model that genuinely needs a different Sw
    silently mislowered. Centralising the convention here makes it
    observable in the IR (every quantized tensor's type carries the
    derived (S, Z)) and emits a remark for each tensor whose Si came
    from the convention path rather than from a real anchor.

    Sources of (S, Z), in priority:
      1. `timvx.output_scale`/`timvx.output_zp` arg-attr on a func arg.
      2. `timvx.output_scale`/`timvx.output_zp` res-attr on a func result.
      3. Dequant chain `cast(i8→f32) → sub(zp_const) → mul(scale_const)`
         immediately downstream of a tensor — `mul`'s scalar f32 const
         is the TFLite-published Si for that tensor.
      4. `tosa.rescale`'s `output_zp` operand and the chain-or-arg-derived
         Si: So = Si × Sw / M (Sw = 1/128 convention).
      5. Pool / pad / transpose / reshape / slice propagate quant from
         operand 0 to result.

    Pass output:
      - Every quantized i8 SSA value's tensor element type becomes
        `!quant.uniform<i8:f32, S:Z>`.
      - The function signature and `func.return` operand types are
        updated to match.
      - `tosa.const` ops with quantized result types keep their
        `dense<...>` storage values verbatim (storage type still i8).

    Run order: tosa-const-fold → tosa-fold-avgpool-reduce →
               canonicalize → **tosa-quant-anchor** → tosa-to-timvx → ...
  }];

  let constructor = "::mlir::timvx::createTosaQuantAnchorPass()";

  let dependentDialects = [
      "::mlir::tosa::TosaDialect",
      "::mlir::quant::QuantDialect"
  ];
}

//===----------------------------------------------------------------------===//
// tosa-fold-avgpool-reduce
//===----------------------------------------------------------------------===//

def TosaFoldAvgPoolReducePass
    : Pass<"tosa-fold-avgpool-reduce", "::mlir::ModuleOp"> {
  let summary = "Fold global avg-pool emitted as `reduce_sum × 2 → mul(1/N)`"
                " into `avg_pool2d` operating on the requantized u8 tensor.";

  let description = [{
    tflite's TOSA exporter emits global average pooling as a chain of two
    `tosa.reduce_sum` ops (one over H, one over W) followed by a
    `tosa.reshape` that drops the spatial size-1 dims and a `tosa.mul` by
    `1/(H*W)`. After that comes the trailing requantize chain
    (`mul scale → add zp → cast f32 → narrow int`) that
    `RequantI32SkipFold` collapses upstream.

    The decomposed `tosa.reduce_sum` lowers via `ReduceSumConversion` to
    `kernel_extent` `tosa.slice` + `kernel_extent − 1` `tosa.add` ops in
    fp32 — each `slice` with a unique `(start, size)` signature compiles
    a dedicated EVIS shader on this chip. For a 7×7 reduce that's 14
    shaders; libCLC's per-shader compile heap doesn't release between
    invocations, and a ResNet18 compile peak balloons to ~5 GiB — large
    enough that the kernel OOM-killer trips it.

    This pass recognizes the avg-pool chain at the TOSA level, walks
    back from the trailing requant cast through the reduce_sum chain,
    and replaces it with a single `tosa.avg_pool2d` operating on the
    u8 tensor. The trick that makes this sound:

      original: `cast(scale * (sum(x)/N) + zp)` over fp32 sum
      rewrite:  `avg_pool2d(cast(scale * x + zp))` over u8

    These are mathematically equal — `cast` (saturate-round) commutes
    with averaging up to half-an-LSB rounding noise. We push the
    per-pixel requant up to before the spatial reduction so the avg
    runs on u8 (a single Pool2D AVG kernel rather than 14 slice
    shaders), and `avg_pool2d` divides by N internally so the explicit
    `mul(1/N)` drops out.

    Limitations:
      * Only matches when the trailing cast carries the
        `timvx.output_scale` / `timvx.output_zp` discardable attrs that
        `RequantI32SkipFold` deposits — i.e. the requant tail must
        already be in canonical post-fold form. (Run order:
        `tosa-const-fold` → this pass → `tosa-to-timvx`.)
      * Reduce axes must be H (axis 1 in NHWC) and W (axis 2). Other
        axis combinations are left alone.
      * If any op between the spatial fp32 input and the requant cast
        has multiple uses (other than the consumer in the chain), the
        rewrite bails — pulling the cast forward would change semantics
        for the other consumer.
  }];

  let constructor = "::mlir::timvx::createTosaFoldAvgPoolReducePass()";

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
// timvx-conv1x1-to-fc
//===----------------------------------------------------------------------===//

def TIMVXConv1x1ToFCPass : Pass<"timvx-conv1x1-to-fc", "::mlir::ModuleOp"> {
  let summary = "Rewrite 1x1 timvx.conv2d with W=H=1 input/weight as fully_connected.";

  let description = [{
    Pattern: a `timvx.conv2d %x_4d, %w_4d, %b` whose input/weight/output
    have W=1, H=1 (TIM-VX inner-first WHCN/WHIcOc) and unit pad/stride/
    dilation collapses to a fully-connected layer over `K = Ic`,
    `M = Oc`, batch = N. tflite's quantized exporter routes the trailing
    classifier into this 1x1 conv shape; FullyConnected has a tighter
    NN-engine path on Vivante NPUs (the weight is pre-tiled at
    bind time vs. Conv2D's per-call spatial setup).

    Rewrites to:
      reshape `%x_4d : 1x1xKxN` → `KxN`
      reshape weight const `1x1xKxM` → `KxM`        (compile-time, dense data unchanged)
      timvx.fully_connected -> `MxN`
      reshape `MxN` → `1x1xMxN`

    Bias and `output_scale` / `output_zp` carry over unchanged. Runs
    after `--tosa-to-timvx`; the rewrite itself is dialect-internal.
  }];

  let constructor = "::mlir::timvx::createTIMVXConv1x1ToFCPass()";

  let dependentDialects = [
      "::mlir::timvx::TIMVXDialect"
  ];
}

//===----------------------------------------------------------------------===//
// timvx-quant-residual-fuse
//===----------------------------------------------------------------------===//

def TIMVXQuantResidualFusePass
    : Pass<"timvx-quant-residual-fuse", "::mlir::ModuleOp"> {
  let summary = "Collapse the fp32 dequant+add+requant chain back to a "
                "quantized timvx.add, eliminating the per-residual fp32 "
                "detour every bottleneck block carries.";

  let description = [{
    The TOSA quantized rescale lowering produces a chain of fp32 ops
    around every residual add in tflite-style models. For two i8
    operands `A` (scale Sa, zp Za) and `B` (scale Sb, zp Zb) summed
    into an i8 result with (Sout, Zout), the lowered chain looks like:

        Af = cast i8 → f32 (A)
        An = sub  (Af, Za)
        Ar = mul  (An, Sa)            // operand A in real-value space
        Bf = cast i8 → f32 (B)
        Bn = sub  (Bf, Zb)
        Br = mul  (Bn, Sb)            // operand B in real-value space
        S  = add  (Ar, Br)            // residual add (still f32)
        [optional clip(S, 0, +inf)]   // post-add ReLU
        Si = mul  (clip?(S), 1/Sout)
        Sz = add  (Si, Zout)
        OUT = cast f32 → i8 (Sz)      // produces (Sout, Zout)

    On VIP9000Nano-DI every fp32 op runs on the GC GPU's CL/EVIS
    kernels at full spatial size — for ResNet50's 16 residuals this
    dominates inference latency (~7× over a fully INT8 path). The
    NN-core's quantized `Add` is a much faster path on the same chip
    (probe matrix: u8↔u8↔u8 with mixed-quant operands PASSes; see
    project_a733_fp32_conv_unsupported.md).

    Rewrites the whole chain to:

        OUT = timvx.add A B {output_scale = Sout, output_zp = Zout}
        [optional timvx.clip OUT ...]    // preserves ReLU semantics

    Constant-value cross-checks: each per-operand `mul scale_const`
    must match the producer's `output_scale`, each `sub zp_const` must
    match the producer's `output_zp`, the tail `mul (1/Sout)` must
    invert `Sout`, and the tail `add Zout` must match the cast's
    `output_zp`. Mismatches fall through (the chain isn't a pure
    rescale chain).

    Runs after `--tosa-to-timvx`; idempotent.
  }];

  let constructor = "::mlir::timvx::createTIMVXQuantResidualFusePass()";

  let dependentDialects = [
      "::mlir::timvx::TIMVXDialect"
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
