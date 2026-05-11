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
// timvx-promote-i8-to-u8
//===----------------------------------------------------------------------===//

def TIMVXPromoteI8ToU8Pass : Pass<"timvx-promote-i8-to-u8", "::mlir::ModuleOp"> {
  let summary = "Single-pass conversion of every quantized i8 tensor in the "
                "TIM-VX dialect to its u8 equivalent (zp += 128, bytes XOR 0x80).";

  let description = [{
    On VIP9000Nano-DI the runtime auto-promotes every asymmetric int8 IO
    tensor to `u8|asym(zp+128)` and inserts a `DataConvert` at the
    boundary; that auto-DataConvert COMPILE_FAILs at graph IO for
    non-trivial shapes. Pre-emitting all asym int8 specs as u8 matches
    the form TIM-VX wants internally — internally consistent throughout,
    so the auto-rewriter sees no candidates and skips inserting the
    boundary DataConvert.

    Historically this promotion was sprinkled across `TimvxToEmitC`
    (`shouldPromoteI8AsymToU8` / `fmtTensorSpec` / `ConstToEmitC`),
    `TosaToTimvx::CastConversion`, and
    `TimvxQuantResidualFuse::StripUnfusedTailQuant`, with each site
    independently checking "is this an i8 that needs promotion" and
    racing to inject the same `+128` compensation. That fragility is
    what motivates this pass: do every promotion exactly once, in one
    place, before any other timvx-stage pass runs. Downstream passes
    then assume u8 throughout and don't need any promotion-aware logic.

    The pass:
      * Rewrites every `quant.uniform<i8:f32, S:Z>` element type to
        `quant.uniform<u8:f32, S:Z+128>` on op results, op operands,
        block arguments, and `func.func` signature inputs/outputs.
      * Updates every op's `output_zp` discardable attr to `Z + 128`
        when the op's result tensor was i8-quantized.
      * For each `timvx.const` whose result type is now u8-quantized,
        rewrites the dense storage attribute to UI8 with each byte
        XOR'd by 0x80, and shifts the op's `quant_zp` by +128.
      * For each `timvx.cast` that crosses the `f32 ↔ u8` boundary,
        inserts the `±128.0` real-value compensation. Optimization:
        when an adjacent `timvx.add` / `timvx.sub` already carries a
        scalar f32 const (the typical TFLite dequant/requant chain),
        the pass folds the +128 into that existing const instead of
        adding a redundant op — preserves QRF's chain pattern match.

    Run order: `tosa-to-timvx` → `--canonicalize` →
               **`timvx-promote-i8-to-u8`** → `timvx-quant-residual-fuse`
               → `timvx-conv1x1-to-fc` → `timvx-to-emitc`.
  }];

  let constructor = "::mlir::timvx::createTIMVXPromoteI8ToU8Pass()";

  let dependentDialects = [
      "::mlir::timvx::TIMVXDialect",
      "::mlir::quant::QuantDialect"
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

    Run order: `tosa-to-timvx → canonicalize →`
    **`timvx-quant-residual-fuse`** `→ canonicalize → timvx-promote-i8-to-u8`.
    QRF MUST run before `--timvx-promote-i8-to-u8`: the matcher and
    cross-checks expect the canonical i8-space TFLite chain (`cast(i8→f32)
    → sub(zp_i8) → mul → … → mul(1/Sout) → add(zp_i8) → cast(f32→i8)`).
    Once promote injects ±128 byte-shift compensation around every f32↔u8
    cast, the chain shape varies (fold-in-place vs separate `add(+128)`
    op depending on zp-const sharing) and the matcher can't reliably
    discriminate. Running QRF first means it sees the canonical i8 chain
    everywhere; promote then handles BOTH the fused `timvx.add` (bump
    its `output_zp` by +128) AND any un-fused dequant chains (insert
    ±128 compensation) with a single consistent rule.

    Correctness limitation when paired with `tosa-quant-anchor`'s
    `Sw=1/128` convention path: the convention stamps a CONVENTION
    scale on conv outputs whose TFLite-real scale isn't recoverable
    locally. The TFLite-real scale lives only in the dequant chain's
    mul const. The un-fused chain works because TIM-VX `cast` ignores
    the type stamp; QRF replaces the chain with a `timvx.add` whose
    runtime dequant reads from the type stamp — divergent scales lead
    to wrong output by `chain_S / type_S`. `matchDequantChain` adds a
    LOCAL cross-check (S+Z chain const vs operand type stamp) that
    soft-fails on disagreement; this prevents emitting a single wrong
    fused add but does NOT prevent the cascade where an upstream
    convention-anchored conv feeds a downstream chain whose own
    scales locally match. For full correctness on models that hit
    the convention path (TFLite resnet-style, mixed symmetric +
    asymmetric quant), pass `--no-fuse`. See the source-file header
    comment in `TimvxQuantResidualFuse.cpp` for the full analysis.
  }];

  let constructor = "::mlir::timvx::createTIMVXQuantResidualFusePass()";

  let dependentDialects = [
      "::mlir::timvx::TIMVXDialect"
  ];
}

//===----------------------------------------------------------------------===//
// timvx-dequant-fuse
//===----------------------------------------------------------------------===//

def TIMVXDequantFusePass
    : Pass<"timvx-dequant-fuse", "::mlir::ModuleOp"> {
  let summary = "Collapse `cast(int_q→f32) + sub(zp) + mul(scale)` into a "
                "single `timvx.dataconvert` (int→f32 dequantize).";

  let description = [{
    Replaces the per-operand dequant chain with a single
    `tim::vx::ops::DataConvert` call. DataConvert reads (S, Z) off the
    input tensor's `quant.uniform` element type and produces
    `f32 = (byte − Z) × S` natively in one NN-engine kernel dispatch,
    instead of three GC kernel launches for cast + sub + mul.

    Pattern matched (the `sub` is omitted when Z == 0 — canonicalize
    folds `sub _, 0.0`):

    ```
    %f = timvx.cast %u8q : (<...x!quant.uniform<u8:f32, S:Z>>) -> <...xf32>
    %n = timvx.sub %f, %zp_const_f32   // optional
    %r = timvx.multiply %n, %scale_const_f32
    ```

    becomes

    ```
    %r = timvx.dataconvert %u8q : (<...x!quant.uniform<u8:f32, S:Z>>) -> <...xf32>
    ```

    Cross-checks `scale_const ≈ S` and `zp_const ≈ Z`; if they disagree
    the chain isn't a standard dequant (e.g. a TFLite-published
    `Si != S_type` case) and the rewrite bails — the `cast + sub + mul`
    stays.

    Does NOT handle the f32→int requant direction
    (`mul → add → cast(f32→u8)`): DataConvert COMPILE_FAILs for
    f32→int on VIP9000Nano-DI even though `vsi_nn_op_dataconvert.c`'s
    op_check table claims support — see the comment on
    `TIMVX_DataConvertOp` in `TIMVXOps.td`.

    Run order: `…→ timvx-quant-residual-fuse → canonicalize →`
    **`timvx-dequant-fuse`** `→ timvx-conv1x1-to-fc → timvx-to-emitc`.
    QRF runs first because its pattern relies on the un-fused
    `cast+sub+mul` shape to recognise the residual-add operand chains.
  }];

  let constructor = "::mlir::timvx::createTIMVXDequantFusePass()";

  let dependentDialects = [
      "::mlir::timvx::TIMVXDialect",
      "::mlir::quant::QuantDialect"
  ];
}

//===----------------------------------------------------------------------===//
// timvx-canonicalize-transpose
//===----------------------------------------------------------------------===//

def TIMVXCanonicalizeTransposePass
    : Pass<"timvx-canonicalize-transpose", "::mlir::ModuleOp"> {
  let summary = "Push timvx.transpose ops upward through see-through "
                "elementwise ops until they cancel or hit a layout-locked op.";

  let description = [{
    `tosa-to-timvx` sandwiches every spatial op (conv2d / pool2d) between
    an NHWC↔WHCN transpose pair so the spatial op runs in TIM-VX's native
    inner-first WHCN layout. The post-spatial transpose returns to NHWC
    for the trailing TOSA elementwise chain (cast / sub / multiply / add
    / clip / dataconvert / slice), and the pre-next-spatial transpose
    flips back to WHCN. When two adjacent spatial ops are separated only
    by an elementwise chain, those two transposes are mutually inverse
    (`{3,1,0,2} ∘ {2,1,3,0} = identity`) — the only reason they exist is
    that the elementwise ops were inserted in NHWC at TOSA time.

    This pass pushes each transpose upward through its producer one op at
    a time:

      * `unary(x)` → `transpose(unary(x)) = unary(transpose(x))`
        (`timvx.cast`, `timvx.dataconvert`, `timvx.clip`). Result type is
        the operand's type permuted by `perms`.
      * `binary(a, b)` →
        `transpose(binary(a, b)) = binary(transpose(a), transpose(b))`
        (`timvx.add`, `timvx.sub`, `timvx.multiply`). Each operand gets
        its own wrapped transpose; broadcast (numel == 1) constants stay
        scalar.
      * `slice(t, start, size)` →
        `transpose(slice(t, start, size)) =
         slice(transpose(t), perm(start), perm(size))`.
        `start` and `size` are operand consts (`1DTensorOf<Index>`); the
        rewrite emits new `tosa.const_shape` ops with permuted values.
      * `timvx.const` (constant tensor) → materialise a new
        `timvx.const` whose dense storage is physically permuted by
        `perms`. For 1×1×…×1 broadcast scalars this is a no-op rebuild
        (same data); for full-rank consts the rotation is paid once at
        compile time.
      * `transpose(transpose(x))` →  composed perms; identity drops both.
        This is the cancellation step that actually retires transposes
        — once both halves of a sandwich pair have walked up to the
        same producer, the composed perm is identity and both are
        erased.

    Barriers (transpose stops here, no rewrite): `timvx.conv2d`,
    `timvx.pool2d`, `timvx.fully_connected`, `timvx.reshape`,
    `timvx.reduce_sum`, `timvx.pad`, function arguments, and any op
    whose result has more than one user (pushing the transpose past a
    fan-out would re-permute every other consumer's view of the data).

    The pass runs after `timvx-promote-i8-to-u8` and before
    `timvx-to-emitc`, against an IR where every elementwise op is
    layout-agnostic (per-element math, scalar consts) and the only
    layout-locked ops are conv / pool / FC / reshape. On a resnet18 lower
    this collapses 26 transposes down to 2 (the entry NCHW→WHCN and the
    exit reshape→transpose into the model's row-major output spec).
    Constant operands hit by the propagation are folded to permuted
    `timvx.const`s; runtime sees no extra transposes.

    Run order: `... → timvx-promote-i8-to-u8 → canonicalize →
                **timvx-canonicalize-transpose** → timvx-conv1x1-to-fc →
                timvx-to-emitc`.
  }];

  let constructor = "::mlir::timvx::createTIMVXCanonicalizeTransposePass()";

  let dependentDialects = [
      "::mlir::timvx::TIMVXDialect",
      "::mlir::tosa::TosaDialect"
  ];
}

//===----------------------------------------------------------------------===//
// timvx-arith-fold
//===----------------------------------------------------------------------===//

def TIMVXArithFoldPass
    : Pass<"timvx-arith-fold", "::mlir::ModuleOp"> {
  let summary = "Collapse chained scalar-const add/sub/multiply ops into one.";

  let description = [{
    Constant-folds adjacent pairs of `timvx.add` / `timvx.sub` /
    `timvx.multiply` whose second operand is a 1-element f32 splat
    const. Targets the per-block dequant/requant chain shape that
    `timvx-promote-i8-to-u8` leaves behind:

      sub(sub(x, a), b) → sub(x, a + b)
      add(add(x, a), b) → add(x, a + b)
      add(sub(x, a), b) → add(x, b - a)
      sub(add(x, a), b) → add(x, a - b)
      multiply(multiply(x, a), b) → multiply(x, a * b)

    Each fold removes a GC kernel dispatch — the ±128 byte-shift
    compensation `timvx-promote-i8-to-u8` injects around every f32↔u8
    cast leaves the head of each dequant chain as `sub → sub` and the
    tail of every requant chain as `add → add`, and both pairs are
    pure scalar arithmetic that folds at compile time.

    Constraints: the inner op must be single-use (otherwise removing
    it would lose the intermediate another consumer reads); both
    constants must be reachable via `matchConstScalarFloat` (scalar
    splat over a same-rank-as-x type). Per-channel `1×1×1×C` constants
    are not folded — same-shape vector folding isn't needed for the
    chains we emit.

    Run order: `... → timvx-canonicalize-transpose → canonicalize →
                **timvx-arith-fold** → canonicalize → timvx-cse → ...`.
  }];

  let constructor = "::mlir::timvx::createTIMVXArithFoldPass()";

  let dependentDialects = [
      "::mlir::timvx::TIMVXDialect"
  ];
}

//===----------------------------------------------------------------------===//
// timvx-cse
//===----------------------------------------------------------------------===//

def TIMVXCsePass : Pass<"timvx-cse", "::mlir::ModuleOp"> {
  let summary = "Structural common-subexpression elimination over pure timvx ops.";

  let description = [{
    Walks the function in pre-order and dedupes every pure timvx op
    that is structurally equivalent (same op name, same operand SSA
    values, same declared attributes, same result types) to an
    earlier op. The duplicate's result SSA uses are redirected to the
    canonical op's results.

    Motivation: the `timvx-canonicalize-transpose` pass's fan-out
    propagation duplicates the producer when it pushes a transpose
    past a multi-use op (each propagating path gets its own copy of
    the producer at the permuted layout, while the original stays for
    the other consumers). After two or three residual blocks, the
    same upstream `timvx.dataconvert(u8 → f32)` ends up materialised
    three to seven times — one per downstream block that re-reads the
    skip operand. CSE collapses these back into a single op.

    Why not upstream `--cse`: this pass is dialect-scoped (we don't
    want CSE running over `func` or `tosa` ops at this stage) and
    keeps the pipeline self-documenting. Uses `OperationEquivalence`
    for hashing/equality, so it respects all declared attributes —
    including `quant_scale` / `quant_zp` on `timvx.const`,
    `output_scale` / `output_zp` on cast/dataconvert/transpose, and
    kernel/stride/pad on conv2d/pool2d. Two ops with identical
    operands but different quant context will NOT be merged.

    Run order: `... → timvx-arith-fold → canonicalize → **timvx-cse** →
                timvx-conv1x1-to-fc → timvx-to-emitc`.
  }];

  let constructor = "::mlir::timvx::createTIMVXCsePass()";

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

    Const externalization: when `extern-const-dir` is set, every `timvx.const`
    whose element count exceeds `extern-const-threshold` (default 1024) is
    written to a raw `.bin` file under that directory (in TIM-VX innermost-
    first byte order, matching `graph->CreateTensor(spec, data)`'s ABI). The
    emitted C++ replaces the `static const T name[N] = {…};` array with a
    `static const T *name = mmap_const("name.bin", N_bytes);` initializer
    that the runtime resolves against `$TIMVX_CONSTS_DIR` (env override) or
    the directory of the running executable. Smaller consts (and all consts
    when `extern-const-dir` is unset) keep the inline-array form.
  }];

  let constructor = "::mlir::timvx::createTIMVXToEmitCPass()";

  let dependentDialects = [
      "::mlir::timvx::TIMVXDialect",
      "::mlir::emitc::EmitCDialect",
      "::mlir::func::FuncDialect"
  ];

  let options = [
    Option<"externConstDir", "extern-const-dir", "std::string",
           /*default=*/"\"\"",
           "Directory to write large `timvx.const` data into as raw `.bin` "
           "files (one per externalized const). Empty (default) inlines "
           "every const as a `static const T name[N] = {…};` array.">,
    Option<"externConstThreshold", "extern-const-threshold", "uint64_t",
           /*default=*/"1024",
           "Numel threshold above which `timvx.const`s are written to "
           "external `.bin` files (only honored when `extern-const-dir` is "
           "set). Use 0 to externalize every const regardless of size.">
  ];
}

#endif // TIMVX_PASS
