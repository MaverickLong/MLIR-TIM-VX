//===- TIMVXOps.td - TIMVX dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TIM-VX Operation Definitions.
// Traits and interfaces are copied from TosaOps.td where they map.
//
//===----------------------------------------------------------------------===//

#ifndef TIMVX_OPS
#define TIMVX_OPS

include "TIMVX/TIMVXTypes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class TIMVX_FuncTypeOp<string mnemonic, list<Trait> traits = []>
    : TIMVX_Op<mnemonic, traits> {
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Enums
//===----------------------------------------------------------------------===//

// tim::vx::PoolType — selects the pooling reduction kind for timvx.pool2d.
def TIMVX_POOL_MAX         : I32EnumAttrCase<"MAX",         0>;
def TIMVX_POOL_AVG         : I32EnumAttrCase<"AVG",         1>;
def TIMVX_POOL_L2          : I32EnumAttrCase<"L2",          2>;
def TIMVX_POOL_AVG_ANDROID : I32EnumAttrCase<"AVG_ANDROID", 3>;

def TIMVX_PoolTypeAttr : I32EnumAttr<"PoolType",
    "TIM-VX pooling reduction kind",
    [TIMVX_POOL_MAX, TIMVX_POOL_AVG,
     TIMVX_POOL_L2, TIMVX_POOL_AVG_ANDROID]> {
  let cppNamespace = "::mlir::timvx";
}

//===----------------------------------------------------------------------===//
// Constant ops
//===----------------------------------------------------------------------===//

// Convention for the optional quant attrs scattered on tensor-producing ops:
//   `*_scale` is F64Attr (matches mlir::quant::UniformQuantizedType's `double`
//   scale).  `*_zp` is I64Attr (storage-width-agnostic).  When both are set,
//   the EmitC pattern emits `Quantization(ASYMMETRIC, scale, zp)` on the
//   corresponding TensorSpec.

def TIMVX_ConstOp : TIMVX_Op<"const", [
        // Note: deliberately *not* `ConstantLike`. MLIR's canonicalizer
        // would otherwise dedupe `timvx.const` ops by `values` alone —
        // discardable attrs and the typed `quant_scale`/`quant_zp`
        // fields aren't part of its uniquing key. So a quant-tagged
        // const emitted by RescaleConvFusion would silently get
        // CSE-merged against an unrelated quant-less const carrying the
        // same `values` (e.g. a `dense<0>` bias vs. a `dense<0>`
        // input_zp), and the conv would lose its quant info
        // non-deterministically.
        Pure,
        AllShapesMatch<["values", "output"]>,
        FirstAttrDerivedResultType
    ]> {
  let summary = "Constant tensor.";
  let description = [{
    A constant tensor for use as input to other ops. At runtime emission,
    lowers to a TIM-VX tensor created with TensorAttribute::CONSTANT and
    the provided data.
  }];

  let arguments = (ins
      ElementsAttr:$values,
      OptionalAttr<F64Attr>:$quant_scale,
      OptionalAttr<I64Attr>:$quant_zp
  );
  let results = (outs TIMVX_Tensor:$output);
  let hasFolder = 1;
}

def TIMVX_ConstShapeOp : TIMVX_Op<"const_shape", [ConstantLike, Pure]> {
  let summary = "Constant <1xIndex> shape vector.";
  let description = [{
    A constant 1-D shape used as a shape operand. Consumed as contant at
    compile time.
  }];

  let arguments = (ins IndexElementsAttr:$values);
  let results = (outs 1DTensorOf<[Index]>:$output);
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Activation ops
//===----------------------------------------------------------------------===//

def TIMVX_ClipOp : TIMVX_FuncTypeOp<"clip", [
        Pure,
        SameOperandsAndResultShape
        // No SameOperandsAndResultElementType: when the input/output
        // are quant.uniform<i8:f32, S:Z>, the post-clip (S, Z) commonly
        // differs from the input's (e.g. QuantResidualFuse stamps the
        // requant tail's Sout on the clipped result). The runtime
        // handles per-tensor scale_factor between input/output.
    ]> {
  let summary = "Elementwise clip(input, min_val, max_val).";
  let description = [{
    Clip every element of `input` into the range [min_val, max_val].
    Lowers to tim::vx::ops::Clip(graph, min_val, max_val).
  }];

  let arguments = (ins
      TIMVX_Tensor:$input,
      F32Attr:$min_val,
      F32Attr:$max_val
  );
  let results = (outs TIMVX_Tensor:$output);
}

//===----------------------------------------------------------------------===//
// Convolution / matmul / pool
//===----------------------------------------------------------------------===//

def TIMVX_Conv2DOp : TIMVX_FuncTypeOp<"conv2d", [Pure]> {
  let summary = "2-D convolution (WHCN input, WHIcOc weight).";
  let description = [{
    Performs 2-D convolution. Input is rank-4 WHCN; weight is rank-4 WHIcOc;
    bias is rank-1.

    Pad order: [left, right, top, bottom].
    Stride / dilation order: [W, H].

    Lowers to tim::vx::ops::Conv2d with constant pad / stride / dilation
    arrays. TIM-VX defaults (DataLayout::WHCN input / DataLayout::WHIcOc
    kernel) match the IR-side layout, so no explicit layout argument is
    passed.
  }];

  let arguments = (ins
      TIMVX_Tensor:$input,
      TIMVX_Tensor:$weight,
      TIMVX_Tensor:$bias,

      DenseI64ArrayAttr:$pad,            // [left, right, top, bottom]
      DenseI64ArrayAttr:$stride,         // [W, H]
      DenseI64ArrayAttr:$dilation,       // [W, H]

      // Optional output Quantization. When set, the result tensor's
      // TensorSpec is emitted with Quantization(ASYMMETRIC, scale, zp).
      OptionalAttr<F64Attr>:$output_scale,
      OptionalAttr<I64Attr>:$output_zp
  );
  let results = (outs TIMVX_Tensor:$output);
}

def TIMVX_MatMulOp : TIMVX_FuncTypeOp<"matmul", [NoMemoryEffect]> {
  let summary = "Batched 2-D matrix multiplication.";
  let description = [{
    Performs 2-D matrix multiplication (rank-3 batched). Lowers to
    tim::vx::ops::Matmul with transpose_a = transpose_b = false.
  }];

  let arguments = (ins
      TIMVX_Tensor:$a,
      TIMVX_Tensor:$b
  );
  let results = (outs TIMVX_Tensor:$output);
}

def TIMVX_FullyConnectedOp : TIMVX_FuncTypeOp<"fully_connected", [Pure]> {
  let summary = "Fully-connected (dense) layer: Y = W^T @ X + b.";
  let description = [{
    Computes `output[m, b] = sum_k(weight[k, m] * input[k, b]) + bias[m]`.
    Tensors follow the TIM-VX inner-first convention with the
    contraction axis K at index 0:

      input:  rank-2 `{K, batch}`               (K innermost)
      weight: rank-2 `{K, out_features}`        (K innermost)
      bias:   rank-1 `{out_features}`
      output: rank-2 `{out_features, batch}`    (out_features innermost)

    Lowers to tim::vx::ops::FullyConnected with axis=0 (the contraction
    dim is at index 0 of input). This matches the convention the
    `vsi_nn_op_fullconnect2.c` kernel walks bytes in — it internally
    reshapes input/output to `{K, batch}` / `{ofm, batch}` before
    dispatching to `vxFullyConnectedLayer`. The earlier
    `{batch, K}` / `axis=1` form happened to give correct `num_fc *
    num_no_fc` arithmetic for batch=1, but produced column-permuted
    output bytes because the weight bytes were laid out for the
    transposed contraction axis.
  }];

  let arguments = (ins
      TIMVX_Tensor:$input,
      TIMVX_Tensor:$weight,
      TIMVX_Tensor:$bias,

      OptionalAttr<F64Attr>:$output_scale,
      OptionalAttr<I64Attr>:$output_zp
  );
  let results = (outs TIMVX_Tensor:$output);
}

def TIMVX_Pool2DOp : TIMVX_FuncTypeOp<"pool2d", [Pure]> {
  let summary = "2-D pooling (MAX / AVG / L2 / AVG_ANDROID).";
  let description = [{
    2-D pooling over a spatial window. Input is rank-4 WHCN.
    This single op covers all tim::vx::PoolType variants (MAX /
    AVG / L2 / AVG_ANDROID). Pool kind is selected by `pool_type`.

    Pad order: [left, right, top, bottom]. Stride / kernel order: [W, H].

    Lowers to tim::vx::ops::Pool2d with the given PoolType and
    RoundType::FLOOR. TIM-VX defaults to DataLayout::WHCN, which matches
    the IR-side layout, so no explicit layout argument is passed.
  }];

  let arguments = (ins
      TIMVX_Tensor:$input,
      TIMVX_PoolTypeAttr:$pool_type,
      DenseI64ArrayAttr:$kernel,         // [W, H]
      DenseI64ArrayAttr:$stride,         // [W, H]
      DenseI64ArrayAttr:$pad,            // [left, right, top, bottom]

      // Pool doesn't change the quant params; input and output share these.
      OptionalAttr<F64Attr>:$output_scale,
      OptionalAttr<I64Attr>:$output_zp
  );
  let results = (outs TIMVX_Tensor:$output);
}

//===----------------------------------------------------------------------===//
// Elementwise binary ops
//===----------------------------------------------------------------------===//

def TIMVX_MultiplyOp : TIMVX_FuncTypeOp<"multiply", [
        Pure,
        Commutative,
    ]> {
  let summary = "Elementwise multiplication (Hadamard product).";
  let description = [{
    Elementwise multiplication of input1 and input2 . Axes of size 1 are
    broadcast as needed; ranks must match. Lowers to tim::vx::ops::Multiply.
  }];

  let arguments = (ins
      TIMVX_Tensor:$input1,
      TIMVX_Tensor:$input2
  );
  let results = (outs TIMVX_Tensor:$output);
}

def TIMVX_PowOp : TIMVX_FuncTypeOp<"pow", [
        Pure,
        SameOperandsAndResultElementType
    ]> {
  let summary = "Elementwise pow(input1, input2).";
  let description = [{
    Elementwise input1 raised to the power of input2 (broadcastable).
    Lowers to tim::vx::ops::Pow.
  }];

  let arguments = (ins
      TIMVX_Tensor:$input1,
      TIMVX_Tensor:$input2
  );
  let results = (outs TIMVX_Tensor:$output);
}

def TIMVX_SubOp : TIMVX_FuncTypeOp<"sub", [
        Pure,
        SameOperandsAndResultElementType
    ]> {
  let summary = "Elementwise subtraction.";
  let description = [{
    Elementwise subtraction of input1 - input2 (broadcastable).
    Lowers to tim::vx::ops::Sub.
  }];

  let arguments = (ins
      TIMVX_Tensor:$input1,
      TIMVX_Tensor:$input2
  );
  let results = (outs TIMVX_Tensor:$output);
}

def TIMVX_AddOp : TIMVX_FuncTypeOp<"add", [
        Pure
        // No SameOperandsAndResultElementType: a quantized residual add
        // takes operands with their own (S_a, Z_a) and (S_b, Z_b), and
        // produces a result with yet-another (S_out, Z_out). The runtime
        // handles per-tensor rescaling internally — the trait's
        // exact-element-type check would reject the common case.
    ]> {
  let summary = "Elementwise addition.";
  let description = [{
    Elementwise addition of input1 + input2 (broadcastable).
    Lowers to tim::vx::ops::Add.
  }];

  let arguments = (ins
      TIMVX_Tensor:$input1,
      TIMVX_Tensor:$input2
  );
  let results = (outs TIMVX_Tensor:$output);
}

def TIMVX_MaximumOp : TIMVX_FuncTypeOp<"maximum", [
        Pure,
        Commutative,
        SameOperandsAndResultElementType
    ]> {
  let summary = "Elementwise max(input1, input2).";
  let description = [{
    Elementwise maximum of input1 and input2 (broadcastable).
    Lowers to tim::vx::ops::Maximum.

    Used by `ClampOpConversion` to decompose `tosa.clamp(x, lo, hi)`
    into `min(max(x, lo), hi)`. The decomposition deliberately avoids
    `tim::vx::ops::Clip`, whose kernel-selector pattern-matches certain
    finite ranges (`(-1, 1)` -> `relu1`, etc.) to specialised kernels
    that mis-behave on VIP9000.
  }];

  let arguments = (ins
      TIMVX_Tensor:$input1,
      TIMVX_Tensor:$input2
  );
  let results = (outs TIMVX_Tensor:$output);
}

def TIMVX_MinimumOp : TIMVX_FuncTypeOp<"minimum", [
        Pure,
        Commutative,
        SameOperandsAndResultElementType
    ]> {
  let summary = "Elementwise min(input1, input2).";
  let description = [{
    Elementwise minimum of input1 and input2 (broadcastable).
    Lowers to tim::vx::ops::Minimum. See `TIMVX_MaximumOp`.
  }];

  let arguments = (ins
      TIMVX_Tensor:$input1,
      TIMVX_Tensor:$input2
  );
  let results = (outs TIMVX_Tensor:$output);
}

//===----------------------------------------------------------------------===//
// Elementwise unary ops
//===----------------------------------------------------------------------===//

def TIMVX_RcpOp : TIMVX_FuncTypeOp<"rcp", [
        Pure,
        SameOperandsAndResultShape,
        SameOperandsAndResultElementType
    ]> {
  let summary = "Elementwise 1 / input.";
  let description = [{
    Elementwise reciprocal. Lowers to tim::vx::ops::Rcp.
  }];

  let arguments = (ins TIMVX_Tensor:$input1);
  let results = (outs TIMVX_Tensor:$output);
}

//===----------------------------------------------------------------------===//
// Shape / data-movement ops
//===----------------------------------------------------------------------===//

def TIMVX_ReshapeOp : TIMVX_FuncTypeOp<"reshape", [Pure]> {
  let summary = "Reshape a tensor to a new shape (no data movement).";
  let description = [{
    Returns a tensor with the same data as `input1` and the shape given by
    the `shape` operand (typically given by timvx.const_shape). Lowers
    to tim::vx::ops::Reshape.
  }];

  let arguments = (ins
      TIMVX_Tensor:$input1,
      1DTensorOf<[Index]>:$shape,

      // Reshape preserves quant: output mirrors input's params. Set when
      // the value is on the quant int8 chain so the produced TIM-VX
      // tensor binds with matching Quantization() (TIM-VX's op_check
      // refuses to wire a tensor with quant set into one without).
      OptionalAttr<F64Attr>:$output_scale,
      OptionalAttr<I64Attr>:$output_zp
  );
  let results = (outs TIMVX_Tensor:$output);
}

def TIMVX_SliceOp : TIMVX_FuncTypeOp<"slice", [Pure]> {
  let summary = "Extract a contiguous slice from a tensor.";
  let description = [{
    Extract a slice beginning at `start` of length `size` from `input1`.
    Both `start` and `size` are 1-D shape operands (typically given by
    timvx.const_shape). Lowers to tim::vx::ops::Slice.
  }];

  let arguments = (ins
      TIMVX_Tensor:$input1,
      1DTensorOf<[Index]>:$start,
      1DTensorOf<[Index]>:$size,

      OptionalAttr<F64Attr>:$output_scale,
      OptionalAttr<I64Attr>:$output_zp
  );
  let results = (outs TIMVX_Tensor:$output);
}

def TIMVX_TransposeOp : TIMVX_FuncTypeOp<"transpose", [
        Pure,
        AllElementTypesMatch<["input1", "output"]>
    ]> {
  let summary = "Permute tensor dimensions.";
  let description = [{
    Permute the dimensions of `input1` according to `perms`. Each entry of
    `perms` must be a unique valid axis index of the input. Lowers to
    tim::vx::ops::Transpose.
  }];

  let arguments = (ins
      TIMVX_Tensor:$input1,
      DenseI32ArrayAttr:$perms,

      OptionalAttr<F64Attr>:$output_scale,
      OptionalAttr<I64Attr>:$output_zp
  );
  let results = (outs TIMVX_Tensor:$output);

  // Canonicalizers:
  //   * `transpose(const)` -> `const` with permuted dense data. Required
  //     for conv2d/pool2d weights: TIM-VX's NN-engine kernel selector
  //     refuses to bind a CONSTANT weight tensor through a runtime
  //     transpose, so the IREE-style wrapper would otherwise compile-fail
  //     for any conv whose weight didn't get pre-permuted at compile time.
  //   * `transpose(transpose)` -> single transpose (composed perms).
  //     Coalesces the wrapping pairs introduced by the spatial-op
  //     converters when adjacent ops both wrap.
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Quant / dtype helpers
//===----------------------------------------------------------------------===//

def TIMVX_ReduceSumOp : TIMVX_FuncTypeOp<"reduce_sum", [Pure]> {
  let summary = "Sum-reduce along one axis (or many).";
  let description = [{
    Lowers to tim::vx::ops::ReduceSum. Used to handle the post-residual
    spatial sum that quantized TOSA emits in fp32 (paired with multiply by
    `1/N` for an avg pool).
  }];

  let arguments = (ins
      TIMVX_Tensor:$input,
      DenseI32ArrayAttr:$axes,    // axis indices to reduce over
      BoolAttr:$keep_dims
  );
  let results = (outs TIMVX_Tensor:$output);
}

def TIMVX_DataConvertOp : TIMVX_FuncTypeOp<"dataconvert", [
        Pure,
        SameOperandsAndResultShape
    ]> {
  let summary = "Requantize between two int dtypes (or same-dtype rescale).";
  let description = [{
    Lowers to tim::vx::ops::DataConvert. The runtime applies the
    (input scale,zp) → (output scale,zp) projection in storage space, so
    DataConvert is the right tool for int↔int requantize and for the
    int→f32 dequantize boundary. It is NOT the right tool for f32→int —
    that direction COMPILE_FAILs on VIP9000 on this chip even
    though `vsi_nn_op_dataconvert.c`'s op_check table claims it is
    accepted; use `timvx.cast` for any f32 boundary.
  }];

  let arguments = (ins
      TIMVX_Tensor:$input,
      OptionalAttr<F64Attr>:$output_scale,
      OptionalAttr<I64Attr>:$output_zp
  );
  let results = (outs TIMVX_Tensor:$output);
}

def TIMVX_CastOp : TIMVX_FuncTypeOp<"cast", [
        Pure,
        SameOperandsAndResultShape
    ]> {
  let summary = "Value-cast between dtypes; ignores any input/output scale/zp.";
  let description = [{
    Lowers to tim::vx::ops::Cast. The op runs the GPU `cast` kernel
    (via vsi_nn_kernel_selector) which performs a real-value-preserving
    cast: it does NOT apply the input or output (scale, zp) — those are
    metadata for downstream ops. Use this to lower `tosa.cast` for any
    pair where fp32 is on either side (`f32↔{u8|asym, i8|asym, i32 raw}`)
    and for `i32_raw → {u8|asym, i8|asym}` rescale-tail conversions.
    Any optional output (scale, zp) is attached to the output tensor's
    spec so consumers see the right metadata, but does not affect the
    cast's numeric behavior.
  }];

  let arguments = (ins
      TIMVX_Tensor:$input,
      OptionalAttr<F64Attr>:$output_scale,
      OptionalAttr<I64Attr>:$output_zp
  );
  let results = (outs TIMVX_Tensor:$output);
}

def TIMVX_PadOp : TIMVX_FuncTypeOp<"pad", [Pure]> {
  let summary = "Pad with a constant value (typically the input zero-point).";
  let description = [{
    Pads `input` along each spatial axis with `pad_const`. `padding` is a
    rank-2 i32/i64 tensor with shape [rank, 2] containing
    [front, back] sizes per dim. Lowers to tim::vx::ops::Pad.

    Quantized models commonly precede a conv2d with this op carrying the
    input's zero-point as `pad_const` so that the padded region is zero in
    real-value space.
  }];

  let arguments = (ins
      TIMVX_Tensor:$input,
      // Flat [front_0, back_0, front_1, back_1, ...] with `2 * rank(input)`
      // entries; matches what `tosa.const_shape` emits after the generic
      // ConstShape lowering reshapes its !tosa.shape<2N> result.
      1DTensorOf<[Index]>:$padding,
      F32Attr:$pad_const,

      // Pad doesn't change the quant params; output mirrors input.
      OptionalAttr<F64Attr>:$output_scale,
      OptionalAttr<I64Attr>:$output_zp
  );
  let results = (outs TIMVX_Tensor:$output);
}

#endif // TIMVX_OPS
