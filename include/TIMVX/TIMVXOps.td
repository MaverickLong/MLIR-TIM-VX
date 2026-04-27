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

def TIMVX_ConstOp : TIMVX_Op<"const", [
        ConstantLike, Pure,
        AllShapesMatch<["values", "output"]>,
        FirstAttrDerivedResultType
    ]> {
  let summary = "Constant tensor.";
  let description = [{
    A constant tensor for use as input to other ops. At runtime emission,
    lowers to a TIM-VX tensor created with TensorAttribute::CONSTANT and
    the provided data.
  }];

  let arguments = (ins ElementsAttr:$values);
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
        SameOperandsAndResultShape,
        SameOperandsAndResultElementType
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
  let summary = "2-D convolution (NHWC input, OHWI weight).";
  let description = [{
    Performs 2-D convolution. Input is rank-4 NHWC; weight is rank-4 OHWI;
    bias is rank-1.

    Pad order: [top, bottom, left, right].
    Stride / dilation order: [h, w].

    Lowers to tim::vx::ops::Conv2d with constant pad / stride / dilation
    arrays and DataLayout::CWHN (input) / DataLayout::IcWHOc (kernel).
  }];

  let arguments = (ins
      TIMVX_Tensor:$input,
      TIMVX_Tensor:$weight,
      TIMVX_Tensor:$bias,

      DenseI64ArrayAttr:$pad,            // [top, bottom, left, right]
      DenseI64ArrayAttr:$stride,         // [h, w]
      DenseI64ArrayAttr:$dilation        // [h, w]
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

def TIMVX_Pool2DOp : TIMVX_FuncTypeOp<"pool2d", [Pure]> {
  let summary = "2-D pooling (MAX / AVG / L2 / AVG_ANDROID).";
  let description = [{
    2-D pooling over a spatial window. Input is rank-4 NHWC.
    This single op covers all tim::vx::PoolType variants (MAX /
    AVG / L2 / AVG_ANDROID). Pool kind is selected by `pool_type`.

    Pad order: [top, bottom, left, right]. Stride / kernel order: [h, w].

    Lowers to tim::vx::ops::Pool2d with the given PoolType,
    RoundType::FLOOR, and DataLayout::CWHN.
  }];

  let arguments = (ins
      TIMVX_Tensor:$input,
      TIMVX_PoolTypeAttr:$pool_type,
      DenseI64ArrayAttr:$kernel,         // [h, w]
      DenseI64ArrayAttr:$stride,         // [h, w]
      DenseI64ArrayAttr:$pad             // [top, bottom, left, right]
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
        Pure,
        SameOperandsAndResultElementType
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
      1DTensorOf<[Index]>:$shape
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
      1DTensorOf<[Index]>:$size
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
      DenseI32ArrayAttr:$perms
  );
  let results = (outs TIMVX_Tensor:$output);
}

#endif // TIMVX_OPS
