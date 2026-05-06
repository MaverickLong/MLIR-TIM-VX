# MLIR-TIM-VX

Lowering Path from MLIR to the TIM-VX backend for VeriSilicon NPUs

## TODO

Conv2D -> BatchNorm folding

Quantization Handling, UINT8 for Conv layers

## NHWC -> WHCN

I have an idea. We add a tensor attribute field _layout_, a list[int|char].

Prerequisite:

- We respect TOSA input as ground truth (assume source assembly layout is correct)
- At this stage, we assume a tensor will never be used for different purposes (e.g. if it is a weight, it is a weight; it's never going to be a input of a conv layer or a bias)

Steps:

1. we tag all TOSA conv / pool2d operand as, NHWC for conv/pool input/output, and OHWI for kernel. (similar for conv1d / 3d)
2. greedy growing (liveliness analysis): since we know we still have SSA, we "grow" and infer the shape of the values that produces the operand, and where the output is used again as the operand. If it is a broadcasting operation, like one is NHWC and the other is just C, tag the original value as 1, 1, 1, C. We grow for ALL element-wise operations until we reach another spaicial operation where the output is already tagged, or if it is a operation that we cannot infer the layout from (e.g. a matrix multiplication). We now have a "boundary" of NHWC region, and we know exactly where we actually cares the layout. This should be O(3) if we use a liveliness analysis similar to a normal compiler liveliness analysis.
3. Now we actually do the lowering to timvx. For each NHWC constant, we convert to WHCN. For each SSA value at the boundary (whether where it is produced (if produced by another op) or where it is used as a operand), add a transpose.
4. We now have a strictly fenced NCHW/NHWC to WHCN conversion.
