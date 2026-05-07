# MLIR-TIM-VX

Lowering Path from MLIR to the TIM-VX backend for VeriSilicon NPUs

## Prerequisites

This repository is only tested on a Allwinner A733 \w Vivante VIP9000 on a Radxa Cubie A7Z. I cannot guarantee the compatibility with other SoCs due to the different subset of operations each NPU supports.

This project serves as the lowering pipeline from `tosa` v1 to the custom `timvx` dialect. It also emits the `timvx` dialect to EmitC backend to generate compilable TIM-VX C++ code. It does NOT include lowering from other upstreams to TOSA, although a path from TFLite to TOSA will be relatively straightforward.

The lowered TOSA dialect must already been signed int8 quantized. Passing an unquantized model to the dialect works, but the graph will not compile at JIT time.

To make the compiler work on-device (and to setup the environment), please refer to [Radxa-A733-NPU-Unified-Driver-Support-Package](https://github.com/MaverickLong/Radxa-A733-NPU-Unified-Driver-Support-Package)

## How to

The files in `example/` can give you a quick idea of how the pipeline works.

A very simple sint8 quantized TOSAv1 example at `sample/sample_cnn.mlir` has been given to help you verify the compiler works end-to-end. It takes in ImageNet-like input (sint8 quantized) and produces 1000-class result (it's garbage data).

## Limitations

Expect a massive amount of issues.

- There is currenly no support for Transformers etc., so you cannot run LLM on the thing yet. Even though I might add Transformers support later, don't expect it to be good --- the NPU is simply not that powerful.

- There will be numerous issues due to specific lowerings from TFLite or otherwise -- I only implemented the lowering from a very limited subset of the TOSA specification.

- I have not tested other NPUs, but the VIP9000 on Allwinner A733 only supports UINT8 / SINT8. There is 100% no way to make unquantized model work due to the NN module on the NPU being INT8 only ASIC. Expect accuracy losses.

- I only implemented the `tosa` -> `timvx` lowering, and I do not plan to extend to other dialects like `linalg` like IREE does. Your upstream has to be TOSA v1 compatible.

## Thoughts

It is a total pain working on this. It is supposed to be a submodule of my Master's thesis project but ended being a massive (from a personal POV) standalone project that probably can benefit most people.

It all starts with Radxa's (or potentially Allwinner's or VeriSilicon's) false advertisement of the NPU being capable of lowering from MLIR -- turns out that thing just don't exist, so I had to make it myself.
