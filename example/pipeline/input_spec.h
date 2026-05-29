// pipeline/input_spec.h — the `InputSpec` struct used to declare the model's
// non-graph function arguments. `runner_main.cpp.tpl` defines an auto-
// generated `kInputs` vector of these (one per function argument), parsed
// from the timvx MLIR's `func.func` signature by the lowering script.
//
// Separated into its own header so PreProcessor / PostProcessor / serve
// can read the spec without dragging the rest of the runner.

#ifndef TIMVX_PIPELINE_INPUT_SPEC_H
#define TIMVX_PIPELINE_INPUT_SPEC_H

#include <cstdint>
#include <vector>

#include "tim/vx/types.h"

namespace timvx_pipeline {

struct InputSpec {
  // MLIR text order (left-to-right = N,C,H,W or N,H,W,C). NOT reversed
  // into TIM-VX innermost-first convention — the lowering pass inserts
  // an explicit transpose immediately after the function-argument tensor
  // that re-maps to TIM-VX's WHCN view, so the input tensor itself
  // carries the MLIR shape verbatim.
  std::vector<uint32_t> shape;
  tim::vx::DataType     dtype;
  // For quantized models: the (scale, zp) the network was calibrated for.
  // Set to (0.0, 0) to mean "no quant" — the runner skips Quantization()
  // on the spec in that case.
  double  quant_scale = 0.0;
  int32_t quant_zp    = 0;
};

} // namespace timvx_pipeline

#endif // TIMVX_PIPELINE_INPUT_SPEC_H
