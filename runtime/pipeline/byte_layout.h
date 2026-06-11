// pipeline/byte_layout.h — byte-layout primitives shared by runner pipeline.
//
// Two utilities:
//   * `bytesPerElem(dt)`   — element-size lookup keyed on `tim::vx::DataType`.
//   * `layoutConvert(...)` — repacks bytes between MLIR row-major (the
//     `input.bin` convention every preprocess in this repo emits) and
//     TIM-VX innermost-first (the convention CreateTensor / CopyDataTo /
//     transpose / conv / pool kernels walk internally).
//
// Both are header-only; they're invoked from every PreProcessor /
// PostProcessor and from the bench / eval loops, so inlining at the
// call site avoids extra TU dependencies.
//
// The layout flip exists ONLY at the harness boundary — the MLIR pipeline
// and the lowered TIM-VX graph stay innocent of the convention mismatch.
// See `runner_main.cpp.tpl`'s historical comment for the simple_v1
// regression that motivated centralising this.

#ifndef TIMVX_PIPELINE_BYTE_LAYOUT_H
#define TIMVX_PIPELINE_BYTE_LAYOUT_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "tim/vx/types.h"

namespace timvx_pipeline {

inline size_t bytesPerElem(tim::vx::DataType dt) {
  switch (dt) {
    case tim::vx::DataType::FLOAT16:
    case tim::vx::DataType::INT16:
    case tim::vx::DataType::UINT16: return 2;
    case tim::vx::DataType::INT8:
    case tim::vx::DataType::UINT8:
    case tim::vx::DataType::BOOL8:  return 1;
    case tim::vx::DataType::INT64:  return 8;
    default:                         return 4;  // FLOAT32 / INT32 / UINT32
  }
}

// Repack `src` (numel * elem_size bytes) into `dst`, preserving the
// (i_0, …, i_{R-1}) coord→value mapping but swapping the in-memory
// stride convention.
//
// `from_mlir_to_tvx = true`  : src is MLIR row-major, dst is TIM-VX innermost-first.
// `from_mlir_to_tvx = false` : src is TIM-VX innermost-first, dst is MLIR row-major.
inline void layoutConvert(const uint8_t* src, uint8_t* dst,
                           const std::vector<uint32_t>& shape,
                           size_t elem_size, bool from_mlir_to_tvx) {
  size_t rank = shape.size();
  size_t numel = 1;
  for (auto d : shape) numel *= d;
  if (rank <= 1) {
    std::memcpy(dst, src, numel * elem_size);
    return;
  }
  // mlir_strides[k] = product(shape[k+1..rank))  — rightmost is fastest.
  // tvx_strides[k]  = product(shape[0..k))       — leftmost is fastest.
  std::vector<size_t> mlir_strides(rank), tvx_strides(rank);
  mlir_strides[rank - 1] = 1;
  for (int k = static_cast<int>(rank) - 2; k >= 0; --k)
    mlir_strides[k] = mlir_strides[k + 1] * shape[k + 1];
  tvx_strides[0] = 1;
  for (size_t k = 1; k < rank; ++k)
    tvx_strides[k] = tvx_strides[k - 1] * shape[k - 1];

  std::vector<size_t> idx(rank, 0);
  for (size_t e = 0; e < numel; ++e) {
    size_t mlir_off = 0, tvx_off = 0;
    for (size_t k = 0; k < rank; ++k) {
      mlir_off += idx[k] * mlir_strides[k];
      tvx_off  += idx[k] * tvx_strides[k];
    }
    if (from_mlir_to_tvx) {
      std::memcpy(dst + tvx_off * elem_size,
                  src + mlir_off * elem_size, elem_size);
    } else {
      std::memcpy(dst + mlir_off * elem_size,
                  src + tvx_off * elem_size, elem_size);
    }
    // Iterate idx in MLIR row-major order (least-significant first).
    for (int k = static_cast<int>(rank) - 1; k >= 0; --k) {
      if (++idx[k] < shape[k]) break;
      idx[k] = 0;
    }
  }
}

} // namespace timvx_pipeline

#endif // TIMVX_PIPELINE_BYTE_LAYOUT_H
