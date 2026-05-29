// pipeline/tensor_io.h — read TIM-VX output tensors and reduce to common
// host-side forms.
//
//   * `dequantize_output(tensor, &out)` copies the device-side bytes,
//     layout-flips TIM-VX → MLIR row-major, and dequantizes to f32.
//     Common between one-shot print, serve top-K, and benchmark runs.
//   * `compute_topk(buf, K)` returns the K largest (idx, value) pairs.

#ifndef TIMVX_PIPELINE_TENSOR_IO_H
#define TIMVX_PIPELINE_TENSOR_IO_H

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "tim/vx/tensor.h"
#include "tim/vx/types.h"

#include "pipeline/byte_layout.h"

namespace timvx_pipeline {

// Returns false only on CopyDataFromTensor failure or an unsupported
// output dtype.
inline bool dequantize_output(const std::shared_ptr<tim::vx::Tensor>& t,
                               std::vector<float>& out_buf) {
  const auto& shape = t->GetShape();
  size_t numel = 1;
  for (auto d : shape) numel *= d;
  out_buf.assign(numel, 0.0f);
  if (numel == 0) return true;

  auto dtype = t->GetDataType();
  auto quant = t->GetQuantization();
  float scale = 1.0f;
  int32_t zp = 0;
  if (quant.Type() != tim::vx::QuantType::NONE && !quant.Scales().empty()) {
    scale = quant.Scales()[0];
    if (!quant.ZeroPoints().empty()) zp = quant.ZeroPoints()[0];
  }

  size_t elem = bytesPerElem(dtype);
  std::vector<uint8_t> tvx_bytes(numel * elem);
  if (!t->CopyDataFromTensor(tvx_bytes.data())) return false;
  std::vector<uint8_t> mlir_bytes(numel * elem);
  layoutConvert(tvx_bytes.data(), mlir_bytes.data(), shape, elem,
                /*from_mlir_to_tvx=*/false);

  if (dtype == tim::vx::DataType::FLOAT32) {
    std::memcpy(out_buf.data(), mlir_bytes.data(), numel * sizeof(float));
  } else if (dtype == tim::vx::DataType::INT8) {
    auto* raw = reinterpret_cast<const int8_t*>(mlir_bytes.data());
    for (size_t i = 0; i < numel; ++i)
      out_buf[i] = (static_cast<float>(raw[i]) - static_cast<float>(zp)) * scale;
  } else if (dtype == tim::vx::DataType::UINT8) {
    auto* raw = mlir_bytes.data();
    for (size_t i = 0; i < numel; ++i)
      out_buf[i] = (static_cast<float>(raw[i]) - static_cast<float>(zp)) * scale;
  } else {
    return false;
  }
  return true;
}

using TopKEntry = std::pair<uint32_t, float>;

// Top-K (largest score first) via partial_sort on indices.
inline std::vector<TopKEntry> compute_topk(const std::vector<float>& buf,
                                            size_t K) {
  K = std::min(K, buf.size());
  std::vector<uint32_t> idx(buf.size());
  std::iota(idx.begin(), idx.end(), 0u);
  std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                    [&](uint32_t a, uint32_t b){ return buf[a] > buf[b]; });
  std::vector<TopKEntry> out;
  out.reserve(K);
  for (size_t i = 0; i < K; ++i) out.emplace_back(idx[i], buf[idx[i]]);
  return out;
}

} // namespace timvx_pipeline

#endif // TIMVX_PIPELINE_TENSOR_IO_H
