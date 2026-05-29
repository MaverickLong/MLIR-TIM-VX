// pipeline/image.h — bilinear resize for 8-bit-per-channel RGB888
// images. Operates on the `RgbImage` from `jpeg.h`.
//
// The implementation walks the destination grid, computes each
// destination pixel's float coordinate in source space via
// `src = (dst + 0.5) * src_dim / dst_dim - 0.5`  (the half-pixel-
// centered convention TF / Pegasus / TIM-VX `ResizeType::BILINEAR`
// with `half_pixel_centers=true` use), interpolates between the four
// surrounding source pixels, and rounds to u8.
//
// Performance note: scalar loop in C++ — for a 224×224×3 output from a
// typical 480×360×3 input the cost is ~150 µs on the A733's big core.
// That's small enough to easily fold into one NPU inference's worth of
// time (7.5 ms for resnet50), which is the point of the streaming
// pipeline.

#ifndef TIMVX_PIPELINE_IMAGE_H
#define TIMVX_PIPELINE_IMAGE_H

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <vector>

#include "pipeline/jpeg.h"

namespace timvx_pipeline {

inline RgbImage resize_bilinear(const RgbImage& src,
                                int dst_w, int dst_h) {
  RgbImage out;
  out.width = dst_w;
  out.height = dst_h;
  out.data.assign(static_cast<size_t>(dst_w) * dst_h * 3, 0);
  if (src.width <= 0 || src.height <= 0 || dst_w <= 0 || dst_h <= 0)
    return out;

  const double sx_scale = static_cast<double>(src.width)  / dst_w;
  const double sy_scale = static_cast<double>(src.height) / dst_h;
  const int max_x = src.width  - 1;
  const int max_y = src.height - 1;

  for (int dy = 0; dy < dst_h; ++dy) {
    // Half-pixel-centered: take the center of the destination pixel and
    // map back to the source grid's pixel-center coords.
    double sy = (dy + 0.5) * sy_scale - 0.5;
    if (sy < 0) sy = 0;
    if (sy > max_y) sy = max_y;
    int y0 = static_cast<int>(std::floor(sy));
    int y1 = std::min(y0 + 1, max_y);
    double wy = sy - y0;

    const uint8_t* row0 = src.data.data() + y0 * src.width * 3;
    const uint8_t* row1 = src.data.data() + y1 * src.width * 3;
    uint8_t* drow = out.data.data() + dy * dst_w * 3;

    for (int dx = 0; dx < dst_w; ++dx) {
      double sx = (dx + 0.5) * sx_scale - 0.5;
      if (sx < 0) sx = 0;
      if (sx > max_x) sx = max_x;
      int x0 = static_cast<int>(std::floor(sx));
      int x1 = std::min(x0 + 1, max_x);
      double wx = sx - x0;

      const uint8_t* p00 = row0 + x0 * 3;
      const uint8_t* p01 = row0 + x1 * 3;
      const uint8_t* p10 = row1 + x0 * 3;
      const uint8_t* p11 = row1 + x1 * 3;
      uint8_t* dp = drow + dx * 3;
      for (int c = 0; c < 3; ++c) {
        double top    = p00[c] * (1.0 - wx) + p01[c] * wx;
        double bottom = p10[c] * (1.0 - wx) + p11[c] * wx;
        double v      = top    * (1.0 - wy) + bottom * wy;
        long iv = std::lround(v);
        if (iv < 0) iv = 0;
        if (iv > 255) iv = 255;
        dp[c] = static_cast<uint8_t>(iv);
      }
    }
  }
  return out;
}

} // namespace timvx_pipeline

#endif // TIMVX_PIPELINE_IMAGE_H
