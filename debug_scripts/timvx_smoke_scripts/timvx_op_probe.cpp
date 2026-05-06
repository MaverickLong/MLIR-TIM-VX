// timvx_op_probe.cpp — probe which (op, dtype) combinations the NPU
// will actually compile and run on this chip (VIP9000Nano-DI, A733).
//
// For each entry in the table, we build the smallest possible graph
// that exercises the op with that dtype, try Compile() + Run(), and
// print one of:
//
//   PASS        — compile + run both succeeded
//   COMPILE_FAIL — Graph::Compile returned false (typical for missing
//                  NN-engine kernels: "vivante.nn.<op>.layer" status=-1)
//   RUN_FAIL    — compile ok but Run failed
//
// We do not check numerical correctness here — see timvx_matmul.cpp /
// timvx_smoke.cpp for that. The point of *this* binary is to map out
// which ops are NN-engine (quantized-only on this Nano) vs shader
// (work with fp32).
//
// Build/run:  bash timvx_op_probe.bash

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <memory>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"
#include "tim/vx/types.h"
#include "tim/vx/ops/clip.h"              // Clip
#include "tim/vx/ops/conv2d.h"           // Conv2d
#include "tim/vx/ops/elementwise.h"      // Add, Sub, Multiply, Pow
#include "tim/vx/ops/fullyconnected.h"
#include "tim/vx/ops/matmul.h"
#include "tim/vx/ops/pool2d.h"
#include "tim/vx/ops/reduce.h"
#include "tim/vx/ops/reshape.h"
#include "tim/vx/ops/simple_operations.h"  // Rcp
#include "tim/vx/ops/slice.h"
#include "tim/vx/ops/transpose.h"
#include "third_party/half/half.hpp"

using DT = tim::vx::DataType;
using TA = tim::vx::TensorAttribute;

namespace {

// Dt::S8  — asymmetric signed int8 (tflite/quant.uniform<i8> default).
// Dt::S8P — symmetric per-channel int8 weights (tflite's PTQ default for
//           conv weights). Only applied to weight tensors; activations and
//           biases for an S8P probe still use Dt::S8.
enum class Dt { FP32, FP16, U8, S8, S8P };
const char* dt_name(Dt d) {
  switch (d) {
    case Dt::FP32: return "fp32";
    case Dt::FP16: return "fp16";
    case Dt::U8:   return "u8";
    case Dt::S8:   return "s8";
    case Dt::S8P:  return "s8pc";
  }
  return "?";
}
DT dt_to_vx(Dt d) {
  switch (d) {
    case Dt::FP32: return DT::FLOAT32;
    case Dt::FP16: return DT::FLOAT16;
    case Dt::U8:   return DT::UINT8;
    case Dt::S8:
    case Dt::S8P:  return DT::INT8;
  }
  return DT::FLOAT32;
}
size_t dt_bytes(Dt d) {
  switch (d) {
    case Dt::U8: case Dt::S8: case Dt::S8P: return 1;
    case Dt::FP16:                          return 2;
    case Dt::FP32:                          return 4;
  }
  return 4;
}
bool dt_is_int8(Dt d) { return d == Dt::U8 || d == Dt::S8 || d == Dt::S8P; }

// Build a TensorSpec, attaching ASYMMETRIC quant params for activations
// and SYMMETRIC_PER_CHANNEL when caller asks for it (S8P weight only).
tim::vx::TensorSpec make_spec(Dt d, const tim::vx::ShapeType& sh, TA attr,
                              bool per_channel_weight = false) {
  if (d == Dt::U8) {
    tim::vx::Quantization q(tim::vx::QuantType::ASYMMETRIC, 2.0f / 255.0f, 128);
    return tim::vx::TensorSpec(DT::UINT8, sh, attr, q);
  }
  if (d == Dt::S8) {
    tim::vx::Quantization q(tim::vx::QuantType::ASYMMETRIC, 2.0f / 255.0f, 0);
    return tim::vx::TensorSpec(DT::INT8, sh, attr, q);
  }
  if (d == Dt::S8P) {
    if (per_channel_weight) {
      // Per-channel symmetric: one (scale, zp=0) per output channel. The
      // probe's conv weight has shape {KW, KH, IC, OC} = OC at index 3.
      uint32_t oc = sh.size() == 4 ? sh[3] : (sh.empty() ? 1 : sh.back());
      std::vector<float> scales(oc, 1.0f / 127.0f);
      std::vector<int32_t> zps(oc, 0);
      tim::vx::Quantization q(tim::vx::QuantType::SYMMETRIC_PER_CHANNEL,
                              /*channel_dim=*/3, std::move(scales),
                              std::move(zps));
      return tim::vx::TensorSpec(DT::INT8, sh, attr, q);
    }
    // Activations / non-weight tensors in an S8P probe: plain asymmetric S8.
    tim::vx::Quantization q(tim::vx::QuantType::ASYMMETRIC, 2.0f / 255.0f, 0);
    return tim::vx::TensorSpec(DT::INT8, sh, attr, q);
  }
  return tim::vx::TensorSpec(dt_to_vx(d), sh, attr);
}
tim::vx::TensorSpec make_bias_spec(Dt d, const tim::vx::ShapeType& sh) {
  if (dt_is_int8(d)) {
    // bias scale = input_scale * weight_scale; for our probe activations
    // use 2/255, weights use either 2/255 (asymmetric) or 1/127
    // (per-channel symmetric). This mismatch on S8P is fine for a probe —
    // we only care whether the kernel instantiates.
    float a_scale = 2.0f / 255.0f;
    float w_scale = (d == Dt::S8P) ? (1.0f / 127.0f) : (2.0f / 255.0f);
    tim::vx::Quantization q(tim::vx::QuantType::ASYMMETRIC,
                            a_scale * w_scale, 0);
    return tim::vx::TensorSpec(DT::INT32, sh, TA::CONSTANT, q);
  }
  return tim::vx::TensorSpec(dt_to_vx(d), sh, TA::CONSTANT);
}

// Fill a buffer with some neutral, dtype-appropriate constant (0.5/64).
// For probes we don't care about values; we just need the kernel to
// instantiate.
std::vector<uint8_t> filler(Dt d, size_t numel, bool small_nonzero = true) {
  std::vector<uint8_t> buf(numel * dt_bytes(d));
  if (d == Dt::U8) {
    std::memset(buf.data(), small_nonzero ? 140 : 128, buf.size());
  } else if (d == Dt::S8 || d == Dt::S8P) {
    std::memset(buf.data(), small_nonzero ? 12 : 0, buf.size());
  } else if (d == Dt::FP16) {
    half_float::half v = small_nonzero ? half_float::half(0.5f)
                                       : half_float::half(0.0f);
    auto* p = reinterpret_cast<half_float::half*>(buf.data());
    for (size_t i = 0; i < numel; ++i) p[i] = v;
  } else {
    float v = small_nonzero ? 0.5f : 0.0f;
    auto* p = reinterpret_cast<float*>(buf.data());
    for (size_t i = 0; i < numel; ++i) p[i] = v;
  }
  return buf;
}
std::vector<int32_t> zeros_i32(size_t n) { return std::vector<int32_t>(n, 0); }

enum class Status { COMPILE_FAIL = 1, RUN_FAIL = 2, PASS = 0, CRASH = 3 };
const char* status_name(Status s) {
  switch (s) {
    case Status::PASS:         return "PASS";
    case Status::COMPILE_FAIL: return "COMPILE_FAIL";
    case Status::RUN_FAIL:     return "RUN_FAIL";
    case Status::CRASH:        return "CRASH";
  }
  return "?";
}
struct Result {
  bool compile_ok;
  bool run_ok;
};

// Run a builder lambda that constructs the graph and returns it; then
// time-share Compile + Run. Returns {compile_ok, run_ok}.
Result probe(std::function<std::shared_ptr<tim::vx::Graph>(
                 std::shared_ptr<tim::vx::Context>)> build) {
  auto ctx = tim::vx::Context::Create();
  if (!ctx) return {false, false};
  auto g = build(ctx);
  if (!g) return {false, false};
  bool c = g->Compile();
  if (!c) return {false, false};
  bool r = g->Run();
  return {true, r};
}

// ── Per-op probes ──────────────────────────────────────────────────
// All shapes are inner-first (TIM-VX convention).

Result probe_add(Dt d) {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType sh{4};
    auto a = g->CreateTensor(make_spec(d, sh, TA::INPUT));
    auto b = g->CreateTensor(make_spec(d, sh, TA::INPUT));
    auto c = g->CreateTensor(make_spec(d, sh, TA::OUTPUT));
    auto buf = filler(d, 4);
    a->CopyDataToTensor(buf.data(), buf.size());
    b->CopyDataToTensor(buf.data(), buf.size());
    auto op = g->CreateOperation<tim::vx::ops::Add>();
    (*op).BindInputs({a, b}).BindOutputs({c});
    return g;
  });
}
Result probe_sub(Dt d) {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType sh{4};
    auto a = g->CreateTensor(make_spec(d, sh, TA::INPUT));
    auto b = g->CreateTensor(make_spec(d, sh, TA::INPUT));
    auto c = g->CreateTensor(make_spec(d, sh, TA::OUTPUT));
    auto buf = filler(d, 4);
    a->CopyDataToTensor(buf.data(), buf.size());
    b->CopyDataToTensor(buf.data(), buf.size());
    auto op = g->CreateOperation<tim::vx::ops::Sub>();
    (*op).BindInputs({a, b}).BindOutputs({c});
    return g;
  });
}
Result probe_mul(Dt d) {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType sh{4};
    auto a = g->CreateTensor(make_spec(d, sh, TA::INPUT));
    auto b = g->CreateTensor(make_spec(d, sh, TA::INPUT));
    auto c = g->CreateTensor(make_spec(d, sh, TA::OUTPUT));
    auto buf = filler(d, 4);
    a->CopyDataToTensor(buf.data(), buf.size());
    b->CopyDataToTensor(buf.data(), buf.size());
    auto op = g->CreateOperation<tim::vx::ops::Multiply>(1.0f);
    (*op).BindInputs({a, b}).BindOutputs({c});
    return g;
  });
}
Result probe_clip(Dt d) {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType sh{4};
    auto a = g->CreateTensor(make_spec(d, sh, TA::INPUT));
    auto c = g->CreateTensor(make_spec(d, sh, TA::OUTPUT));
    auto buf = filler(d, 4);
    a->CopyDataToTensor(buf.data(), buf.size());
    auto op = g->CreateOperation<tim::vx::ops::Clip>(-1.0f, 1.0f);
    (*op).BindInputs({a}).BindOutputs({c});
    return g;
  });
}
Result probe_pow(Dt d) {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType sh{4};
    auto a = g->CreateTensor(make_spec(d, sh, TA::INPUT));
    auto b = g->CreateTensor(make_spec(d, sh, TA::INPUT));
    auto c = g->CreateTensor(make_spec(d, sh, TA::OUTPUT));
    auto buf = filler(d, 4);
    a->CopyDataToTensor(buf.data(), buf.size());
    b->CopyDataToTensor(buf.data(), buf.size());
    auto op = g->CreateOperation<tim::vx::ops::Pow>();
    (*op).BindInputs({a, b}).BindOutputs({c});
    return g;
  });
}
Result probe_rcp(Dt d) {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType sh{4};
    auto a = g->CreateTensor(make_spec(d, sh, TA::INPUT));
    auto c = g->CreateTensor(make_spec(d, sh, TA::OUTPUT));
    auto buf = filler(d, 4);
    a->CopyDataToTensor(buf.data(), buf.size());
    auto op = g->CreateOperation<tim::vx::ops::Rcp>();
    (*op).BindInputs({a}).BindOutputs({c});
    return g;
  });
}
Result probe_reshape(Dt d) {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType in_sh{4};
    tim::vx::ShapeType out_sh{2, 2};
    auto a = g->CreateTensor(make_spec(d, in_sh, TA::INPUT));
    auto c = g->CreateTensor(make_spec(d, out_sh, TA::OUTPUT));
    auto buf = filler(d, 4);
    a->CopyDataToTensor(buf.data(), buf.size());
    std::vector<uint32_t> shape{2, 2};
    auto op = g->CreateOperation<tim::vx::ops::Reshape>(shape);
    (*op).BindInputs({a}).BindOutputs({c});
    return g;
  });
}
Result probe_slice(Dt d) {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType in_sh{4};
    tim::vx::ShapeType out_sh{2};
    auto a = g->CreateTensor(make_spec(d, in_sh, TA::INPUT));
    auto c = g->CreateTensor(make_spec(d, out_sh, TA::OUTPUT));
    auto buf = filler(d, 4);
    a->CopyDataToTensor(buf.data(), buf.size());
    auto op = g->CreateOperation<tim::vx::ops::Slice>(
        /*dims=*/1u, std::vector<int32_t>{1}, std::vector<int32_t>{2});
    (*op).BindInputs({a}).BindOutputs({c});
    return g;
  });
}
Result probe_transpose(Dt d) {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType in_sh{3, 4};
    tim::vx::ShapeType out_sh{4, 3};
    auto a = g->CreateTensor(make_spec(d, in_sh, TA::INPUT));
    auto c = g->CreateTensor(make_spec(d, out_sh, TA::OUTPUT));
    auto buf = filler(d, 12);
    a->CopyDataToTensor(buf.data(), buf.size());
    auto op = g->CreateOperation<tim::vx::ops::Transpose>(
        std::vector<uint32_t>{1, 0});
    (*op).BindInputs({a}).BindOutputs({c});
    return g;
  });
}
Result probe_conv2d(Dt d) {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType in_sh{8, 8, 1, 1};   // W H C N (inner-first)
    tim::vx::ShapeType w_sh {3, 3, 1, 4};   // KW KH IC OC
    tim::vx::ShapeType b_sh {4};
    tim::vx::ShapeType out_sh{6, 6, 4, 1};
    auto a = g->CreateTensor(make_spec(d, in_sh, TA::INPUT));
    auto w_buf = filler(d, 36);
    auto w = g->CreateTensor(
        tim::vx::TensorSpec(make_spec(d, w_sh, TA::CONSTANT,
                                       /*per_channel_weight=*/d == Dt::S8P)),
        w_buf.data());
    auto bias_spec = make_bias_spec(d, b_sh);
    std::vector<uint8_t> b_dummy(4 * (dt_is_int8(d) ? 4 : dt_bytes(d)), 0);
    auto b = g->CreateTensor(bias_spec, b_dummy.data());
    auto c = g->CreateTensor(make_spec(d, out_sh, TA::OUTPUT));
    auto in_buf = filler(d, 64);
    a->CopyDataToTensor(in_buf.data(), in_buf.size());
    auto op = g->CreateOperation<tim::vx::ops::Conv2d>(
        /*weights=*/4u, tim::vx::PadType::VALID,
        std::array<uint32_t, 2>{3, 3}, std::array<uint32_t, 2>{1, 1},
        std::array<uint32_t, 2>{1, 1});
    (*op).BindInputs({a, w, b}).BindOutputs({c});
    return g;
  });
}
Result probe_pool2d_max(Dt d) {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType in_sh{8, 8, 1, 1};
    tim::vx::ShapeType out_sh{4, 4, 1, 1};
    auto a = g->CreateTensor(make_spec(d, in_sh, TA::INPUT));
    auto c = g->CreateTensor(make_spec(d, out_sh, TA::OUTPUT));
    auto buf = filler(d, 64);
    a->CopyDataToTensor(buf.data(), buf.size());
    auto op = g->CreateOperation<tim::vx::ops::Pool2d>(
        tim::vx::PoolType::MAX, tim::vx::PadType::VALID,
        std::array<uint32_t, 2>{2, 2}, std::array<uint32_t, 2>{2, 2});
    (*op).BindInputs({a}).BindOutputs({c});
    return g;
  });
}
Result probe_pool2d_avg(Dt d) {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType in_sh{8, 8, 1, 1};
    tim::vx::ShapeType out_sh{4, 4, 1, 1};
    auto a = g->CreateTensor(make_spec(d, in_sh, TA::INPUT));
    auto c = g->CreateTensor(make_spec(d, out_sh, TA::OUTPUT));
    auto buf = filler(d, 64);
    a->CopyDataToTensor(buf.data(), buf.size());
    auto op = g->CreateOperation<tim::vx::ops::Pool2d>(
        tim::vx::PoolType::AVG, tim::vx::PadType::VALID,
        std::array<uint32_t, 2>{2, 2}, std::array<uint32_t, 2>{2, 2});
    (*op).BindInputs({a}).BindOutputs({c});
    return g;
  });
}
Result probe_matmul(Dt d) {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType a_sh{3, 4};   // K=3, M=4
    tim::vx::ShapeType b_sh{5, 3};   // N=5, K=3
    tim::vx::ShapeType c_sh{5, 4};   // N=5, M=4
    auto a = g->CreateTensor(make_spec(d, a_sh, TA::INPUT));
    auto b = g->CreateTensor(make_spec(d, b_sh, TA::INPUT));
    auto c = g->CreateTensor(make_spec(d, c_sh, TA::OUTPUT));
    auto a_buf = filler(d, 12);
    auto b_buf = filler(d, 15);
    a->CopyDataToTensor(a_buf.data(), a_buf.size());
    b->CopyDataToTensor(b_buf.data(), b_buf.size());
    auto op = g->CreateOperation<tim::vx::ops::Matmul>();
    (*op).BindInputs({a, b}).BindOutputs({c});
    return g;
  });
}
// 3D batched matmul, mirroring the upstream DISABLED_shape_2_3_2 uint8
// test layout. Lets us verify whether the chip's quantized batch-GEMM
// path is *truly* missing or just rejects the 2D probe shape.
Result probe_matmul3d(Dt d) {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType a_sh{2, 3, 2};  // batch=2, math (3x2)
    tim::vx::ShapeType b_sh{2, 3, 2};  // batch=2, math (3x2), then transposed
    tim::vx::ShapeType c_sh{3, 3, 2};  // (3x3) per batch
    auto a = g->CreateTensor(make_spec(d, a_sh, TA::INPUT));
    auto b = g->CreateTensor(make_spec(d, b_sh, TA::INPUT));
    auto c = g->CreateTensor(make_spec(d, c_sh, TA::OUTPUT));
    auto a_buf = filler(d, 12);
    auto b_buf = filler(d, 12);
    a->CopyDataToTensor(a_buf.data(), a_buf.size());
    b->CopyDataToTensor(b_buf.data(), b_buf.size());
    auto op = g->CreateOperation<tim::vx::ops::Matmul>(false, true);
    (*op).BindInputs({a, b}).BindOutputs({c});
    return g;
  });
}
// DataConvert: int8/uint8 source (any quant) → fp32 target. Mirrors the
// dequant cast our quantized lowering emits — verify whether the NPU
// supports it natively before relying on it.
Result probe_dataconvert_i_to_f32(Dt d) {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType sh{4};
    auto a = g->CreateTensor(make_spec(d, sh, TA::INPUT));
    auto c = g->CreateTensor(make_spec(Dt::FP32, sh, TA::OUTPUT));
    auto buf = filler(d, 4);
    a->CopyDataToTensor(buf.data(), buf.size());
    auto op = g->CreateOperation<tim::vx::ops::DataConvert>();
    (*op).BindInputs({a}).BindOutputs({c});
    return g;
  });
}
// DataConvert: fp32 → i32 (the residual quantize cast).
Result probe_dataconvert_f32_to_i32(Dt /*ignored*/) {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType sh{4};
    auto a = g->CreateTensor(make_spec(Dt::FP32, sh, TA::INPUT));
    auto c = g->CreateTensor(
        tim::vx::TensorSpec(DT::INT32, sh, TA::OUTPUT));
    std::vector<float> buf(4, 0.5f);
    a->CopyDataToTensor(buf.data(), buf.size() * sizeof(float));
    auto op = g->CreateOperation<tim::vx::ops::DataConvert>();
    (*op).BindInputs({a}).BindOutputs({c});
    return g;
  });
}
// DataConvert: f32 → i8 directly. If supported, we can fuse the
// `f32 → i32 → add zp → i8` requantization chain into a single
// quantizing DataConvert that encodes (scale, zp) on the output spec.
Result probe_dataconvert_f32_to_i8(Dt /*ignored*/) {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType sh{4};
    auto a = g->CreateTensor(make_spec(Dt::FP32, sh, TA::INPUT));
    auto c = g->CreateTensor(make_spec(Dt::S8, sh, TA::OUTPUT));
    std::vector<float> buf(4, 0.5f);
    a->CopyDataToTensor(buf.data(), buf.size() * sizeof(float));
    auto op = g->CreateOperation<tim::vx::ops::DataConvert>();
    (*op).BindInputs({a}).BindOutputs({c});
    return g;
  });
}
// DataConvert: i32 → i8 (the requant truncation cast).
Result probe_dataconvert_i32_to_i8(Dt /*ignored*/) {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType sh{4};
    auto a = g->CreateTensor(
        tim::vx::TensorSpec(DT::INT32, sh, TA::INPUT));
    auto c = g->CreateTensor(make_spec(Dt::S8, sh, TA::OUTPUT));
    std::vector<int32_t> buf(4, 0);
    a->CopyDataToTensor(buf.data(), buf.size() * sizeof(int32_t));
    auto op = g->CreateOperation<tim::vx::ops::DataConvert>();
    (*op).BindInputs({a}).BindOutputs({c});
    return g;
  });
}

Result probe_reduce_sum(Dt d) {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType in_sh{4, 4, 1, 1};
    tim::vx::ShapeType out_sh{1, 1, 1, 1};
    auto a = g->CreateTensor(make_spec(d, in_sh, TA::INPUT));
    auto c = g->CreateTensor(make_spec(d, out_sh, TA::OUTPUT));
    auto buf = filler(d, 16);
    a->CopyDataToTensor(buf.data(), buf.size());
    auto op = g->CreateOperation<tim::vx::ops::ReduceSum>(
        std::vector<int32_t>{0, 1}, /*keep_dims=*/true);
    (*op).BindInputs({a}).BindOutputs({c});
    return g;
  });
}
Result probe_reduce_mean(Dt d) {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType in_sh{4, 4, 1, 1};
    tim::vx::ShapeType out_sh{1, 1, 1, 1};
    auto a = g->CreateTensor(make_spec(d, in_sh, TA::INPUT));
    auto c = g->CreateTensor(make_spec(d, out_sh, TA::OUTPUT));
    auto buf = filler(d, 16);
    a->CopyDataToTensor(buf.data(), buf.size());
    auto op = g->CreateOperation<tim::vx::ops::ReduceMean>(
        std::vector<int32_t>{0, 1}, /*keep_dims=*/true);
    (*op).BindInputs({a}).BindOutputs({c});
    return g;
  });
}
Result probe_fc(Dt d) {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType in_sh{3, 1};      // K=3, batch=1
    tim::vx::ShapeType w_sh {3, 5};      // K=3, N=5
    tim::vx::ShapeType b_sh {5};
    tim::vx::ShapeType out_sh{5, 1};
    auto a = g->CreateTensor(make_spec(d, in_sh, TA::INPUT));
    auto w_buf = filler(d, 15);
    auto w = g->CreateTensor(make_spec(d, w_sh, TA::CONSTANT), w_buf.data());
    std::vector<uint8_t> b_dummy(5 * (dt_is_int8(d) ? 4 : dt_bytes(d)), 0);
    auto b = g->CreateTensor(make_bias_spec(d, b_sh), b_dummy.data());
    auto c = g->CreateTensor(make_spec(d, out_sh, TA::OUTPUT));
    auto in_buf = filler(d, 3);
    a->CopyDataToTensor(in_buf.data(), in_buf.size());
    auto op = g->CreateOperation<tim::vx::ops::FullyConnected>(/*axis=*/0u,
                                                               /*weights=*/5u);
    (*op).BindInputs({a, w, b}).BindOutputs({c});
    return g;
  });
}

struct Probe {
  const char* op;
  std::function<Result(Dt)> fn;
};

}  // namespace

int main() {
  std::vector<Probe> probes{
      {"Add",          probe_add},
      {"Sub",          probe_sub},
      {"Multiply",     probe_mul},
      {"Pow",          probe_pow},
      {"Rcp",          probe_rcp},
      {"Clip",         probe_clip},
      {"Reshape",      probe_reshape},
      {"Slice",        probe_slice},
      {"Transpose",    probe_transpose},
      {"Pool2d MAX",   probe_pool2d_max},
      {"Pool2d AVG",   probe_pool2d_avg},
      {"Conv2d",       probe_conv2d},
      {"Matmul (2D)",  probe_matmul},
      {"Matmul (3D tB)", probe_matmul3d},
      {"FullyConnect", probe_fc},
      {"ReduceSum",     probe_reduce_sum},
      {"ReduceMean",    probe_reduce_mean},
      {"DataConvert→f32", probe_dataconvert_i_to_f32},
      {"DC f32→i32",      probe_dataconvert_f32_to_i32},
      {"DC i32→i8",       probe_dataconvert_i32_to_i8},
  };
  std::vector<Dt> dtypes{Dt::FP32, Dt::FP16, Dt::U8, Dt::S8, Dt::S8P};

  // Run each probe in a forked child so a SIGSEGV in libGAL/libOpenVX
  // (we observed one for Matmul + UINT8) doesn't kill the report.
  // The child redirects stdout/stderr to /dev/null to suppress
  // libGAL's "DW Enhancement is missing" chatter and OVXLIB error
  // logs, then exits with the Status code. Parent maps exit codes
  // (and signal-termination) back to a label.
  std::printf("%-15s |", "op");
  for (Dt d : dtypes) std::printf(" %-13s |", dt_name(d));
  std::printf("\n----------------+");
  for (size_t i = 0; i < dtypes.size(); ++i) std::printf("---------------+");
  std::printf("\n");
  std::fflush(stdout);
  for (const auto& p : probes) {
    std::printf("%-15s |", p.op);
    std::fflush(stdout);
    for (Dt d : dtypes) {
      pid_t pid = fork();
      if (pid == 0) {
        // child
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
          dup2(devnull, STDOUT_FILENO);
          dup2(devnull, STDERR_FILENO);
          close(devnull);
        }
        Result r = p.fn(d);
        Status s = r.compile_ok ? (r.run_ok ? Status::PASS : Status::RUN_FAIL)
                                : Status::COMPILE_FAIL;
        _exit(static_cast<int>(s));
      }
      int wstatus = 0;
      waitpid(pid, &wstatus, 0);
      Status s = Status::CRASH;
      if (WIFEXITED(wstatus)) s = static_cast<Status>(WEXITSTATUS(wstatus));
      std::printf(" %-13s |", status_name(s));
      std::fflush(stdout);
    }
    std::printf("\n");
  }
  return 0;
}
