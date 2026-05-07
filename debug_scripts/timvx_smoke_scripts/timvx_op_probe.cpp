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
// Dt::I32 — plain int32 with no quantization (the dtype the residual
//           `cast f32→i32 → add zp → cast i32→i8|asym` chain uses for the
//           middle slot; also the rescale-bias storage in some lowerings).
enum class Dt { FP32, FP16, U8, S8, S8P, I32 };
const char* dt_name(Dt d) {
  switch (d) {
    case Dt::FP32: return "fp32";
    case Dt::FP16: return "fp16";
    case Dt::U8:   return "u8";
    case Dt::S8:   return "s8";
    case Dt::S8P:  return "s8pc";
    case Dt::I32:  return "i32";
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
    case Dt::I32:  return DT::INT32;
  }
  return DT::FLOAT32;
}
size_t dt_bytes(Dt d) {
  switch (d) {
    case Dt::U8: case Dt::S8: case Dt::S8P: return 1;
    case Dt::FP16:                          return 2;
    case Dt::FP32:                          return 4;
    case Dt::I32:                           return 4;
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
  // Dt::I32 / FP32 / FP16 — plain dtype, no quantization.
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
  } else if (d == Dt::I32) {
    int32_t v = small_nonzero ? 1 : 0;
    auto* p = reinterpret_cast<int32_t*>(buf.data());
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
// ── Broadcast-elementwise probe ──────────────────────────────────────
//
// The residual quantize chain has `add %f32_or_i32, %scalar_zp` where the
// zp is a 1-element broadcast operand. Broadcast handling can fail
// independently of the same-shape case. We probe i32 with a 4×4 vs 1×1
// shape pair to validate broadcast behavior on whatever dtypes pass the
// same-shape Add column. (Same-shape i32 add is already covered by the
// `Dt::I32` column of the main sweep.)
Result probe_add_i32_broadcast() {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType lhs_sh{4, 4};
    tim::vx::ShapeType rhs_sh{1, 1};
    tim::vx::ShapeType out_sh{4, 4};
    auto a = g->CreateTensor(tim::vx::TensorSpec(DT::INT32, lhs_sh, TA::INPUT));
    auto b = g->CreateTensor(tim::vx::TensorSpec(DT::INT32, rhs_sh, TA::INPUT));
    auto c = g->CreateTensor(tim::vx::TensorSpec(DT::INT32, out_sh, TA::OUTPUT));
    std::vector<int32_t> buf_a(16, 0);
    std::vector<int32_t> buf_b(1, 0);
    a->CopyDataToTensor(buf_a.data(), buf_a.size() * sizeof(int32_t));
    b->CopyDataToTensor(buf_b.data(), buf_b.size() * sizeof(int32_t));
    auto op = g->CreateOperation<tim::vx::ops::Add>();
    (*op).BindInputs({a, b}).BindOutputs({c});
    return g;
  });
}

// Realistic-shape probes: the rank-1 4-element probes in the main sweep
// confirm KERNEL EXISTENCE, not full-resolution behavior. ResNet18's
// residual blocks operate on 4D fp32 tensors of e.g. 56×56×64; a kernel
// that PASSes for `{4}` may COMPILE_FAIL or hang at runtime on the
// production shape. These probes mirror the actual feature-map sizes.
Result probe_add_fp32_4d() {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType sh{56, 56, 64, 1};  // TIM-VX inner-first: W, H, C, N
    auto a = g->CreateTensor(make_spec(Dt::FP32, sh, TA::INPUT));
    auto b = g->CreateTensor(make_spec(Dt::FP32, sh, TA::INPUT));
    auto c = g->CreateTensor(make_spec(Dt::FP32, sh, TA::OUTPUT));
    auto buf = filler(Dt::FP32, 56 * 56 * 64);
    a->CopyDataToTensor(buf.data(), buf.size());
    b->CopyDataToTensor(buf.data(), buf.size());
    auto op = g->CreateOperation<tim::vx::ops::Add>();
    (*op).BindInputs({a, b}).BindOutputs({c});
    return g;
  });
}
Result probe_clip_fp32_4d() {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType sh{56, 56, 64, 1};
    auto a = g->CreateTensor(make_spec(Dt::FP32, sh, TA::INPUT));
    auto c = g->CreateTensor(make_spec(Dt::FP32, sh, TA::OUTPUT));
    auto buf = filler(Dt::FP32, 56 * 56 * 64);
    a->CopyDataToTensor(buf.data(), buf.size());
    auto op = g->CreateOperation<tim::vx::ops::Clip>(0.0f, 3.4028235e38f);
    (*op).BindInputs({a}).BindOutputs({c});
    return g;
  });
}
Result probe_cast_fp32_to_u8_4d() {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType sh{56, 56, 64, 1};
    auto a = g->CreateTensor(make_spec(Dt::FP32, sh, TA::INPUT));
    auto c = g->CreateTensor(make_spec(Dt::U8, sh, TA::OUTPUT));
    auto buf = filler(Dt::FP32, 56 * 56 * 64);
    a->CopyDataToTensor(buf.data(), buf.size());
    auto op = g->CreateOperation<tim::vx::ops::Cast>();
    (*op).BindInputs({a}).BindOutputs({c});
    return g;
  });
}
// The exact "transpose → slice → transpose → 1x1 stride-2 conv" chain
// the layout pass produces around the projection shortcut in stages 2/3/4
// of ResNet18. Each individual op probes PASS, but TIM-VX's runtime
// kernel scheduler may stall on the strided u8 transpose pair under
// real graph pressure. Reproducing the exact chain in isolation is the
// fastest way to confirm whether that's the failing pattern.
Result probe_transpose_slice_transpose_conv() {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    // Start: u8 WHCN tensor matching stage 2's input.
    tim::vx::ShapeType whcn_in{56, 56, 64, 1};
    auto in = g->CreateTensor(make_spec(Dt::U8, whcn_in, TA::INPUT));
    auto in_buf = filler(Dt::U8, 56 * 56 * 64);
    in->CopyDataToTensor(in_buf.data(), in_buf.size());

    // 1) WHCN → NHWC: perms [3, 1, 0, 2] turns {W=56, H=56, C=64, N=1}
    //    into {N=1, H=56, W=56, C=64}.
    tim::vx::ShapeType nhwc_full{1, 56, 56, 64};
    auto nhwc = g->CreateTensor(make_spec(Dt::U8, nhwc_full, TA::TRANSIENT));
    auto t1 = g->CreateOperation<tim::vx::ops::Transpose>(
        std::vector<uint32_t>{3, 1, 0, 2});
    (*t1).BindInputs({in}).BindOutputs({nhwc});

    // 2) Slice in NHWC: crop H and W from 56 to 55.
    tim::vx::ShapeType nhwc_crop{1, 55, 55, 64};
    auto cropped = g->CreateTensor(
        make_spec(Dt::U8, nhwc_crop, TA::TRANSIENT));
    auto sl = g->CreateOperation<tim::vx::ops::Slice>(
        /*dims=*/4u,
        std::vector<int32_t>{0, 0, 0, 0},
        std::vector<int32_t>{1, 55, 55, 64});
    (*sl).BindInputs({nhwc}).BindOutputs({cropped});

    // 3) NHWC → WHCN: perms [2, 1, 3, 0].
    tim::vx::ShapeType whcn_crop{55, 55, 64, 1};
    auto back = g->CreateTensor(make_spec(Dt::U8, whcn_crop, TA::TRANSIENT));
    auto t2 = g->CreateOperation<tim::vx::ops::Transpose>(
        std::vector<uint32_t>{2, 1, 3, 0});
    (*t2).BindInputs({cropped}).BindOutputs({back});

    // 4) 1x1 stride-2 conv: 55x55 → 28x28.
    tim::vx::ShapeType w_sh{1, 1, 64, 128};
    tim::vx::ShapeType b_sh{128};
    tim::vx::ShapeType out_sh{28, 28, 128, 1};
    auto w_buf = filler(Dt::U8, 64 * 128);
    auto w = g->CreateTensor(make_spec(Dt::U8, w_sh, TA::CONSTANT),
                              w_buf.data());
    std::vector<uint8_t> b_dummy(128 * 4, 0);
    auto bias = g->CreateTensor(make_bias_spec(Dt::U8, b_sh), b_dummy.data());
    auto out = g->CreateTensor(make_spec(Dt::U8, out_sh, TA::OUTPUT));
    auto conv = g->CreateOperation<tim::vx::ops::Conv2d>(
        /*weights=*/128u, tim::vx::PadType::VALID,
        std::array<uint32_t, 2>{1, 1},
        std::array<uint32_t, 2>{2, 2},
        std::array<uint32_t, 2>{1, 1});
    (*conv).BindInputs({back, w, bias}).BindOutputs({out});
    return g;
  });
}

// 1x1 stride-2 conv2d on u8 — the ResNet projection-shortcut downsample
// pattern. Only present in BasicBlocks where (stride != 1 or in_ch !=
// out_ch). Stage 1 of the user's ResNet18 has identity shortcuts and
// runs end-to-end fine; stages 2/3/4 add this 1x1 stride-2 op and the
// model hangs at Run(). The general probe of Conv2D u8 PASSed but used
// 3x3 stride-1 — this targets the specific kernel-stride-1×1/stride-2
// permutation.
Result probe_conv2d_1x1_stride2_u8() {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType in_sh{56, 56, 64, 1};   // W H C N inner-first
    tim::vx::ShapeType w_sh {1, 1, 64, 128};   // KW KH IC OC
    tim::vx::ShapeType b_sh {128};
    tim::vx::ShapeType out_sh{28, 28, 128, 1};
    auto a = g->CreateTensor(make_spec(Dt::U8, in_sh, TA::INPUT));
    auto w_buf = filler(Dt::U8, 64 * 128);
    auto w = g->CreateTensor(make_spec(Dt::U8, w_sh, TA::CONSTANT), w_buf.data());
    std::vector<uint8_t> b_dummy(128 * 4, 0);  // i32 bias
    auto b = g->CreateTensor(make_bias_spec(Dt::U8, b_sh), b_dummy.data());
    auto c = g->CreateTensor(make_spec(Dt::U8, out_sh, TA::OUTPUT));
    auto in_buf = filler(Dt::U8, 56 * 56 * 64);
    a->CopyDataToTensor(in_buf.data(), in_buf.size());
    auto op = g->CreateOperation<tim::vx::ops::Conv2d>(
        /*weights=*/128u, tim::vx::PadType::VALID,
        std::array<uint32_t, 2>{1, 1},      // kernel = 1x1
        std::array<uint32_t, 2>{2, 2},      // stride = 2x2
        std::array<uint32_t, 2>{1, 1});     // dilation
    (*op).BindInputs({a, w, b}).BindOutputs({c});
    return g;
  });
}

Result probe_pool2d_avg_u8_global7() {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType in_sh{7, 7, 512, 1};   // global avg-pool input
    tim::vx::ShapeType out_sh{1, 1, 512, 1};
    auto a = g->CreateTensor(make_spec(Dt::U8, in_sh, TA::INPUT));
    auto c = g->CreateTensor(make_spec(Dt::U8, out_sh, TA::OUTPUT));
    auto buf = filler(Dt::U8, 7 * 7 * 512);
    a->CopyDataToTensor(buf.data(), buf.size());
    auto op = g->CreateOperation<tim::vx::ops::Pool2d>(
        tim::vx::PoolType::AVG, tim::vx::PadType::VALID,
        std::array<uint32_t, 2>{7, 7}, std::array<uint32_t, 2>{1, 1});
    (*op).BindInputs({a}).BindOutputs({c});
    return g;
  });
}

// ── Conversion-pair probes (DataConvert vs Cast) ─────────────────────
//
// TIM-VX exposes two distinct ops for changing tensor dtype/quant.
// Their op_check tables (vsi_nn_op_dataconvert.c, vsi_nn_op_cast.c)
// allow disjoint sets of (input_dtype|qnt, output_dtype|qnt) pairs:
//
//   DataConvert (VSI_NN_OP_DATACONVERT) — internally vxTensorCopyNode;
//     intended for true (de)quantize / requantize. Accepts e.g.
//     F32→U8|ASYM, F32→I32|ASYM, U8|ASYM→F32, U8|ASYM⇄I8|ASYM. NOT
//     F32→I8|ASYM, NOT F32→I32(no quant), NOT I8|ASYM→F32.
//
//   Cast (VSI_NN_OP_CAST) — value cast that ignores scale/zp. Internally
//     dispatches to the GPU "cast" kernel (or falls through to a
//     DataConvert when both ends are quant-free non-FP). Accepts e.g.
//     F32→I32 raw, F32→I8|ASYM, I32→I8|ASYM, U8|ASYM→F32, I8(raw)→I32.
//
// These probes verify which subset *this* chip actually instantiates,
// pair by pair. Each probe takes no dtype argument (the pair is fixed)
// and reports a single PASS/COMPILE_FAIL/RUN_FAIL/CRASH cell.

namespace pair_probe {

struct QSpec {
  DT dt;
  tim::vx::QuantType qt;  // NONE for plain (no Quantization on spec)
  float scale;
  int32_t zp;
};

tim::vx::TensorSpec make_qspec(const QSpec& q,
                               const tim::vx::ShapeType& sh, TA attr) {
  if (q.qt == tim::vx::QuantType::NONE)
    return tim::vx::TensorSpec(q.dt, sh, attr);
  // For per-tensor ASYM the c++ Quantization ctor takes (type, scale, zp).
  // PER_CHANNEL variants need separate vectors and aren't exercised here —
  // every interesting case for our pipeline is per-tensor.
  tim::vx::Quantization qq(q.qt, q.scale, q.zp);
  return tim::vx::TensorSpec(q.dt, sh, attr, qq);
}

size_t bytes(DT dt) {
  switch (dt) {
    case DT::INT8: case DT::UINT8: case DT::BOOL8: return 1;
    case DT::INT16: case DT::UINT16: case DT::FLOAT16: return 2;
    case DT::INT32: case DT::UINT32: case DT::FLOAT32: return 4;
    default: return 4;
  }
}

// Allocate a small zero-filled buffer and feed it to the input tensor.
void feed_zeros(const std::shared_ptr<tim::vx::Tensor>& t, size_t numel) {
  std::vector<uint8_t> buf(numel * bytes(t->GetSpec().GetDataType()), 0);
  t->CopyDataToTensor(buf.data(), buf.size());
}

enum class OpKind { DATACONVERT, CAST };

Result run_pair(OpKind kind, QSpec in, QSpec out) {
  return probe([&](std::shared_ptr<tim::vx::Context> ctx) {
    auto g = ctx->CreateGraph();
    tim::vx::ShapeType sh{4};
    auto a = g->CreateTensor(make_qspec(in, sh, TA::INPUT));
    auto c = g->CreateTensor(make_qspec(out, sh, TA::OUTPUT));
    feed_zeros(a, 4);
    if (kind == OpKind::DATACONVERT) {
      auto op = g->CreateOperation<tim::vx::ops::DataConvert>();
      (*op).BindInputs({a}).BindOutputs({c});
    } else {
      auto op = g->CreateOperation<tim::vx::ops::Cast>();
      (*op).BindInputs({a}).BindOutputs({c});
    }
    return g;
  });
}

// Convenience builders for the only quant flavors exposed by tim::vx::QuantType
// that we care about here.
constexpr QSpec none(DT dt) { return {dt, tim::vx::QuantType::NONE, 0.0f, 0}; }
constexpr QSpec asym(DT dt, float s, int32_t z) {
  return {dt, tim::vx::QuantType::ASYMMETRIC, s, z};
}

struct Entry {
  const char* label;
  OpKind kind;
  QSpec in;
  QSpec out;
};

// Note: `tim::vx::QuantType` does NOT expose a per-tensor SYMMETRIC nor
// the legacy DYNAMIC_FIXED_POINT in a way that's wired through the
// activations path of typical pipelines, so the probes below stick to
// {NONE, ASYMMETRIC} — these are the only quant flavors our MLIR
// lowering can plausibly emit, and they cover every pair-row in the
// op_check tables that matters in practice.
const std::vector<Entry> kEntries = {
    // -- DataConvert --
    // canonical quantize: f32 → u8|asym (DC table HAS this)
    {"DC f32        → u8|asym",   OpKind::DATACONVERT, none(DT::FLOAT32),
     asym(DT::UINT8, 1.0f / 255.0f, 128)},
    // canonical dequantize: u8|asym → f32 (DC table HAS this)
    {"DC u8|asym    → f32",       OpKind::DATACONVERT,
     asym(DT::UINT8, 1.0f / 255.0f, 128), none(DT::FLOAT32)},
    // f32 → i8|asym (DC table does NOT have ASYM-on-i8)
    {"DC f32        → i8|asym",   OpKind::DATACONVERT, none(DT::FLOAT32),
     asym(DT::INT8, 1.0f / 127.0f, 0)},
    // i8|asym → f32 (DC table only has I8|DFP → F32; ASYM not listed)
    {"DC i8|asym    → f32",       OpKind::DATACONVERT,
     asym(DT::INT8, 1.0f / 127.0f, 0), none(DT::FLOAT32)},
    // requant same-dtype with different (s,zp): u8 → u8 (DC table HAS this)
    {"DC u8|asym    → u8|asym'",  OpKind::DATACONVERT,
     asym(DT::UINT8, 1.0f / 255.0f, 128),
     asym(DT::UINT8, 2.0f / 255.0f, 64)},
    // requant: i8 → i8 (DC table HAS I8|ASYM → I8|ASYM)
    {"DC i8|asym    → i8|asym'",  OpKind::DATACONVERT,
     asym(DT::INT8, 1.0f / 127.0f, 0),
     asym(DT::INT8, 2.0f / 127.0f, 5)},
    // crossing: u8 → i8 (DC table HAS this both ways)
    {"DC u8|asym    → i8|asym",   OpKind::DATACONVERT,
     asym(DT::UINT8, 1.0f / 255.0f, 128),
     asym(DT::INT8, 1.0f / 127.0f, 0)},
    {"DC i8|asym    → u8|asym",   OpKind::DATACONVERT,
     asym(DT::INT8, 1.0f / 127.0f, 0),
     asym(DT::UINT8, 1.0f / 255.0f, 128)},
    // f32 → i32|asym (DC table HAS this — requires quant on i32 output)
    {"DC f32        → i32|asym",  OpKind::DATACONVERT, none(DT::FLOAT32),
     asym(DT::INT32, 1.0f / 4096.0f, 0)},
    // f32 → i32 raw (no quant on either) — DC table does NOT include
    {"DC f32        → i32 raw",   OpKind::DATACONVERT, none(DT::FLOAT32),
     none(DT::INT32)},
    // i32|asym → i8|dfp / u8|asym — DC table HAS i32|asym → u8|asym
    // (the rescale-tail target). We use i32|asym → u8|asym here since
    // i8|DFP isn't representable in QuantType. ASYM-on-i32 has scale,
    // zp=0.
    {"DC i32|asym   → u8|asym",   OpKind::DATACONVERT,
     asym(DT::INT32, 1.0f / 4096.0f, 0),
     asym(DT::UINT8, 1.0f / 255.0f, 128)},
    {"DC i32|asym   → i8|asym",   OpKind::DATACONVERT,
     asym(DT::INT32, 1.0f / 4096.0f, 0),
     asym(DT::INT8, 1.0f / 127.0f, 0)},

    // -- Cast --
    // raw value casts (no quant on either side)
    {"Cast f32      → i32 raw",   OpKind::CAST,   none(DT::FLOAT32),
     none(DT::INT32)},
    {"Cast i32 raw  → i32 raw",   OpKind::CAST,   none(DT::INT32),
     none(DT::INT32)},
    {"Cast i8 raw   → i32 raw",   OpKind::CAST,   none(DT::INT8),
     none(DT::INT32)},
    // f32 → quantized int (this is the residual quantize the user's TOSA
    // emits as `cast f32→i32 → add zp → cast i32→i8`)
    {"Cast f32      → u8|asym",   OpKind::CAST,   none(DT::FLOAT32),
     asym(DT::UINT8, 1.0f / 255.0f, 128)},
    {"Cast f32      → i8|asym",   OpKind::CAST,   none(DT::FLOAT32),
     asym(DT::INT8, 1.0f / 127.0f, 0)},
    // i32 (raw) → quantized int (the second cast in the residual chain)
    {"Cast i32 raw  → u8|asym",   OpKind::CAST,   none(DT::INT32),
     asym(DT::UINT8, 1.0f / 255.0f, 128)},
    {"Cast i32 raw  → i8|asym",   OpKind::CAST,   none(DT::INT32),
     asym(DT::INT8, 1.0f / 127.0f, 0)},
    // dequant via Cast (Cast table HAS u8|asym→f32, but NOT i8|asym→f32)
    {"Cast u8|asym  → f32",       OpKind::CAST,
     asym(DT::UINT8, 1.0f / 255.0f, 128), none(DT::FLOAT32)},
    {"Cast i8|asym  → f32",       OpKind::CAST,
     asym(DT::INT8, 1.0f / 127.0f, 0), none(DT::FLOAT32)},
};

}  // namespace pair_probe

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
  };
  std::vector<Dt> dtypes{Dt::FP32, Dt::FP16, Dt::U8, Dt::S8, Dt::S8P, Dt::I32};

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

  // ── Broadcast-shape variants ──
  // Same-shape behavior is already covered in the main sweep above; this
  // section exists to flag dtypes whose broadcast path differs (the
  // residual quantize chain uses a 1-element `add zp` broadcast).
  std::printf("\n%-26s | %-13s\n", "broadcast variants", "result");
  std::printf("---------------------------+---------------\n");
  std::fflush(stdout);
  struct BcProbe { const char* label; std::function<Result()> fn; };
  std::vector<BcProbe> bc_probes{
      {"Add i32 broadcast 4x4/1x1",   probe_add_i32_broadcast},
      // Realistic-shape probes — see the comment block on the
      // probe_add_fp32_4d / probe_clip_fp32_4d defs for why these
      // are essential. The main sweep's rank-1 4-element probes
      // confirm kernel registration, not behavior on production-size
      // tensors. A kernel that PASSes the small probe but fails on
      // the real shape is the canonical "10-second NPU watchdog"
      // failure signature.
      {"Add fp32 56x56x64x1",        probe_add_fp32_4d},
      {"Clip fp32 56x56x64x1",       probe_clip_fp32_4d},
      {"Cast fp32→u8 56x56x64x1",    probe_cast_fp32_to_u8_4d},
      {"Pool2d AVG u8 7x7→1x1",      probe_pool2d_avg_u8_global7},
      {"Conv2d 1x1 s=2 u8 56→28",    probe_conv2d_1x1_stride2_u8},
      {"transp+slice+transp+conv1x1s2", probe_transpose_slice_transpose_conv},
  };
  for (const auto& p : bc_probes) {
    pid_t pid = fork();
    if (pid == 0) {
      int devnull = open("/dev/null", O_WRONLY);
      if (devnull >= 0) {
        dup2(devnull, STDOUT_FILENO);
        dup2(devnull, STDERR_FILENO);
        close(devnull);
      }
      Result r = p.fn();
      Status s = r.compile_ok ? (r.run_ok ? Status::PASS : Status::RUN_FAIL)
                              : Status::COMPILE_FAIL;
      _exit(static_cast<int>(s));
    }
    int wstatus = 0;
    waitpid(pid, &wstatus, 0);
    Status s = Status::CRASH;
    if (WIFEXITED(wstatus)) s = static_cast<Status>(WEXITSTATUS(wstatus));
    std::printf("%-26s | %-13s\n", p.label, status_name(s));
    std::fflush(stdout);
  }

  // ── Pair-conversion section ──
  // One row per (op, in_pair, out_pair) entry; one PASS/FAIL cell each.
  std::printf("\n%-26s | %-13s\n", "DataConvert/Cast pair", "result");
  std::printf("---------------------------+---------------\n");
  std::fflush(stdout);
  for (const auto& e : pair_probe::kEntries) {
    pid_t pid = fork();
    if (pid == 0) {
      int devnull = open("/dev/null", O_WRONLY);
      if (devnull >= 0) {
        dup2(devnull, STDOUT_FILENO);
        dup2(devnull, STDERR_FILENO);
        close(devnull);
      }
      Result r = pair_probe::run_pair(e.kind, e.in, e.out);
      Status s = r.compile_ok ? (r.run_ok ? Status::PASS : Status::RUN_FAIL)
                              : Status::COMPILE_FAIL;
      _exit(static_cast<int>(s));
    }
    int wstatus = 0;
    waitpid(pid, &wstatus, 0);
    Status s = Status::CRASH;
    if (WIFEXITED(wstatus)) s = static_cast<Status>(WEXITSTATUS(wstatus));
    std::printf("%-26s | %-13s\n", e.label, status_name(s));
    std::fflush(stdout);
  }
  return 0;
}
