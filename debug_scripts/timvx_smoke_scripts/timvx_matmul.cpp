// timvx_matmul.cpp — end-to-end matmul correctness test on the NPU.
//
// Why FullyConnected with UINT8 quantization (not fp32 matmul)?
//   We discovered two stacked constraints on this chip
//   (VIP9000, pid 0x1000003b, on the Allwinner A733):
//
//     1. tim::vx::ops::Matmul routes to "vivante.nn.batch.gemm.layer",
//        a NN-engine batched-GEMM block this Nano variant lacks. Every
//        Matmul shape fails Compile with status = -1.
//
//     2. Even when we reroute through tim::vx::ops::FullyConnected
//        (which calls vxFullyConnectedLayer directly), the NN-engine
//        kernel "vivante.nn.fullyconnected.layer" itself rejects
//        fmt[f32] qnt[NONE] — also status = -1. The NN engine on this
//        chip is quantized-only.
//
//   The lenet sample under TIM-VX/samples/lenet runs fine here and
//   uses UINT8 ASYMMETRIC quantized inputs/weights with INT32 bias
//   throughout — that is the known-good op + dtype combination on
//   this hardware. This test mirrors that pattern: matmul C = A @ B is
//   computed as a quantized FullyConnected with weight = B^T and
//   bias = 0. We then dequantize the UINT8 output and compare it to
//   an fp64 CPU reference, with a tolerance that accounts for
//   quantization noise (~1 output LSB).
//
// Mapping:  C(M,N) = A(M,K) @ B(K,N) computed as
//           FullyConnected(input=A, weight=B^T, bias=0).
//   FC computes y[m,n] = Σ_k x[m,k] * W[n,k] + b[n], so we store the
//   weight buffer as B-transposed: W_buf[n*K + k] = B[k][n].
//
// TIM-VX shape convention is inner-first:
//   input  shape {K, M}  (K features, M batch rows)
//   weight shape {K, N}  (K features, N output features)
//   bias   shape {N}
//   output shape {N, M}
//
// Quantization:
//   input  / weight: UINT8 ASYMMETRIC, scale = 2/255, zp = 128
//                    (covers symmetric float range [-1, +1])
//   bias:           INT32 ASYMMETRIC, scale = input_scale * weight_scale,
//                    zp = 0 (Vivante constraint for FC bias)
//   output:         UINT8 ASYMMETRIC, scale + zp picked per-scenario from
//                    the observed max |ref| with 20% headroom.
//
// Build/run:  bash timvx_matmul.bash

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/ops/fullyconnected.h"
#include "tim/vx/tensor.h"
#include "tim/vx/types.h"

namespace {

struct Scenario {
  const char* name;
  uint32_t M;  // batch rows (input outer dim)
  uint32_t K;  // input features (also reduction axis)
  uint32_t N;  // output features
};

// Deterministic, integer-derived fill in roughly [-1, +1]. Distinct
// generators for A and B so cancellation patterns can't accidentally
// hide a real arithmetic bug.
float fill_a(uint32_t m, uint32_t k) {
  int v = static_cast<int>((m * 31u + k * 17u) % 41u) - 20;
  return static_cast<float>(v) / 20.0f;
}
float fill_b(uint32_t k, uint32_t n) {
  int v = static_cast<int>((k * 23u + n * 11u) % 37u) - 18;
  return static_cast<float>(v) / 18.0f;
}

// CPU reference computed in fp64 then narrowed to fp32 — keeps the
// reference's own rounding noise well below the quantization budget.
std::vector<float> ref_matmul(const std::vector<float>& A,
                              const std::vector<float>& B, uint32_t M,
                              uint32_t K, uint32_t N) {
  std::vector<float> C(static_cast<size_t>(M) * N);
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      double acc = 0.0;
      for (uint32_t k = 0; k < K; ++k)
        acc += static_cast<double>(A[m * K + k]) *
               static_cast<double>(B[k * N + n]);
      C[m * N + n] = static_cast<float>(acc);
    }
  }
  return C;
}

// Symmetric UINT8 quantization for a value range [-V, +V]:
//   q = clip(round(x / scale + zp), 0, 255), with scale = 2V/255, zp = 128.
// Half-scale rounding match what the hardware does internally (round-to-nearest, ties-away-from-zero is fine here).
uint8_t quantize_u8(float x, float scale, int zp) {
  long q = std::lround(x / scale) + zp;
  if (q < 0) q = 0;
  if (q > 255) q = 255;
  return static_cast<uint8_t>(q);
}
float dequantize_u8(uint8_t q, float scale, int zp) {
  return (static_cast<int>(q) - zp) * scale;
}

bool run(const Scenario& s) {
  // ── Float buffers (math row-major) and reference ────────────────────
  std::vector<float> A_buf(static_cast<size_t>(s.M) * s.K);
  std::vector<float> B_buf(static_cast<size_t>(s.K) * s.N);
  for (uint32_t m = 0; m < s.M; ++m)
    for (uint32_t k = 0; k < s.K; ++k) A_buf[m * s.K + k] = fill_a(m, k);
  for (uint32_t k = 0; k < s.K; ++k)
    for (uint32_t n = 0; n < s.N; ++n) B_buf[k * s.N + n] = fill_b(k, n);

  std::vector<float> ref = ref_matmul(A_buf, B_buf, s.M, s.K, s.N);

  // ── Quantization params ────────────────────────────────────────────
  // Inputs/weights cover [-1, +1] (the fill functions guarantee that).
  const float in_scale  = 2.0f / 255.0f;
  const float w_scale   = 2.0f / 255.0f;
  const int   in_zp     = 128;
  const int   w_zp      = 128;
  const float bias_scale = in_scale * w_scale;  // Vivante FC constraint
  const int   bias_zp    = 0;

  // Output scale/zp from the observed range with 20% headroom on each
  // side, mapped symmetrically into UINT8 (zp = 128).
  float ref_max = 1e-6f;  // avoid div-by-zero on degenerate cases
  for (float v : ref) ref_max = std::max(ref_max, std::fabs(v));
  const float out_v = 1.2f * ref_max;
  const float out_scale = (2.0f * out_v) / 255.0f;
  const int   out_zp    = 128;

  // ── Quantize input and weight buffers ──────────────────────────────
  std::vector<uint8_t> in_q(A_buf.size());
  for (size_t i = 0; i < A_buf.size(); ++i)
    in_q[i] = quantize_u8(A_buf[i], in_scale, in_zp);

  // FC weight is B^T: W[n][k] = B[k][n].
  std::vector<uint8_t> w_q(static_cast<size_t>(s.K) * s.N);
  for (uint32_t k = 0; k < s.K; ++k)
    for (uint32_t n = 0; n < s.N; ++n)
      w_q[n * s.K + k] = quantize_u8(B_buf[k * s.N + n], w_scale, w_zp);

  // Bias = 0. INT32 storage; quantized 0 with zp=0 is just 0.
  std::vector<int32_t> bias_q(s.N, 0);

  // ── Build NPU graph ────────────────────────────────────────────────
  auto ctx = tim::vx::Context::Create();
  if (!ctx) {
    std::printf("[%-30s] FAIL  Context::Create returned null\n", s.name);
    return false;
  }
  auto g = ctx->CreateGraph();
  if (!g) {
    std::printf("[%-30s] FAIL  Graph::Create returned null\n", s.name);
    return false;
  }

  tim::vx::ShapeType in_shape{s.K, s.M};
  tim::vx::ShapeType weight_shape{s.K, s.N};
  tim::vx::ShapeType bias_shape{s.N};
  tim::vx::ShapeType out_shape{s.N, s.M};

  tim::vx::Quantization in_quant(tim::vx::QuantType::ASYMMETRIC, in_scale, in_zp);
  tim::vx::Quantization w_quant(tim::vx::QuantType::ASYMMETRIC, w_scale, w_zp);
  tim::vx::Quantization b_quant(tim::vx::QuantType::ASYMMETRIC, bias_scale,
                                bias_zp);
  tim::vx::Quantization out_quant(tim::vx::QuantType::ASYMMETRIC, out_scale,
                                  out_zp);

  tim::vx::TensorSpec in_spec(tim::vx::DataType::UINT8, in_shape,
                              tim::vx::TensorAttribute::INPUT, in_quant);
  tim::vx::TensorSpec w_spec(tim::vx::DataType::UINT8, weight_shape,
                             tim::vx::TensorAttribute::CONSTANT, w_quant);
  tim::vx::TensorSpec b_spec(tim::vx::DataType::INT32, bias_shape,
                             tim::vx::TensorAttribute::CONSTANT, b_quant);
  tim::vx::TensorSpec out_spec(tim::vx::DataType::UINT8, out_shape,
                               tim::vx::TensorAttribute::OUTPUT, out_quant);

  auto in_t  = g->CreateTensor(in_spec);
  auto w_t   = g->CreateTensor(w_spec, w_q.data());
  auto b_t   = g->CreateTensor(b_spec, bias_q.data());
  auto out_t = g->CreateTensor(out_spec);
  if (!in_t || !w_t || !b_t || !out_t) {
    std::printf("[%-30s] FAIL  CreateTensor returned null\n", s.name);
    return false;
  }

  auto op = g->CreateOperation<tim::vx::ops::FullyConnected>(
      /*axis=*/0u, /*weights=*/s.N);
  (*op).BindInputs({in_t, w_t, b_t}).BindOutputs({out_t});

  // lenet's pattern: Compile *before* CopyDataToTensor for inputs.
  if (!g->Compile()) {
    std::printf("[%-30s] FAIL  Graph::Compile\n", s.name);
    return false;
  }
  if (!in_t->CopyDataToTensor(in_q.data(), in_q.size())) {
    std::printf("[%-30s] FAIL  CopyDataToTensor (input)\n", s.name);
    return false;
  }
  if (!g->Run()) {
    std::printf("[%-30s] FAIL  Graph::Run\n", s.name);
    return false;
  }

  std::vector<uint8_t> out_q(ref.size());
  if (!out_t->CopyDataFromTensor(out_q.data())) {
    std::printf("[%-30s] FAIL  CopyDataFromTensor\n", s.name);
    return false;
  }

  // ── Dequantize and compare ─────────────────────────────────────────
  // Tolerance budget for asymmetric UINT8 quantization:
  //   - Output dequant rounding: up to out_scale / 2 per element.
  //   - Input + weight quantization noise propagates as
  //     ~ (in_scale + w_scale) * sqrt(K) (random-walk model).
  //   Sum and pad to 1.5 * out_scale + 0.05 * |ref|. That works out to
  //   ~ a couple of LSBs on this chip and catches "wrong arithmetic"
  //   (the kinds of failures we'd actually want to detect).
  const float atol = 1.5f * out_scale;
  const float rtol = 0.05f;

  size_t mismatches = 0;
  float max_abs = 0.0f, max_rel = 0.0f;
  size_t worst_idx = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    float deq = dequantize_u8(out_q[i], out_scale, out_zp);
    float diff = std::fabs(deq - ref[i]);
    float rel = diff / (std::fabs(ref[i]) + 1e-30f);
    if (diff > max_abs) { max_abs = diff; worst_idx = i; }
    if (rel > max_rel) max_rel = rel;
    if (diff > atol + rtol * std::fabs(ref[i])) ++mismatches;
  }

  if (mismatches != 0) {
    float worst_deq = dequantize_u8(out_q[worst_idx], out_scale, out_zp);
    std::printf(
        "[%-30s] FAIL  M=%u K=%u N=%u  mismatches=%zu/%zu  "
        "max_abs=%.3e max_rel=%.3e (atol=%.3e)  at[%zu] ref=%.6f got=%.6f\n",
        s.name, s.M, s.K, s.N, mismatches, ref.size(), max_abs, max_rel, atol,
        worst_idx, ref[worst_idx], worst_deq);
    return false;
  }
  std::printf(
      "[%-30s] PASS  M=%u K=%u N=%u  out_scale=%.3e  max_abs=%.3e max_rel=%.3e\n",
      s.name, s.M, s.K, s.N, out_scale, max_abs, max_rel);
  return true;
}

}  // namespace

int main() {
  std::vector<Scenario> scenarios{
      // Small, hand-checkable.
      {"matmul_4x3_3x5",         4,   3,   5},
      {"dot_1x8_8x1",            1,   8,   1},
      {"matmul_2x6_6x2",         2,   6,   2},

      // Square — same op shape, larger accumulation.
      {"matmul_16x16_16x16",     16,  16,  16},
      {"matmul_32x32_32x32",     32,  32,  32},

      // Rectangular, larger reduction axis.
      {"matmul_8x64_64x8",        8,  64,   8},
      {"matmul_64x128_128x64",   64, 128,  64},

      // Vector × matrix and matrix × vector.
      {"vec_x_mat_1x32_32x16",    1,  32,  16},
      {"mat_x_vec_16x32_32x1",   16,  32,   1},

      // Larger reduction (stresses fp accumulation in the NPU).
      {"matmul_8x256_256x8",      8, 256,   8},
  };

  int failed = 0;
  for (const auto& s : scenarios)
    if (!run(s)) ++failed;

  std::printf("---\n%d / %zu scenario(s) failed\n", failed, scenarios.size());
  return failed == 0 ? 0 : 1;
}
