// timvx_smoke.cpp — minimal repro for `Create tensor fail!`.
//
// Build + run via timvx_smoke.bash — uses the same LD_PRELOAD / LD_LIBRARY_PATH
// setup as the lowered runner, so the only variable left is how we drive
// tim::vx itself.
//
// CreateTensor returning a non-null shared_ptr is NOT enough — TensorImpl
// keeps the wrapper alive even when its inner Init() failed (id_ = NA).
// Each scenario therefore goes further: write a known byte pattern with
// CopyDataToTensor (for writeable attrs), read it back with
// CopyDataFromTensor (for readable attrs), and confirm the bytes match.
// That's the actual usability test the lowered runner is implicitly
// performing each time it builds a const_tensor.

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"
#include "tim/vx/types.h"

namespace {

struct Scenario {
  const char* name;
  std::vector<uint32_t> shape;
  tim::vx::DataType dtype;
  tim::vx::TensorAttribute attr;
  bool pass_data;  // CONSTANT tensors need data at create time
};

size_t bytesPerElem(tim::vx::DataType dt) {
  switch (dt) {
    case tim::vx::DataType::FLOAT16:
    case tim::vx::DataType::INT16:
    case tim::vx::DataType::UINT16: return 2;
    case tim::vx::DataType::INT8:
    case tim::vx::DataType::UINT8:
    case tim::vx::DataType::BOOL8:  return 1;
    case tim::vx::DataType::INT64:  return 8;
    default:                        return 4;  // FLOAT32 / INT32 / UINT32 etc.
  }
}

bool run(const Scenario& s) {
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
  tim::vx::TensorSpec spec(s.dtype, s.shape, s.attr);

  size_t numel = 1;
  for (auto d : s.shape) numel *= d;
  if (numel == 0) numel = 1;  // rank-0 = single scalar
  size_t nbytes = numel * bytesPerElem(s.dtype);

  // Deterministic write pattern: byte i = (i * 31) ^ 0xA5.
  std::vector<uint8_t> writeBuf(nbytes);
  for (size_t i = 0; i < nbytes; ++i)
    writeBuf[i] = static_cast<uint8_t>((i * 31u) ^ 0xA5u);

  // CONSTANT must be supplied at create time; everything else creates blank
  // and we'll CopyDataToTensor below.
  const void* createData = (s.attr == tim::vx::TensorAttribute::CONSTANT)
                               ? writeBuf.data()
                               : nullptr;
  auto t = g->CreateTensor(spec, createData);
  if (!t) {
    std::printf("[%-30s] FAIL  CreateTensor returned null shared_ptr\n",
                s.name);
    return false;
  }

  bool writeable = s.attr != tim::vx::TensorAttribute::TRANSIENT &&
                   s.attr != tim::vx::TensorAttribute::OUTPUT;
  bool readable  = s.attr != tim::vx::TensorAttribute::TRANSIENT;

  // For non-CONSTANT writeables, write the pattern explicitly. (CONSTANT
  // already had it via createData.)
  if (writeable && s.attr != tim::vx::TensorAttribute::CONSTANT) {
    if (!t->CopyDataToTensor(writeBuf.data(),
                             static_cast<uint32_t>(nbytes))) {
      std::printf("[%-30s] FAIL  CopyDataToTensor (size=%zu)\n", s.name,
                  nbytes);
      return false;
    }
  }

  // Read back and compare. OUTPUT contents are undefined before Run(), so
  // we only compare for INPUT/CONSTANT.
  if (readable && s.attr != tim::vx::TensorAttribute::OUTPUT) {
    std::vector<uint8_t> readBuf(nbytes, 0xCC);
    if (!t->CopyDataFromTensor(readBuf.data())) {
      std::printf("[%-30s] FAIL  CopyDataFromTensor (size=%zu)\n", s.name,
                  nbytes);
      return false;
    }
    size_t mismatches = 0;
    for (size_t i = 0; i < nbytes; ++i)
      if (readBuf[i] != writeBuf[i]) ++mismatches;
    if (mismatches != 0) {
      std::printf(
          "[%-30s] FAIL  read != write   first byte: read=0x%02x expect=0x%02x"
          "  (%zu of %zu mismatched)\n",
          s.name, readBuf[0], writeBuf[0], mismatches, nbytes);
      return false;
    }
  }

  std::printf("[%-30s] PASS  rank=%zu nbytes=%zu\n", s.name, s.shape.size(),
              nbytes);
  return true;
}

}  // namespace

int main() {
  using DT = tim::vx::DataType;
  using TA = tim::vx::TensorAttribute;
  std::vector<Scenario> scenarios{
      // The image input the runner is currently failing on.
      {"input_1x3x224x224_fp32", {1, 3, 224, 224}, DT::FLOAT32, TA::INPUT, false},

      // Smaller variants — pin whether shape/rank is the trigger.
      {"input_1x224x224x3_fp32", {1, 224, 224, 3}, DT::FLOAT32, TA::INPUT, false},
      {"input_2d_4x4_fp32",      {4, 4},          DT::FLOAT32, TA::INPUT, false},
      {"input_1d_64_fp32",       {64},            DT::FLOAT32, TA::INPUT, false},

      // CONSTANTs with data — what most of the lowered body emits.
      {"const_1d_64_fp32",       {64},            DT::FLOAT32, TA::CONSTANT, true},
      {"const_1d_1000_fp32",     {1000},          DT::FLOAT32, TA::CONSTANT, true},
      {"const_4d_1x3x224x224",   {1, 3, 224, 224},DT::FLOAT32, TA::CONSTANT, true},

      // OUTPUT — what we mark the function-return tensor as.
      {"output_1x1000_fp32",     {1, 1000},       DT::FLOAT32, TA::OUTPUT, false},

      // TRANSIENT intermediates — most of the body's helper calls.
      {"transient_1x224x224x3",  {1, 224, 224, 3},DT::FLOAT32, TA::TRANSIENT, false},
  };

  int failed = 0;
  for (const auto& s : scenarios) {
    if (!run(s)) ++failed;
  }
  std::printf("---\n%d scenario(s) failed\n", failed);
  return failed == 0 ? 0 : 1;
}
