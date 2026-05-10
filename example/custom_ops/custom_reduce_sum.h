// custom_reduce_sum.h — fp32 single-axis reduce_sum on the GC GPU.
//
// On VIP9000Nano-DI / A733 the NN-engine REDUCE kernel is INT-only —
// FP32/FP16/U8 fail at Compile() (op_probe table). For graphs that are
// still fp32 at the reduce site, we route reduce_sum through this
// OpenCL kernel rather than tim::vx::ops::ReduceSum.
//
// Two kernel variants share the same I/O contract; the helper picks
// whichever fits within the chip's image2d dimension limits:
//
//   * `CustomReduceSum`     — Layout A (axis-along-width).
//     Caller reshapes input to {inner * axis_size, outer}; the kernel
//     reads at (gx + k*inner, gy). Limit: inner*axis_size <= W_max.
//
//   * `CustomReduceSumTall` — Layout B (axis-along-height).
//     Caller reshapes input to {inner, axis_size * outer}; the kernel
//     reads at (gx, gy*axis_size + k). Limit: axis_size*outer <= H_max.
//
// Both produce the same 2-D output {inner, outer}; the caller does the
// keep_dims=true reshape back to the N-D output shape. Both walk
// `axis_size` accumulations per work item in fp32. `inner` is the in-
// memory stride of the reduction axis in TIM-VX innermost-first byte
// order (= product of dims at indices < K under our convention where
// the harness layoutConverts inputs at the function boundary).

#ifndef CUSTOM_OPS_CUSTOM_REDUCE_SUM_H_
#define CUSTOM_OPS_CUSTOM_REDUCE_SUM_H_

#include "tim/vx/ops/custom_base.h"

namespace tim {
namespace vx {
namespace ops {

class CustomReduceSum : public CustomOpBase {
 public:
  using ParamTuple = std::tuple<int /* axis_size */, int /* inner */>;

  CustomReduceSum(Graph* graph, int axis_size, int inner, int outer)
      : CustomOpBase(graph, /*input_num=*/1, /*output_num=*/1,
                     CustomReduceSum::kernel_id_,
                     CustomReduceSum::kernel_name_),
        axis_size_(axis_size),
        inner_(inner),
        outer_(outer) {
    ParamTuple tup{axis_size, inner};
    param_transform(tup, param_list_);

    kernel_resource_ =
        "__kernel void custom_reduce_sum_v3(\n"
        "    __read_only image2d_t  input,\n"
        "    __write_only image2d_t output,\n"
        "    int axis_size,\n"
        "    int inner)\n"
        "{\n"
        "    int gx = get_global_id(0);\n"
        "    int gy = get_global_id(1);\n"
        "    float sum = 0.0f;\n"
        "    for (int k = 0; k < axis_size; ++k) {\n"
        "        float4 v = read_imagef(input, (int2)(gx + k * inner, gy));\n"
        "        sum += v.x;\n"
        "    }\n"
        "    write_imagef(output, (int2)(gx, gy), (float4)(sum, sum, sum, sum));\n"
        "}\n";
  }

 protected:
  static const char* kernel_name_;
  static int32_t kernel_id_;
  int axis_size_;
  int inner_;
  int outer_;

  void SetupShapeInfor() override {
    // Output 2-D {inner, outer}: width = inner, height = outer. For
    // the inner=1 case this collapses to a width-1 image, which the
    // chip handles fine (we tested round-tripping at width=1 in the
    // earlier diagnostic round).
    outputs_size_[0].push_back(static_cast<uint32_t>(inner_));
    outputs_size_[0].push_back(static_cast<uint32_t>(outer_));
  }

  void SetupParams(std::vector<tim::vx::DataType> input_types,
                   std::string& build_option) override {
    func_name_ = "custom_reduce_sum_v3";
    build_option = "";
  }

  void SetupEnqueue(uint32_t& dim, std::vector<size_t>& global_size,
                    std::vector<size_t>& local_size) override {
    dim = 2;
    local_size[0] = 0;
    local_size[1] = 0;
    global_size[0] = static_cast<size_t>(inner_);
    global_size[1] = static_cast<size_t>(outer_);
  }

  std::shared_ptr<Operation> Clone(
      std::shared_ptr<Graph>& graph) const override {
    return graph->CreateOperation<CustomReduceSum>(axis_size_, inner_,
                                                    outer_);
  }
};

// Layout B variant: axis-along-height.  Use when `inner * axis_size`
// would overflow the image-width limit but `axis_size * outer` fits in
// height. Caller reshapes input to {inner, axis_size * outer}; the
// kernel walks gy_base + k along image-y for each output (gx, gy).
class CustomReduceSumTall : public CustomOpBase {
 public:
  using ParamTuple = std::tuple<int /* axis_size */>;

  CustomReduceSumTall(Graph* graph, int axis_size, int inner, int outer)
      : CustomOpBase(graph, /*input_num=*/1, /*output_num=*/1,
                     CustomReduceSumTall::kernel_id_,
                     CustomReduceSumTall::kernel_name_),
        axis_size_(axis_size),
        inner_(inner),
        outer_(outer) {
    ParamTuple tup{axis_size};
    param_transform(tup, param_list_);

    kernel_resource_ =
        "__kernel void custom_reduce_sum_tall_v1(\n"
        "    __read_only image2d_t  input,\n"
        "    __write_only image2d_t output,\n"
        "    int axis_size)\n"
        "{\n"
        "    int gx = get_global_id(0);\n"
        "    int gy = get_global_id(1);\n"
        "    int row_base = gy * axis_size;\n"
        "    float sum = 0.0f;\n"
        "    for (int k = 0; k < axis_size; ++k) {\n"
        "        float4 v = read_imagef(input, (int2)(gx, row_base + k));\n"
        "        sum += v.x;\n"
        "    }\n"
        "    write_imagef(output, (int2)(gx, gy), (float4)(sum, sum, sum, sum));\n"
        "}\n";
  }

 protected:
  static const char* kernel_name_;
  static int32_t kernel_id_;
  int axis_size_;
  int inner_;
  int outer_;

  void SetupShapeInfor() override {
    outputs_size_[0].push_back(static_cast<uint32_t>(inner_));
    outputs_size_[0].push_back(static_cast<uint32_t>(outer_));
  }

  void SetupParams(std::vector<tim::vx::DataType> input_types,
                   std::string& build_option) override {
    func_name_ = "custom_reduce_sum_tall_v1";
    build_option = "";
  }

  void SetupEnqueue(uint32_t& dim, std::vector<size_t>& global_size,
                    std::vector<size_t>& local_size) override {
    dim = 2;
    local_size[0] = 0;
    local_size[1] = 0;
    global_size[0] = static_cast<size_t>(inner_);
    global_size[1] = static_cast<size_t>(outer_);
  }

  std::shared_ptr<Operation> Clone(
      std::shared_ptr<Graph>& graph) const override {
    return graph->CreateOperation<CustomReduceSumTall>(axis_size_, inner_,
                                                       outer_);
  }
};

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif  // CUSTOM_OPS_CUSTOM_REDUCE_SUM_H_
