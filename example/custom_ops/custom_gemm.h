// custom_gemm.h — fp32 OpenCL GEMM custom op for the GC GPU.
//
// Lifted verbatim from TIM-VX/samples/custom_op_test/custom_gemm.{h,cc}
// so this project owns its custom-op surface (TIM-VX docs/customized_op.md
// explicitly supports hosting custom ops outside the TIM-VX tree).
//
// Used by timvx_runtime::matmul to bypass the NN-engine Matmul, which is
// unreliable on VIP9000Nano-DI; this kernel runs on the GC shader core.

#ifndef CUSTOM_OPS_CUSTOM_GEMM_H_
#define CUSTOM_OPS_CUSTOM_GEMM_H_

#include "tim/vx/ops/custom_base.h"

namespace tim {
namespace vx {
namespace ops {

class CustomGemm : public CustomOpBase {
 public:
  using ParamTuple = std::tuple<int,   /* M */
                                int,   /* K */
                                int,   /* N */
                                int,   /* ac2zero */
                                int,   /* bc2zero */
                                float, /* scale_a */
                                float, /* zp_a */
                                float, /* scale_b */
                                float, /* zp_b */
                                float, /* scale_out */
                                float  /* zp_out */>;

  CustomGemm(Graph* graph, bool trans_a, bool trans_b,
             ParamTuple tuple_list, uint32_t input_num = 2,
             uint32_t output_num = 1)
      : CustomOpBase(graph, input_num, output_num, CustomGemm::kernel_id_,
                     CustomGemm::kernel_name_),
        trans_a_(trans_a),
        trans_b_(trans_b) {
    tuple_list_.swap(tuple_list);
    param_transform(tuple_list_, param_list_);

    kernel_resource_ =
        "__kernel void gemm_F32F32toF32_2D(\n"
        "    __read_only image2d_t   inputA,\n"
        "    __read_only image2d_t   inputB,\n"
        "    __write_only image2d_t  output,\n"
        "    int M, int K, int N,\n"
        "    int ac2zero, int bc2zero,\n"
        "    float scale_a, float zp_a,\n"
        "    float scale_b, float zp_b,\n"
        "    float scale_out, float zp_out)\n"
        "{\n"
        "    int4 coord = (int4)(get_global_id(0), get_global_id(1), 0, 0);\n"
        "    float4 sum = (float4)(0);\n"
        "    for (; coord.z < K;) {\n"
        "        float4 tempA0 = read_imagef(inputA, coord.zy);\n"
        "        float4 tempB0 = read_imagef(inputB, coord.xz);\n"
        "        coord.z++;\n"
        "        sum = sum + tempA0 * tempB0;\n"
        "    }\n"
        "    write_imagef(output, coord.xy, sum);\n"
        "}\n";
  }

 protected:
  const char* kernel_NotTransA_NotTransB = "gemm_F32F32toF32_2D";
  ParamTuple tuple_list_;
  bool trans_a_;
  bool trans_b_;
  static const char* kernel_name_;
  static int32_t kernel_id_;

  void SetupShapeInfor() override {
    if (!trans_a_ && !trans_b_) {
      outputs_size_[0].push_back(inputs_size_[0][1]);
      outputs_size_[0].push_back(inputs_size_[1][0]);
    }
  }

  void SetupParams(std::vector<tim::vx::DataType> input_types,
                   std::string& build_option) override {
    if (!trans_a_ && !trans_b_ &&
        input_types[0] == tim::vx::DataType::FLOAT32 &&
        input_types[1] == tim::vx::DataType::FLOAT32) {
      func_name_ = kernel_NotTransA_NotTransB;
      build_option = "";
    }
  }

  void SetupEnqueue(uint32_t& dim, std::vector<size_t>& global_size,
                    std::vector<size_t>& local_size) override {
    dim = 3;
    local_size[0] = 0;
    local_size[1] = 0;
    local_size[2] = 0;
    global_size[0] = gpu_align(outputs_size_[0][0], 4);
    global_size[1] = gpu_align(outputs_size_[0][1], 4);
    global_size[2] = outputs_size_[0].size() > 2 ? outputs_size_[0][2] : 1;
  }

  std::shared_ptr<Operation> Clone(
      std::shared_ptr<Graph>& graph) const override {
    return graph->CreateOperation<CustomGemm>(
        trans_a_, trans_b_, this->tuple_list_, this->input_num_,
        this->output_num_);
  }
};

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif  // CUSTOM_OPS_CUSTOM_GEMM_H_
