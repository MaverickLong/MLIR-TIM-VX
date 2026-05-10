#include "custom_gemm.h"

namespace tim {
namespace vx {
namespace ops {

const char* CustomGemm::kernel_name_ = "custom_gemm_v1";
int32_t CustomGemm::kernel_id_ = -1 * (++gobal_kernel_id_);

}  // namespace ops
}  // namespace vx
}  // namespace tim
