#include "custom_reduce_sum.h"

namespace tim {
namespace vx {
namespace ops {

const char* CustomReduceSum::kernel_name_ = "custom_reduce_sum_v3";
int32_t CustomReduceSum::kernel_id_ = -1 * (++gobal_kernel_id_);

const char* CustomReduceSumTall::kernel_name_ = "custom_reduce_sum_tall_v1";
int32_t CustomReduceSumTall::kernel_id_ = -1 * (++gobal_kernel_id_);

}  // namespace ops
}  // namespace vx
}  // namespace tim
