
#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

#include "common/vec_type_traits.h"
#include "funcs/cast_functor.h"

namespace colossalAI {
namespace cuda {
namespace utils {

template <typename T, int VecSize>
__device__ __inline__ void copy_vector(T *dst, const T *src) {
  using VT = typename common::VecTypeTrait<T, VecSize>::Type;
  // Note(LiuYang): Here static_cast can't be used for cast between two pointer
  *(reinterpret_cast<VT *>(dst)) = *(reinterpret_cast<const VT *>(src));
}

template <>
__device__ __inline__ void copy_vector<float, 8>(float *dst, const float *src) {
  // Since the maximum memory alignment length is 128 bits, we choose float4
  // here.
  *(reinterpret_cast<float4 *>(dst)) = *(reinterpret_cast<const float4 *>(src));
  *(reinterpret_cast<float4 *>(dst + 4)) =
      *(reinterpret_cast<const float4 *>(src + 4));
}

template <typename T, int VecSize>
__device__ __inline__ void copy_zero_vector(T *dst) {
  using VT = typename common::VecTypeTrait<T, VecSize>::Type;
  *(reinterpret_cast<VT *>(dst)) = funcs::CastFunctor<float, VT>()(0.0f);
}

template <typename T>
int get_vec_size(const torch::Tensor &tensor) {
  uint64_t address = reinterpret_cast<uint64_t>(tensor.data_ptr<T>());
  const int max_aligned_size = 128;
  const int dtype_size = sizeof(T) * 8;

  const int vec_size = max_aligned_size / sizeof(T) / 8;

  // Note(LiuYang): Performance of situation of which
  // vec_size equals to 8 need to be profiled in the future
  // if (address % (dtype_size * 8) == 0) {
  //   return std::min(8, vec_size);
  // }
  if (address % (dtype_size * 4) == 0) {
    return std::min(4, vec_size);
  } else if (address % (dtype_size * 2) == 0) {
    return std::min(2, vec_size);
  } else {
    return 1;
  }
}

}  // namespace utils
}  // namespace cuda
}  // namespace colossalAI
