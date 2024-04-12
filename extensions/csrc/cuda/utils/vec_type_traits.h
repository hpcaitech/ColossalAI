#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <torch/extension.h>

#include <cfloat>

namespace colossalAI {
namespace cuda {
namespace utils {

template <typename T, int VecSize>
struct VecTypeTrait {};

#define VEC_TYPE_TRAITS_SPECIALIZATION(T, VEC_SIZE, VECT, ARGS...) \
  template <ARGS>                                                  \
  struct VecTypeTrait<T, VEC_SIZE> {                               \
    using Type = VECT;                                             \
  };

VEC_TYPE_TRAITS_SPECIALIZATION(T, 1, T, typename T)
VEC_TYPE_TRAITS_SPECIALIZATION(at::BFloat16, 1, __nv_bfloat16)
VEC_TYPE_TRAITS_SPECIALIZATION(at::BFloat16, 2, __nv_bfloat162)
VEC_TYPE_TRAITS_SPECIALIZATION(at::BFloat16, 4, float2)
VEC_TYPE_TRAITS_SPECIALIZATION(at::BFloat16, 8, float4)
VEC_TYPE_TRAITS_SPECIALIZATION(at::Half, 1, half)
VEC_TYPE_TRAITS_SPECIALIZATION(at::Half, 2, half2)
VEC_TYPE_TRAITS_SPECIALIZATION(at::Half, 4, float2)
VEC_TYPE_TRAITS_SPECIALIZATION(at::Half, 8, float4)
VEC_TYPE_TRAITS_SPECIALIZATION(float, 2, float2)
VEC_TYPE_TRAITS_SPECIALIZATION(float, 4, float4)
VEC_TYPE_TRAITS_SPECIALIZATION(float, 8, float4)
VEC_TYPE_TRAITS_SPECIALIZATION(uint8_t, 2, half)
VEC_TYPE_TRAITS_SPECIALIZATION(uint8_t, 4, half2)
VEC_TYPE_TRAITS_SPECIALIZATION(uint8_t, 8, float2)

#undef VEC_TYPE_TRAITS_SPECIALIZATION

}  // namespace utils
}  // namespace cuda
}  // namespace colossalAI
