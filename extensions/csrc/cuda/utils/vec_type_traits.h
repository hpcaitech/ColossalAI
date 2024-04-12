#pragma once

#include <c10/macros/Macros.h>
#include <cuda_fp16.h>
#include <stdint.h>

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
VEC_TYPE_TRAITS_SPECIALIZATION(c10::BFloat16, 2, float)
VEC_TYPE_TRAITS_SPECIALIZATION(c10::BFloat16, 4, float2)
VEC_TYPE_TRAITS_SPECIALIZATION(c10::BFloat16, 8, float4)
VEC_TYPE_TRAITS_SPECIALIZATION(c10::Half, 2, float)
VEC_TYPE_TRAITS_SPECIALIZATION(c10::Half, 4, float2)
VEC_TYPE_TRAITS_SPECIALIZATION(c10::Half, 8, float4)
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
