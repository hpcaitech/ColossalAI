#pragma once

#if defined(COLOSSAL_WITH_CUDA)
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif

#include <ATen/ATen.h>
#include <stdint.h>

#include "common/data_type.h"

namespace colossalAI {
namespace common {

template <typename T, int VecSize>
struct VecTypeTrait {};

template <typename T>
struct FloatVecTypeTrait {};

#define VEC_TYPE_TRAITS_SPECIALIZATION(T, VEC_SIZE, VECT, ARGS...) \
  template <ARGS>                                                  \
  struct VecTypeTrait<T, VEC_SIZE> {                               \
    using Type = VECT;                                             \
  };

VEC_TYPE_TRAITS_SPECIALIZATION(T, 1, T, typename T)

#if defined(COLOSSAL_WITH_CUDA)

VEC_TYPE_TRAITS_SPECIALIZATION(at::BFloat16, 1, __nv_bfloat16)
VEC_TYPE_TRAITS_SPECIALIZATION(at::BFloat16, 2, __nv_bfloat162)
VEC_TYPE_TRAITS_SPECIALIZATION(at::BFloat16, 4, float2)
VEC_TYPE_TRAITS_SPECIALIZATION(at::BFloat16, 8, float4)
VEC_TYPE_TRAITS_SPECIALIZATION(at::Half, 1, half)
VEC_TYPE_TRAITS_SPECIALIZATION(at::Half, 2, half2)
VEC_TYPE_TRAITS_SPECIALIZATION(at::Half, 4, float2)
VEC_TYPE_TRAITS_SPECIALIZATION(at::Half, 8, float4)

VEC_TYPE_TRAITS_SPECIALIZATION(uint8_t, 2, uint16_t)
VEC_TYPE_TRAITS_SPECIALIZATION(uint8_t, 4, uint32_t)
VEC_TYPE_TRAITS_SPECIALIZATION(uint8_t, 8, uint2)
VEC_TYPE_TRAITS_SPECIALIZATION(__nv_bfloat16, 2, __nv_bfloat162);
VEC_TYPE_TRAITS_SPECIALIZATION(__nv_bfloat16, 4, dtype::bfloat164);
VEC_TYPE_TRAITS_SPECIALIZATION(__nv_bfloat16, 8, dtype::bfloat168);
VEC_TYPE_TRAITS_SPECIALIZATION(half, 2, half2);
VEC_TYPE_TRAITS_SPECIALIZATION(half, 4, dtype::half4);
VEC_TYPE_TRAITS_SPECIALIZATION(half, 8, dtype::half8);
VEC_TYPE_TRAITS_SPECIALIZATION(float, 2, float2)
VEC_TYPE_TRAITS_SPECIALIZATION(float, 4, float4)
VEC_TYPE_TRAITS_SPECIALIZATION(float, 8, dtype::float8)
#endif /* defined(COLOSSAL_WITH_CUDA) */

#undef VEC_TYPE_TRAITS_SPECIALIZATION

#define FLOATVEC_TYPE_TRAITS_SPECIALIZATION(T, FLOATT, ARGS...) \
  template <ARGS>                                               \
  struct FloatVecTypeTrait<T> {                                 \
    using Type = FLOATT;                                        \
  };

#if defined(COLOSSAL_WITH_CUDA)
FLOATVEC_TYPE_TRAITS_SPECIALIZATION(float2, float2)
FLOATVEC_TYPE_TRAITS_SPECIALIZATION(float4, float4)
FLOATVEC_TYPE_TRAITS_SPECIALIZATION(__nv_bfloat162, float2);
FLOATVEC_TYPE_TRAITS_SPECIALIZATION(dtype::bfloat164, float4);
FLOATVEC_TYPE_TRAITS_SPECIALIZATION(dtype::bfloat168, dtype::float8);
FLOATVEC_TYPE_TRAITS_SPECIALIZATION(half2, float2);
FLOATVEC_TYPE_TRAITS_SPECIALIZATION(dtype::half4, float4);
FLOATVEC_TYPE_TRAITS_SPECIALIZATION(dtype::half8, dtype::float8);
#endif /* COLOSSAL_WITH_CUDA */

#undef FLOATVEC_TYPE_TRAITS_SPECIALIZATION
}  // namespace common
}  // namespace colossalAI
