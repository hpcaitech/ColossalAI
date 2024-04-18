#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <torch/extension.h>

#include <cfloat>

namespace colossalAI {
namespace cuda {
namespace utils {

struct bfloat164 {
  __nv_bfloat162 x;
  __nv_bfloat162 y;
};
struct bfloat168 {
  __nv_bfloat162 x;
  __nv_bfloat162 y;
  __nv_bfloat162 z;
  __nv_bfloat162 w;
};

struct half4 {
  half2 x;
  half2 y;
};
struct half8 {
  half2 x;
  half2 y;
  half2 z;
  half2 w;
};

struct float4_ {
  float2 x;
  float2 y;
};
struct float8_ {
  float2 x;
  float2 y;
  float2 z;
  float2 w;
};

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
VEC_TYPE_TRAITS_SPECIALIZATION(float, 8, float8_)
VEC_TYPE_TRAITS_SPECIALIZATION(uint8_t, 2, half)
VEC_TYPE_TRAITS_SPECIALIZATION(uint8_t, 4, half2)
VEC_TYPE_TRAITS_SPECIALIZATION(uint8_t, 8, float2)
VEC_TYPE_TRAITS_SPECIALIZATION(__nv_bfloat16, 2, __nv_bfloat162);
VEC_TYPE_TRAITS_SPECIALIZATION(__nv_bfloat16, 4, bfloat164);
VEC_TYPE_TRAITS_SPECIALIZATION(__nv_bfloat16, 8, bfloat168);
VEC_TYPE_TRAITS_SPECIALIZATION(half, 2, half2);
VEC_TYPE_TRAITS_SPECIALIZATION(half, 4, half4);
VEC_TYPE_TRAITS_SPECIALIZATION(half, 8, half8);

#undef VEC_TYPE_TRAITS_SPECIALIZATION

#define FLOATVEC_TYPE_TRAITS_SPECIALIZATION(T, FLOATT, ARGS...) \
  template <ARGS>                                               \
  struct FloatVecTypeTrait<T> {                                 \
    using Type = FLOATT;                                        \
  };

FLOATVEC_TYPE_TRAITS_SPECIALIZATION(float2, float2)
FLOATVEC_TYPE_TRAITS_SPECIALIZATION(float4, float4)
FLOATVEC_TYPE_TRAITS_SPECIALIZATION(__nv_bfloat162, float2);
FLOATVEC_TYPE_TRAITS_SPECIALIZATION(bfloat164, float4_);
FLOATVEC_TYPE_TRAITS_SPECIALIZATION(bfloat168, float8_);
FLOATVEC_TYPE_TRAITS_SPECIALIZATION(half2, float2);
FLOATVEC_TYPE_TRAITS_SPECIALIZATION(half4, float4_);
FLOATVEC_TYPE_TRAITS_SPECIALIZATION(half8, float8_);

#undef FLOATVEC_TYPE_TRAITS_SPECIALIZATION

}  // namespace utils
}  // namespace cuda
}  // namespace colossalAI
