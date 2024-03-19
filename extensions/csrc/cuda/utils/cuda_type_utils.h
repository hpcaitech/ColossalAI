/*
 * This code from NVIDIA FasterTransformer:
 * https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/utils/cuda_type_utils.cuh
 */

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>

template <typename T>
inline __device__ T add(T a, T b) {
  return a + b;
}

template <>
inline __device__ half2 add(half2 a, half2 b) {
  return __hadd2(a, b);
}

template <>
inline __device__ half add(half a, half b) {
  return __hadd(a, b);
}

#if ENABLE_BF16
template <>
inline __device__ __nv_bfloat162 add(__nv_bfloat162 a, __nv_bfloat162 b) {
  return bf16hadd2(a, b);
}

template <>
inline __device__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b) {
  return bf16hadd(a, b);
}

#endif  // ENABLE_BF16

template <typename T>
inline __device__ T mul(T a, T b, T c) {
  return a * b * c;
}

template <>
inline __device__ half2 mul(half2 a, half2 b, half2 c) {
  return __hmul2(__hmul2(a, b), c);
}

#if ENABLE_BF16
template <>
inline __device__ __nv_bfloat16 mul(__nv_bfloat16 a, __nv_bfloat16 b,
                                    __nv_bfloat16 c) {
  return bf16hmul(a, b, c);
}

inline __device__ __nv_bfloat162 mul(__nv_bfloat162 a, __nv_bfloat162 b,
                                     __nv_bfloat162 c) {
  return bf16hmul2(a, b, c);
}
#endif  // ENABLE_BF16

template <typename T_OUT, typename T_IN>
__device__ inline T_OUT cuda_cast(T_IN val) {
  return val;
}

template <>
__device__ inline float2 cuda_cast<float2, int2>(int2 val) {
  return make_float2(val.x, val.y);
}
template <>
__device__ inline float2 cuda_cast<float2, float>(float val) {
  return make_float2(val, val);
}
template <>
__device__ inline float2 cuda_cast<float2, half2>(half2 val) {
  return __half22float2(val);
}
template <>
__device__ inline half2 cuda_cast<half2, float2>(float2 val) {
  return __float22half2_rn(val);
}
template <>
__device__ inline half2 cuda_cast<half2, float>(float val) {
  return __float2half2_rn(val);
}
template <>
__device__ inline half2 cuda_cast<half2, half>(half val) {
  return __half2half2(val);
}
template <>
__device__ inline float cuda_cast<float, half>(half val) {
  return __half2float(val);
}

// Get type2 from type or vice versa (applied to half and bfloat16)
template <typename T>
struct TypeConverter {
  using Type = half2;
};  // keep for generality

template <>
struct TypeConverter<half2> {
  using Type = at::Half;
};

template <>
struct TypeConverter<at::Half> {
  using Type = half2;
};

#if ENABLE_BF16
template <>
struct TypeConverter<__nv_bfloat162> {
  using Type = at::BFloat16;
};

template <>
struct TypeConverter<at::BFloat16> {
  using Type = __nv_bfloat162;
};
#endif  // ENABLE_BF16
