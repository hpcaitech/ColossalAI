/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * Copyright (c) 2024, The Colossal-AI team.
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <float.h>

#include "attention_generic.h"
#include "cuda_type_utils.h"

// Define custom BF16 vector data types.
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

template <>
struct VecType<__nv_bfloat16, 2> {
  using Type = __nv_bfloat162;
};
template <>
struct VecType<__nv_bfloat16, 4> {
  using Type = bfloat164;
};
template <>
struct VecType<__nv_bfloat16, 8> {
  using Type = bfloat168;
};

template <>
struct VecType<half, 2> {
  using Type = half2;
};
template <>
struct VecType<half, 4> {
  using Type = half4;
};
template <>
struct VecType<half, 8> {
  using Type = half8;
};

template <>
struct VecType<float, 2> {
  using Type = float2;
};
template <>
struct VecType<float, 4> {
  using Type = float4;
};
template <>
struct VecType<float, 8> {
  using Type = float8_;
};

template <>
struct FloatVec<__nv_bfloat162> {
  using Type = float2;
};
template <>
struct FloatVec<bfloat164> {
  using Type = float4_;
};
template <>
struct FloatVec<bfloat168> {
  using Type = float8_;
};

template <>
struct FloatVec<half2> {
  using Type = float2;
};
template <>
struct FloatVec<half4> {
  using Type = float4_;
};
template <>
struct FloatVec<half8> {
  using Type = float8_;
};

template <>
struct FloatVec<float2> {
  using Type = float2;
};
template <>
struct FloatVec<float4> {
  using Type = float4;
};

template <typename T>
inline __device__ void zero(T& dst) {
  constexpr int WORDS = sizeof(T) / 4;
  union {
    T raw;
    uint32_t words[WORDS];
  } tmp;

#pragma unroll
  for (int ii = 0; ii < WORDS; ii++) {
    tmp.words[ii] = 0u;
  }
  dst = tmp.raw;
}

#define SHFL_XOR_SYNC(var, lane_mask) \
  __shfl_xor_sync(uint32_t(-1), var, lane_mask)
#define SHFL_SYNC(var, src_lane) __shfl_sync(uint32_t(-1), var, src_lane)

inline __device__ __nv_bfloat162 bf162bf162(const __nv_bfloat16 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  __nv_bfloat162 val2;
  val2.x = val;
  val2.y = val;
  return val2;
#else
  return __bfloat162bfloat162(val);
#endif
}

inline __device__ float2 bf1622float2(const __nv_bfloat162 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float2 f_val;
  f_val.x = __low2float(val);
  f_val.y = __high2float(val);
  return f_val;
#else
  return __bfloat1622float2(val);
#endif
}

template <>
inline __device__ float2 mul(float2 a, float2 b) {
  float2 c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  return c;
}

template <>
inline __device__ float4 mul(float4 a, float4 b) {
  float4 c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  c.z = a.z * b.z;
  c.w = a.w * b.w;
  return c;
}

template <>
inline __device__ float2 mul(__nv_bfloat162 a, __nv_bfloat162 b) {
  float2 fa = bf1622float2(a);
  float2 fb = bf1622float2(b);
  return mul<float2, float2, float2>(fa, fb);
}

template <>
inline __device__ float4_ mul(bfloat164 a, bfloat164 b) {
  float4_ fc;
  fc.x = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.x, b.x);
  fc.y = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.y, b.y);
  return fc;
}

template <>
inline __device__ float8_ mul(bfloat168 a, bfloat168 b) {
  float8_ fc;
  fc.x = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.x, b.x);
  fc.y = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.y, b.y);
  fc.z = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.z, b.z);
  fc.w = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.w, b.w);
  return fc;
}

template <>
inline __device__ float2 mul(half2 a, half2 b) {
  float2 fa = __half22float2(a);
  float2 fb = __half22float2(b);
  return mul<float2, float2, float2>(fa, fb);
}

template <>
inline __device__ float4_ mul(half4 a, half4 b) {
  float4_ fc;
  fc.x = mul<float2, half2, half2>(a.x, b.x);
  fc.y = mul<float2, half2, half2>(a.y, b.y);
  return fc;
}

template <>
inline __device__ float8_ mul(half8 a, half8 b) {
  float8_ fc;
  fc.x = mul<float2, half2, half2>(a.x, b.x);
  fc.y = mul<float2, half2, half2>(a.y, b.y);
  fc.z = mul<float2, half2, half2>(a.z, b.z);
  fc.w = mul<float2, half2, half2>(a.w, b.w);
  return fc;
}

template <>
inline __device__ float sum(float2 v) {
  return v.x + v.y;
}

template <>
inline __device__ float sum(float4 v) {
  return v.x + v.y + v.z + v.w;
}

template <>
inline __device__ float sum(float4_ v) {
  return v.x.x + v.x.y + v.y.x + v.y.y;
}

template <>
inline __device__ float sum(float8_ v) {
  return v.x.x + v.x.y + v.y.x + v.y.y + v.z.x + v.z.y + v.w.x + v.w.y;
}

template <typename T, typename TFLOAT>
inline __device__ void from_float(T& dst, TFLOAT src) {
  dst = src;
}

template <>
inline __device__ void from_float(__nv_bfloat16& dst, float src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  dst = __float2bfloat16_rn(src);
#endif
}

template <>
inline __device__ void from_float(__nv_bfloat162& dst, float2 src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  dst = __float22bfloat162_rn(src);
#endif
}

template <>
inline __device__ void from_float(bfloat164& dst, float4 src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  dst.x = __floats2bfloat162_rn(src.x, src.y);
  dst.y = __floats2bfloat162_rn(src.z, src.w);
#endif
}

template <>
inline __device__ void from_float(bfloat164& dst, float4_ src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  dst.x = __float22bfloat162_rn(src.x);
  dst.y = __float22bfloat162_rn(src.y);
#endif
}

template <>
inline __device__ void from_float(bfloat168& dst, float8_ src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  dst.x = __float22bfloat162_rn(src.x);
  dst.y = __float22bfloat162_rn(src.y);
  dst.z = __float22bfloat162_rn(src.z);
  dst.w = __float22bfloat162_rn(src.w);
#endif
}

template <>
inline __device__ void from_float(half& dst, float src) {
  dst = __float2half_rn(src);
}

template <>
inline __device__ void from_float(half2& dst, float2 src) {
  dst = __float22half2_rn(src);
}

template <>
inline __device__ void from_float(half4& dst, float4 src) {
  dst.x = __floats2half2_rn(src.x, src.y);
  dst.y = __floats2half2_rn(src.z, src.w);
}

template <>
inline __device__ void from_float(half4& dst, float4_ src) {
  dst.x = __float22half2_rn(src.x);
  dst.y = __float22half2_rn(src.y);
}

template <>
inline __device__ void from_float(half8& dst, float8_ src) {
  dst.x = __float22half2_rn(src.x);
  dst.y = __float22half2_rn(src.y);
  dst.z = __float22half2_rn(src.z);
  dst.w = __float22half2_rn(src.w);
}

template <>
inline __device__ float fma(__nv_bfloat16 a, __nv_bfloat16 b, float c) {
  return __bfloat162float(a) * __bfloat162float(b) + c;
}

template <>
inline __device__ float fma(half a, half b, float c) {
  return __half2float(a) * __half2float(b) + c;
}

template <>
inline __device__ float2 fma(float2 a, float2 b, float2 c) {
  float2 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  return d;
}

template <>
inline __device__ float2 fma(float a, float2 b, float2 c) {
  float2 d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  return d;
}

template <>
inline __device__ float2 fma(half2 a, half2 b, float2 c) {
  float2 fa = __half22float2(a);
  float2 fb = __half22float2(b);
  return fma(fa, fb, c);
}

template <>
inline __device__ float2 fma(half a, half2 b, float2 c) {
  return fma(__half2half2(a), b, c);
}

template <>
inline __device__ float4_ fma(half4 a, half4 b, float4_ c) {
  float4_ fd;
  fd.x = fma(a.x, b.x, c.x);
  fd.y = fma(a.y, b.y, c.y);
  return fd;
}

template <>
inline __device__ float4_ fma(half a, half4 b, float4_ c) {
  half2 s = __half2half2(a);
  float4_ fd;
  fd.x = fma(s, b.x, c.x);
  fd.y = fma(s, b.y, c.y);
  return fd;
}

template <>
inline __device__ float8_ fma(half8 a, half8 b, float8_ c) {
  float8_ fd;
  fd.x = fma(a.x, b.x, c.x);
  fd.y = fma(a.y, b.y, c.y);
  fd.z = fma(a.z, b.z, c.z);
  fd.w = fma(a.w, b.w, c.w);
  return fd;
}

template <>
inline __device__ float8_ fma(half a, half8 b, float8_ c) {
  half2 s = __half2half2(a);
  float8_ fd;
  fd.x = fma(s, b.x, c.x);
  fd.y = fma(s, b.y, c.y);
  fd.z = fma(s, b.z, c.z);
  fd.w = fma(s, b.w, c.w);
  return fd;
}

template <>
inline __device__ float2 fma(__nv_bfloat162 a, __nv_bfloat162 b, float2 c) {
  float2 fa = bf1622float2(a);
  float2 fb = bf1622float2(b);
  return fma(fa, fb, c);
}

template <>
inline __device__ float2 fma(__nv_bfloat16 a, __nv_bfloat162 b, float2 c) {
  return fma(bf162bf162(a), b, c);
}

template <>
inline __device__ float4_ fma(bfloat164 a, bfloat164 b, float4_ c) {
  float4_ fd;
  fd.x = fma(a.x, b.x, c.x);
  fd.y = fma(a.y, b.y, c.y);
  return fd;
}

template <>
inline __device__ float4_ fma(__nv_bfloat16 a, bfloat164 b, float4_ c) {
  __nv_bfloat162 s = bf162bf162(a);
  float4_ fd;
  fd.x = fma(s, b.x, c.x);
  fd.y = fma(s, b.y, c.y);
  return fd;
}

template <>
inline __device__ float8_ fma(bfloat168 a, bfloat168 b, float8_ c) {
  float8_ fd;
  fd.x = fma(a.x, b.x, c.x);
  fd.y = fma(a.y, b.y, c.y);
  fd.z = fma(a.z, b.z, c.z);
  fd.w = fma(a.w, b.w, c.w);
  return fd;
}

template <>
inline __device__ float8_ fma(__nv_bfloat16 a, bfloat168 b, float8_ c) {
  __nv_bfloat162 s = bf162bf162(a);
  float8_ fd;
  fd.x = fma(s, b.x, c.x);
  fd.y = fma(s, b.y, c.y);
  fd.z = fma(s, b.z, c.z);
  fd.w = fma(s, b.w, c.w);
  return fd;
}

template <>
inline __device__ float4 fma(float4 a, float4 b, float4 c) {
  float4 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  d.z = fma(a.z, b.z, c.z);
  d.w = fma(a.w, b.w, c.w);
  return d;
}

template <>
inline __device__ float4 fma(float a, float4 b, float4 c) {
  float4 d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  d.z = fma(a, b.z, c.z);
  d.w = fma(a, b.w, c.w);
  return d;
}

// Q*K^T operation.
template <int NUM_THREADS_PER_TOKEN, typename Vec, int N>
inline __device__ float qk_dot_(const Vec (&q)[N], const Vec (&k)[N]) {
  using A_vec = typename FloatVec<Vec>::Type;
  // Compute the parallel products for Q*K^T (treat vector lanes separately).
  A_vec qk_vec = mul<A_vec, Vec, Vec>(q[0], k[0]);
#pragma unroll
  for (int ii = 1; ii < N; ii++) {
    qk_vec = fma(q[ii], k[ii], qk_vec);
  }

  // Finalize the reduction across lanes.
  float qk = sum(qk_vec);
#pragma unroll
  for (int mask = (NUM_THREADS_PER_TOKEN >> 1); mask > 0; mask >>= 1) {
    qk += SHFL_XOR_SYNC(qk, mask);
  }
  return qk;
}

template <typename T, int NUM_THREADS_PER_TOKEN>
struct Qk_dot {
  template <typename Vec, int N>
  static inline __device__ float dot(const Vec (&q)[N], const Vec (&k)[N]) {
    return qk_dot_<NUM_THREADS_PER_TOKEN>(q, k);
  }
};
