#pragma once

#if defined(COLOSSAL_WITH_CUDA)
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

#include <functional>

#include "common/data_type.h"
#include "common/micros.h"

// Note(LiuYang): This file provides base math operation for data type
// include POD and cuda built-in type such as half and __nv_bfloat16

namespace colossalAI {
namespace funcs {

template <typename From, typename To>
struct CastFunctor : public std::unary_function<From, To> {
  HOSTDEVICE To operator()(From val) { return static_cast<To>(val); }
};

#define COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(FROM, TO, STMTS,           \
                                             FUNCTION_MODIFIER)         \
  template <>                                                           \
  struct CastFunctor<FROM, TO> : public std::unary_function<FROM, TO> { \
    FUNCTION_MODIFIER TO operator()(FROM val) STMTS                     \
  };

#if defined(COLOSSAL_WITH_CUDA)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    int2, float2, { return make_float2(val.x, val.y); }, DEVICE)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    float, float2, { return make_float2(val, val); }, DEVICE)

COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    half2, float2, { return __half22float2(val); }, DEVICE)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    float2, half2, { return __float22half2_rn(val); }, DEVICE)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    float, half, { return __float2half_rn(val); }, DEVICE)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    float, half2, { return __float2half2_rn(val); }, DEVICE)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    half, half2, { return __half2half2(val); }, DEVICE)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    half, float, { return __half2float(val); }, DEVICE)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    float4, dtype::half4,
    {
      dtype::half4 dst;
      dst.x = __floats2half2_rn(val.x, val.y);
      dst.y = __floats2half2_rn(val.z, val.w);
      return dst;
    },
    DEVICE)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    dtype::float4_, dtype::half4,
    {
      dtype::half4 dst;
      dst.x = __float22half2_rn(val.x);
      dst.y = __float22half2_rn(val.y);
      return dst;
    },
    DEVICE)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    dtype::float8_, dtype::half8,
    {
      dtype::half8 dst;
      dst.x = __float22half2_rn(val.x);
      dst.y = __float22half2_rn(val.y);
      dst.z = __float22half2_rn(val.z);
      dst.w = __float22half2_rn(val.w);
      return dst;
    },
    DEVICE)

COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    float, __nv_bfloat162, { return __float2bfloat162_rn(val); }, DEVICE)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    float, __nv_bfloat16, { return __float2bfloat16_rn(val); }, DEVICE)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    float4, dtype::bfloat164,
    {
      dtype::bfloat164 dst;
      dst.x = __floats2bfloat162_rn(val.x, val.y);
      dst.y = __floats2bfloat162_rn(val.z, val.w);
      return dst;
    },
    DEVICE)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    __nv_bfloat16, __nv_bfloat162, { return __bfloat162bfloat162(val); },
    DEVICE)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    __nv_bfloat162, float2, { return __bfloat1622float2(val); }, DEVICE)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    float2, __nv_bfloat162, { return __float22bfloat162_rn(val); }, DEVICE)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    dtype::float4_, dtype::bfloat164,
    {
      dtype::bfloat164 dst;
      dst.x = __float22bfloat162_rn(val.x);
      dst.y = __float22bfloat162_rn(val.y);
      return dst;
    },
    DEVICE)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    dtype::float8_, dtype::bfloat168,
    {
      dtype::bfloat168 dst;
      dst.x = __float22bfloat162_rn(val.x);
      dst.y = __float22bfloat162_rn(val.y);
      dst.z = __float22bfloat162_rn(val.z);
      dst.w = __float22bfloat162_rn(val.w);
      return dst;
    },
    DEVICE)
#else
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    __nv_bfloat16, __nv_bfloat162,
    {
      __nv_bfloat162 dst;
      dst.x = val;
      dst.y = val;
      return dst;
    },
    DEVICE)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    __nv_bfloat162, float2,
    { return make_float2(__low2float(val), __high2float(val)); }, DEVICE)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    float2, __nv_bfloat162, { return __floats2bfloat162_rn(val.x, val.y); },
    DEVICE)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    dtype::float4_, dtype::bfloat164,
    {
      dtype::bfloat164 dst;
      dst.x = __floats2bfloat162_rn(val.x.x, val.x.y);
      dst.y = __floats2bfloat162_rn(val.y.x, val.y.y);
      return dst;
    },
    DEVICE)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    dtype::float8_, dtype::bfloat168,
    {
      dtype::bfloat168 dst;
      dst.x = __floats2bfloat162_rn(val.x.x, val.x.y);
      dst.y = __floats2bfloat162_rn(val.y.x, val.y.y);
      dst.z = __floats2bfloat162_rn(val.z.x, val.z.y);
      dst.w = __floats2bfloat162_rn(val.w.x, val.w.y);
      return dst;
    },
    DEVICE)
#endif /* defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 */
#endif /* defined(COLOSSAL_WITH_CUDA) */

#undef COLOSSAL_CAST_FUNCTOR_SPECIALIZATION
}  // namespace funcs
}  // namespace colossalAI
