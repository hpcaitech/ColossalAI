#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <functional>

#include "../utils/micros.h"

// Note(LiuYang): This file provides base math operation for data type
// include POD and cuda built-in type such as half and __nv_bfloat16

namespace colossalAI {
namespace cuda {
namespace funcs {

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

template <>
struct TypeConverter<__nv_bfloat162> {
  using Type = at::BFloat16;
};

template <>
struct TypeConverter<at::BFloat16> {
  using Type = __nv_bfloat162;
};

template <typename From, typename To>
struct CastFunctor : public std::unary_function<From, To> {
  HOSTDEVICE To operator()(From val) { return static_cast<To>(val); }
};

#define COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(FROM, TO, STMT,            \
                                             FUNCTION_MODIFIER)         \
  template <>                                                           \
  struct CastFunctor<FROM, TO> : public std::unary_function<FROM, TO> { \
    FUNCTION_MODIFIER TO operator()(FROM val) { return STMT; }          \
  };

COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(int2, float2, make_float2(val.x, val.y),
                                     DEVICE)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(float, float2, make_float2(val, val),
                                     DEVICE)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(half2, float2, __half22float2(val), DEVICE)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(float2, half2, __float22half2_rn(val),
                                     DEVICE)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(float, half2, __float2half2_rn(val),
                                     DEVICE)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(half, half2, __half2half2(val), DEVICE)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(half, float, __half2float(val), DEVICE)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(float, nv_bfloat162,
                                     __float2bfloat162_rn(val), DEVICE)

#undef COLOSSAL_CAST_FUNCTOR_SPECIALIZATION
}  // namespace funcs
}  // namespace cuda
}  // namespace colossalAI
