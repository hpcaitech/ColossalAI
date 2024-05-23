#pragma once

#if defined(COLOSSAL_WITH_CUDA)
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

#include <functional>

#include "cast_functor.h"
#include "common/data_type.h"
#include "common/micros.h"

namespace colossalAI {
namespace funcs {

enum class BinaryOpType { kAdd = 0, kMinus, kMul, kDiv, kMax, kMin };

// Note(LiuYang): This file provides base math operation for data type
// include POD and cuda built-in type such as half and __nv_bfloat16.
// Implementation of common and simple binary operators should be placed here,
// otherwise, they should be placed in a new file under functors dir.
template <typename LT, typename RT, typename RET, BinaryOpType op_type>
struct BinaryOpFunctor;

#define STMTS_WRAPPER(...) __VA_ARGS__

#define COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(                     \
    LT, RT, RET, BINARY_OP_TYPE, FUNCTION_MODIFIER, STMTS, ARGS...) \
  template <ARGS>                                                   \
  struct BinaryOpFunctor<LT, RT, RET, BINARY_OP_TYPE>               \
      : public std::binary_function<LT, RT, RET> {                  \
    FUNCTION_MODIFIER RET operator()(LT lhs, RT rhs) STMTS          \
  };

COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(T, T, T, BinaryOpType::kAdd, HOSTDEVICE,
                                       STMTS_WRAPPER({ return lhs + rhs; }),
                                       typename T)
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(T, T, T, BinaryOpType::kMinus,
                                       HOSTDEVICE,
                                       STMTS_WRAPPER({ return lhs - rhs; }),
                                       typename T)
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(T, T, T, BinaryOpType::kMul, HOSTDEVICE,
                                       STMTS_WRAPPER({ return lhs * rhs; }),
                                       typename T)
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(T, T, T, BinaryOpType::kDiv, HOSTDEVICE,
                                       STMTS_WRAPPER({ return lhs / rhs; }),
                                       typename T)
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(T, T, T, BinaryOpType::kMax, HOSTDEVICE,
                                       STMTS_WRAPPER({ return max(lhs, rhs); }),
                                       typename T)
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(T, T, T, BinaryOpType::kMin, HOSTDEVICE,
                                       STMTS_WRAPPER({ return min(lhs, rhs); }),
                                       typename T)

#if defined(COLOSSAL_WITH_CUDA)
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(half, half, half, BinaryOpType::kMinus,
                                       DEVICE, STMTS_WRAPPER({
                                         return __hsub(lhs, rhs);
                                       }))

COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(half, half, half, BinaryOpType::kAdd,
                                       DEVICE, STMTS_WRAPPER({
                                         return __hadd(lhs, rhs);
                                       }))
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(half2, half2, half2, BinaryOpType::kAdd,
                                       DEVICE, STMTS_WRAPPER({
                                         return __hadd2(lhs, rhs);
                                       }))

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(__nv_bfloat16, __nv_bfloat16,
                                       __nv_bfloat16, BinaryOpType::kAdd,
                                       DEVICE, STMTS_WRAPPER({
                                         return __hadd(lhs, rhs);
                                       }))

COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(__nv_bfloat16, __nv_bfloat16,
                                       __nv_bfloat16, BinaryOpType::kMinus,
                                       DEVICE, STMTS_WRAPPER({
                                         return __hsub(lhs, rhs);
                                       }))

COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(__nv_bfloat162, __nv_bfloat162,
                                       __nv_bfloat162, BinaryOpType::kAdd,
                                       DEVICE, STMTS_WRAPPER({
                                         return __hadd2(lhs, rhs);
                                       }))
#else
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(
    __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, BinaryOpType::kAdd, DEVICE,
    STMTS_WRAPPER({
      return __float2bfloat16(__bfloat162float(lhs) + __bfloat162float(rhs));
    }))

COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(
    __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, BinaryOpType::kMinus, DEVICE,
    STMTS_WRAPPER({
      return __float2bfloat16(__bfloat162float(lhs) - __bfloat162float(rhs));
    }))

COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(
    __nv_bfloat162, __nv_bfloat162, __nv_bfloat162, BinaryOpType::kAdd, DEVICE,
    STMTS_WRAPPER({
      return __floats2bfloat162_rn(__low2float(lhs) + __low2float(rhs),
                                   __high2float(lhs) + __high2float(rhs));
    }))
#endif /* defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 */

COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(half, half, half, BinaryOpType::kMul,
                                       DEVICE, STMTS_WRAPPER({
                                         return __hmul(lhs, rhs);
                                       }))
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(half2, half2, half2, BinaryOpType::kMul,
                                       DEVICE, STMTS_WRAPPER({
                                         return __hmul2(lhs, rhs);
                                       }))

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(__nv_bfloat16, __nv_bfloat16,
                                       __nv_bfloat16, BinaryOpType::kMul,
                                       DEVICE, STMTS_WRAPPER({
                                         return __hmul(lhs, rhs);
                                       }))
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(__nv_bfloat162, __nv_bfloat162,
                                       __nv_bfloat162, BinaryOpType::kMul,
                                       DEVICE, STMTS_WRAPPER({
                                         return __hmul2(lhs, rhs);
                                       }))
#else
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(
    __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, BinaryOpType::kMul, DEVICE,
    STMTS_WRAPPER({
      return __float2bfloat16(__bfloat162float(lhs) * __bfloat162float(rhs));
    }))
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(
    __nv_bfloat162, __nv_bfloat162, __nv_bfloat162, BinaryOpType::kMul, DEVICE,
    STMTS_WRAPPER({
      return __floats2bfloat162_rn(__low2float(lhs) * __low2float(rhs),
                                   __high2float(lhs) * __high2float(rhs));
    }))
#endif /* defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 */

COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(
    float2, float2, float2, BinaryOpType::kMul, DEVICE,
    STMTS_WRAPPER({ return make_float2(lhs.x * rhs.x, lhs.y * rhs.y); }))
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(float4, float4, float4,
                                       BinaryOpType::kMul, DEVICE,
                                       STMTS_WRAPPER({
                                         return make_float4(
                                             lhs.x * rhs.x, lhs.y * rhs.y,
                                             lhs.z * rhs.z, lhs.w * rhs.w);
                                       }))

COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(
    __nv_bfloat162, __nv_bfloat162, float2, BinaryOpType::kMul, DEVICE,
    STMTS_WRAPPER({
      CastFunctor<__nv_bfloat162, float2> cast;
      BinaryOpFunctor<float2, float2, float2, BinaryOpType::kMul> mul;
      float2 fa = cast(lhs);
      float2 fb = cast(rhs);
      return mul(fa, fb);
    }))

COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(dtype::bfloat164, dtype::bfloat164,
                                       float4, BinaryOpType::kMul, DEVICE,
                                       STMTS_WRAPPER({
                                         float4 fc;
                                         CastFunctor<__nv_bfloat16, float> cast;
                                         fc.x = cast(lhs.x.x) * cast(rhs.x.x);
                                         fc.y = cast(lhs.x.y) * cast(rhs.x.y);
                                         fc.z = cast(lhs.y.x) * cast(rhs.y.x);
                                         fc.w = cast(lhs.y.y) * cast(rhs.y.y);
                                         return fc;
                                       }))

COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(
    dtype::bfloat168, dtype::bfloat168, dtype::float8, BinaryOpType::kMul,
    DEVICE, STMTS_WRAPPER({
      dtype::float8 fc;
      BinaryOpFunctor<__nv_bfloat162, __nv_bfloat162, float2,
                      BinaryOpType::kMul>
          mul;
      fc.x = mul(lhs.x, rhs.x);
      fc.y = mul(lhs.y, rhs.y);
      fc.z = mul(lhs.z, rhs.z);
      fc.w = mul(lhs.w, rhs.w);
      return fc;
    }))

COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(
    half2, half2, float2, BinaryOpType::kMul, DEVICE, STMTS_WRAPPER({
      CastFunctor<half2, float2> cast;
      BinaryOpFunctor<float2, float2, float2, BinaryOpType::kMul> mul;
      float2 fa = cast(lhs);
      float2 fb = cast(rhs);
      return mul(fa, fb);
    }))

COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(dtype::half4, dtype::half4, float4,
                                       BinaryOpType::kMul, DEVICE,
                                       STMTS_WRAPPER({
                                         float4 fc;
                                         CastFunctor<half, float> cast;
                                         fc.x = cast(lhs.x.x) * cast(rhs.x.x);
                                         fc.y = cast(lhs.x.y) * cast(rhs.x.y);
                                         fc.z = cast(lhs.y.x) * cast(rhs.y.x);
                                         fc.w = cast(lhs.y.y) * cast(rhs.y.y);
                                         return fc;
                                       }))

COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(
    dtype::half8, dtype::half8, dtype::float8, BinaryOpType::kMul, DEVICE,
    STMTS_WRAPPER({
      dtype::float8 fc;
      BinaryOpFunctor<half2, half2, float2, BinaryOpType::kMul> mul;
      fc.x = mul(lhs.x, rhs.x);
      fc.y = mul(lhs.y, rhs.y);
      fc.z = mul(lhs.z, rhs.z);
      fc.w = mul(lhs.w, rhs.w);
      return fc;
    }))

#endif /* defined(COLOSSAL_WITH_CUDA) */

#undef COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION
#undef STMTS_WRAPPER
}  // namespace funcs
}  // namespace colossalAI
