#pragma once

#if defined(COLOSSAL_WITH_CUDA)
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

#include <float.h>

#include <functional>

#include "cast_functor.h"
#include "common/micros.h"

namespace colossalAI {
namespace funcs {

enum class TernaryOpType { kFma = 0 };

template <typename LT, typename RT, typename RET, TernaryOpType op_type>
struct TernaryOpFunctor;

#define STMTS_WRAPPER(...) __VA_ARGS__

#define COLOSSAL_TERNARY_FUNCTOR_SPECIALIZATION(                     \
    LT, RT, RET, TERNARY_OP_TYPE, FUNCTION_MODIFIER, STMTS, ARGS...) \
  template <ARGS>                                                    \
  struct TernaryOpFunctor<LT, RT, RET, TERNARY_OP_TYPE> {            \
    FUNCTION_MODIFIER RET operator()(LT a, RT b, RET c) STMTS        \
  };

#if defined(COLOSSAL_WITH_CUDA)
COLOSSAL_TERNARY_FUNCTOR_SPECIALIZATION(float, float, float,
                                        TernaryOpType::kFma, DEVICE,
                                        STMTS_WRAPPER({
                                          float d;
                                          d = fma(a, b, c);
                                          return d;
                                        }))
COLOSSAL_TERNARY_FUNCTOR_SPECIALIZATION(float2, float2, float2,
                                        TernaryOpType::kFma, DEVICE,
                                        STMTS_WRAPPER({
                                          float2 d;
                                          d.x = fma(a.x, b.x, c.x);
                                          d.y = fma(a.y, b.y, c.y);
                                          return d;
                                        }))
COLOSSAL_TERNARY_FUNCTOR_SPECIALIZATION(float, float2, float2,
                                        TernaryOpType::kFma, DEVICE,
                                        STMTS_WRAPPER({
                                          float2 d;
                                          d.x = fma(a, b.x, c.x);
                                          d.y = fma(a, b.y, c.y);
                                          return d;
                                        }))
COLOSSAL_TERNARY_FUNCTOR_SPECIALIZATION(float4, float4, float4,
                                        TernaryOpType::kFma, DEVICE,
                                        STMTS_WRAPPER({
                                          float4 d;
                                          d.x = fma(a.x, b.x, c.x);
                                          d.y = fma(a.y, b.y, c.y);
                                          d.z = fma(a.z, b.z, c.z);
                                          d.w = fma(a.w, b.w, c.w);
                                          return d;
                                        }))
COLOSSAL_TERNARY_FUNCTOR_SPECIALIZATION(float, float4, float4,
                                        TernaryOpType::kFma, DEVICE,
                                        STMTS_WRAPPER({
                                          float4 d;
                                          d.x = fma(a, b.x, c.x);
                                          d.y = fma(a, b.y, c.y);
                                          d.z = fma(a, b.z, c.z);
                                          d.w = fma(a, b.w, c.w);
                                          return d;
                                        }))

COLOSSAL_TERNARY_FUNCTOR_SPECIALIZATION(
    half, half, float, TernaryOpType::kFma, DEVICE,
    STMTS_WRAPPER({ return __half2float(a) * __half2float(b) + c; }))
COLOSSAL_TERNARY_FUNCTOR_SPECIALIZATION(
    half2, half2, float2, TernaryOpType::kFma, DEVICE, STMTS_WRAPPER({
      CastFunctor<half2, float2> cast;
      TernaryOpFunctor<float2, float2, float2, TernaryOpType::kFma> fma;
      float2 fa = cast(a);
      float2 fb = cast(b);
      return fma(fa, fb, c);
    }))
COLOSSAL_TERNARY_FUNCTOR_SPECIALIZATION(
    half, half2, float2, TernaryOpType::kFma, DEVICE, STMTS_WRAPPER({
      CastFunctor<half, half2> cast;
      TernaryOpFunctor<half2, half2, float2, TernaryOpType::kFma> fma;
      return fma(cast(a), b, c);
    }))
COLOSSAL_TERNARY_FUNCTOR_SPECIALIZATION(
    dtype::half4, dtype::half4, float4, TernaryOpType::kFma, DEVICE,
    STMTS_WRAPPER({
      float4 fd;
      CastFunctor<dtype::half4, float4> cast;
      TernaryOpFunctor<float4, float4, float4, TernaryOpType::kFma> fma;
      fd = fma(cast(a), cast(b), c);
      return fd;
    }))
COLOSSAL_TERNARY_FUNCTOR_SPECIALIZATION(
    half, dtype::half4, float4, TernaryOpType::kFma, DEVICE, STMTS_WRAPPER({
      float4 fd;
      CastFunctor<half, float> cast0;
      CastFunctor<dtype::half4, float4> cast1;
      TernaryOpFunctor<float, float4, float4, TernaryOpType::kFma> fma;
      fd = fma(cast0(a), cast1(b), c);
      return fd;
    }))
COLOSSAL_TERNARY_FUNCTOR_SPECIALIZATION(
    dtype::half8, dtype::half8, dtype::float8, TernaryOpType::kFma, DEVICE,
    STMTS_WRAPPER({
      dtype::float8 fd;
      TernaryOpFunctor<half2, half2, float2, TernaryOpType::kFma> fma;
      fd.x = fma(a.x, b.x, c.x);
      fd.y = fma(a.y, b.y, c.y);
      fd.z = fma(a.z, b.z, c.z);
      fd.w = fma(a.w, b.w, c.w);
      return fd;
    }))
COLOSSAL_TERNARY_FUNCTOR_SPECIALIZATION(
    half, dtype::half8, dtype::float8, TernaryOpType::kFma, DEVICE,
    STMTS_WRAPPER({
      dtype::float8 fd;
      CastFunctor<half, half2> cast;
      TernaryOpFunctor<half2, half2, float2, TernaryOpType::kFma> fma;
      half2 s = cast(a);
      fd.x = fma(s, b.x, c.x);
      fd.y = fma(s, b.y, c.y);
      fd.z = fma(s, b.z, c.z);
      fd.w = fma(s, b.w, c.w);
      return fd;
    }))

COLOSSAL_TERNARY_FUNCTOR_SPECIALIZATION(
    __nv_bfloat16, __nv_bfloat16, float, TernaryOpType::kFma, DEVICE,
    STMTS_WRAPPER({ return __bfloat162float(a) * __bfloat162float(b) + c; }))
COLOSSAL_TERNARY_FUNCTOR_SPECIALIZATION(
    __nv_bfloat162, __nv_bfloat162, float2, TernaryOpType::kFma, DEVICE,
    STMTS_WRAPPER({
      CastFunctor<__nv_bfloat162, float2> cast;
      TernaryOpFunctor<float2, float2, float2, TernaryOpType::kFma> fma;
      float2 fa = cast(a);
      float2 fb = cast(b);
      return fma(fa, fb, c);
    }))
COLOSSAL_TERNARY_FUNCTOR_SPECIALIZATION(
    __nv_bfloat16, __nv_bfloat162, float2, TernaryOpType::kFma, DEVICE,
    STMTS_WRAPPER({
      CastFunctor<__nv_bfloat16, __nv_bfloat162> cast;
      TernaryOpFunctor<__nv_bfloat162, __nv_bfloat162, float2,
                       TernaryOpType::kFma>
          fma;
      return fma(cast(a), b, c);
    }))
COLOSSAL_TERNARY_FUNCTOR_SPECIALIZATION(
    dtype::bfloat164, dtype::bfloat164, float4, TernaryOpType::kFma, DEVICE,
    STMTS_WRAPPER({
      float4 fd;
      CastFunctor<dtype::bfloat164, float4> cast;
      TernaryOpFunctor<float4, float4, float4, TernaryOpType::kFma> fma;
      fd = fma(cast(a), cast(b), c);
      return fd;
    }))
COLOSSAL_TERNARY_FUNCTOR_SPECIALIZATION(
    __nv_bfloat16, dtype::bfloat164, float4, TernaryOpType::kFma, DEVICE,
    STMTS_WRAPPER({
      float4 fd;
      CastFunctor<__nv_bfloat16, float> cast0;
      CastFunctor<dtype::bfloat164, float4> cast1;
      TernaryOpFunctor<float, float4, float4, TernaryOpType::kFma> fma;
      fd = fma(cast0(a), cast1(b), c);
      return fd;
    }))
COLOSSAL_TERNARY_FUNCTOR_SPECIALIZATION(
    dtype::bfloat168, dtype::bfloat168, dtype::float8, TernaryOpType::kFma,
    DEVICE, STMTS_WRAPPER({
      dtype::float8 fd;
      TernaryOpFunctor<__nv_bfloat162, __nv_bfloat162, float2,
                       TernaryOpType::kFma>
          fma;
      fd.x = fma(a.x, b.x, c.x);
      fd.y = fma(a.y, b.y, c.y);
      fd.z = fma(a.z, b.z, c.z);
      fd.w = fma(a.w, b.w, c.w);
      return fd;
    }))
COLOSSAL_TERNARY_FUNCTOR_SPECIALIZATION(
    __nv_bfloat16, dtype::bfloat168, dtype::float8, TernaryOpType::kFma, DEVICE,
    STMTS_WRAPPER({
      dtype::float8 fd;
      CastFunctor<__nv_bfloat16, __nv_bfloat162> cast;
      TernaryOpFunctor<__nv_bfloat162, __nv_bfloat162, float2,
                       TernaryOpType::kFma>
          fma;
      __nv_bfloat162 s = cast(a);
      fd.x = fma(s, b.x, c.x);
      fd.y = fma(s, b.y, c.y);
      fd.z = fma(s, b.z, c.z);
      fd.w = fma(s, b.w, c.w);
      return fd;
    }))

#endif /* defined(COLOSSAL_WITH_CUDA) */

#undef COLOSSAL_TERNARY_FUNCTOR_SPECIALIZATION
#undef STMTS_WRAPPER

}  // namespace funcs
}  // namespace colossalAI
