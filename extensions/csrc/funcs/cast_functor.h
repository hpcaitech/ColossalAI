#pragma once

#if defined(COLOSSAL_WITH_CUDA)
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#endif

#include <assert.h>

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

#define STMTS_WRAPPER(...) __VA_ARGS__

#define COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(FROM, TO, FUNCTION_MODIFIER, \
                                             STMTS)                       \
  template <>                                                             \
  struct CastFunctor<FROM, TO> : public std::unary_function<FROM, TO> {   \
    FUNCTION_MODIFIER TO operator()(FROM val) STMTS                       \
  };

#if defined(COLOSSAL_WITH_CUDA)
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(int2, float2, DEVICE, STMTS_WRAPPER({
                                       return make_float2(val.x, val.y);
                                     }))
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(float, float2, DEVICE, STMTS_WRAPPER({
                                       return make_float2(val, val);
                                     }))

COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(half2, float2, DEVICE, STMTS_WRAPPER({
                                       return __half22float2(val);
                                     }))
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(float2, half2, DEVICE, STMTS_WRAPPER({
                                       return __float22half2_rn(val);
                                     }))
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(float, half, DEVICE, STMTS_WRAPPER({
                                       return __float2half_rn(val);
                                     }))
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(float, half2, DEVICE, STMTS_WRAPPER({
                                       return __float2half2_rn(val);
                                     }))
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(half, half2, DEVICE, STMTS_WRAPPER({
                                       return __half2half2(val);
                                     }))
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(half, float, DEVICE, STMTS_WRAPPER({
                                       return __half2float(val);
                                     }))
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(float4, dtype::half4, DEVICE,
                                     STMTS_WRAPPER({
                                       dtype::half4 dst;
                                       dst.x = __floats2half2_rn(val.x, val.y);
                                       dst.y = __floats2half2_rn(val.z, val.w);
                                       return dst;
                                     }))
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(dtype::float4_, dtype::half4, DEVICE,
                                     STMTS_WRAPPER({
                                       dtype::half4 dst;
                                       dst.x = __float22half2_rn(val.x);
                                       dst.y = __float22half2_rn(val.y);
                                       return dst;
                                     }))
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(dtype::float8_, dtype::half8, DEVICE,
                                     STMTS_WRAPPER({
                                       dtype::half8 dst;
                                       dst.x = __float22half2_rn(val.x);
                                       dst.y = __float22half2_rn(val.y);
                                       dst.z = __float22half2_rn(val.z);
                                       dst.w = __float22half2_rn(val.w);
                                       return dst;
                                     }))

COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(float, __nv_bfloat162, DEVICE,
                                     STMTS_WRAPPER({
                                       return __float2bfloat162_rn(val);
                                     }))
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(float, __nv_bfloat16, DEVICE,
                                     STMTS_WRAPPER({
                                       return __float2bfloat16_rn(val);
                                     }))
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(float4, dtype::bfloat164, DEVICE,
                                     STMTS_WRAPPER({
                                       dtype::bfloat164 dst;
                                       dst.x =
                                           __floats2bfloat162_rn(val.x, val.y);
                                       dst.y =
                                           __floats2bfloat162_rn(val.z, val.w);
                                       return dst;
                                     }))
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(__nv_bfloat16, __nv_bfloat162, DEVICE,
                                     STMTS_WRAPPER({
                                       return __bfloat162bfloat162(val);
                                     }))
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(__nv_bfloat162, float2, DEVICE,
                                     STMTS_WRAPPER({
                                       return __bfloat1622float2(val);
                                     }))
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(float2, __nv_bfloat162, DEVICE,
                                     STMTS_WRAPPER({
                                       return __float22bfloat162_rn(val);
                                     }))
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(dtype::float4_, dtype::bfloat164, DEVICE,
                                     STMTS_WRAPPER({
                                       dtype::bfloat164 dst;
                                       dst.x = __float22bfloat162_rn(val.x);
                                       dst.y = __float22bfloat162_rn(val.y);
                                       return dst;
                                     }))
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(dtype::float8_, dtype::bfloat168, DEVICE,
                                     STMTS_WRAPPER({
                                       dtype::bfloat168 dst;
                                       dst.x = __float22bfloat162_rn(val.x);
                                       dst.y = __float22bfloat162_rn(val.y);
                                       dst.z = __float22bfloat162_rn(val.z);
                                       dst.w = __float22bfloat162_rn(val.w);
                                       return dst;
                                     }))
#else
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(__nv_bfloat16, __nv_bfloat162, DEVICE,
                                     STMTS_WRAPPER({
                                       __nv_bfloat162 dst;
                                       dst.x = val;
                                       dst.y = val;
                                       return dst;
                                     }))
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(__nv_bfloat162, float2, DEVICE,
                                     STMTS_WRAPPER({
                                       return make_float2(__low2float(val),
                                                          __high2float(val));
                                     }))
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(float2, __nv_bfloat162, DEVICE,
                                     STMTS_WRAPPER({
                                       return __floats2bfloat162_rn(val.x,
                                                                    val.y);
                                     }))
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    dtype::float4_, dtype::bfloat164, DEVICE, STMTS_WRAPPER({
      dtype::bfloat164 dst;
      dst.x = __floats2bfloat162_rn(val.x.x, val.x.y);
      dst.y = __floats2bfloat162_rn(val.y.x, val.y.y);
      return dst;
    }))
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    dtype::float8_, dtype::bfloat168, DEVICE, STMTS_WRAPPER({
      dtype::bfloat168 dst;
      dst.x = __floats2bfloat162_rn(val.x.x, val.x.y);
      dst.y = __floats2bfloat162_rn(val.y.x, val.y.y);
      dst.z = __floats2bfloat162_rn(val.z.x, val.z.y);
      dst.w = __floats2bfloat162_rn(val.w.x, val.w.y);
      return dst;
    }))
#endif /* defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 */

// quant utils
// fp8 -> half raw
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(uint8_t, uint16_t, DEVICE, STMTS_WRAPPER({
                                       __half_raw res = __nv_cvt_fp8_to_halfraw(
                                           val, __NV_E5M2);
                                       return res.x;
                                     }))

// fp8x2 -> half2 raw
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(uint16_t, uint32_t, DEVICE, STMTS_WRAPPER({
                                       union {
                                         uint16_t u16[2];
                                         uint32_t u32;
                                       } tmp;
                                       __half2_raw res =
                                           __nv_cvt_fp8x2_to_halfraw2(
                                               val, __NV_E5M2);
                                       tmp.u16[0] = res.x;
                                       tmp.u16[1] = res.y;
                                       return tmp.u32;
                                     }))

// fp8x4 -> half2x2 raw
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    uint32_t, uint2, DEVICE, STMTS_WRAPPER({
      union {
        uint2 u32x2;
        uint32_t u32[2];
      } tmp;
      tmp.u32[0] =
          CastFunctor<uint16_t, uint32_t>()(static_cast<uint16_t>(val));
      tmp.u32[1] =
          CastFunctor<uint16_t, uint32_t>()(static_cast<uint16_t>(val >> 16U));
      return tmp.u32x2;
    }))

// fp8x8 -> half2x4 raw
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    uint2, uint4, DEVICE, STMTS_WRAPPER({
      union {
        uint4 u64x2;
        uint2 u64[2];
      } tmp;
      tmp.u64[0] = CastFunctor<uint32_t, uint2>()(val.x);
      tmp.u64[1] = CastFunctor<uint32_t, uint2>()(val.y);
      return tmp.u64x2;
    }))

// fp8 -> half
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(uint8_t, half, DEVICE, STMTS_WRAPPER({
                                       __half_raw res = __nv_cvt_fp8_to_halfraw(
                                           val, __NV_E5M2);
                                       return half(res);
                                     }))

// fp8x2 -> half2
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(uint16_t, half2, DEVICE, STMTS_WRAPPER({
                                       __half2_raw res =
                                           __nv_cvt_fp8x2_to_halfraw2(
                                               val, __NV_E5M2);
                                       return half2(res);
                                     }))

// fp8x4 -> half4
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    uint32_t, dtype::half4, DEVICE, STMTS_WRAPPER({
      half2 tmp1, tmp2;
      tmp1 = CastFunctor<uint16_t, half2>()(static_cast<uint16_t>(val));
      tmp2 = CastFunctor<uint16_t, half2>()(static_cast<uint16_t>(val >> 16U));
      dtype::half4 res;
      res.x = tmp1;
      res.y = tmp2;
      return res;
    }))

// fp8x8 -> half8
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    uint2, dtype::half8, DEVICE, STMTS_WRAPPER({
      dtype::half4 tmp1, tmp2;
      tmp1 = CastFunctor<uint32_t, dtype::half4>()(val.x);
      tmp2 = CastFunctor<uint32_t, dtype::half4>()(val.y);
      dtype::half8 res;
      res.x = tmp1.x;
      res.y = tmp1.y;
      res.z = tmp2.x;
      res.w = tmp2.y;
      return res;
    }))

// fp8 -> __nv_bfloat16
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    uint8_t, __nv_bfloat16, DEVICE, STMTS_WRAPPER({
      // Note there is no direct convert function from fp8 to bf16.
      // fp8 -> half
      __half_raw res = __nv_cvt_fp8_to_halfraw(val, __NV_E5M2);
      // half -> float -> bf16
      float tmp;
      asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(tmp) : "h"(res.x));
      return __float2bfloat16(tmp);
    }))

// fp8x2 -> __nv_bfloat162
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    uint16_t, __nv_bfloat162, DEVICE, STMTS_WRAPPER({
      __nv_bfloat162 res;
      res.x = CastFunctor<uint8_t, __nv_bfloat16>()(static_cast<uint8_t>(val));
      res.y = CastFunctor<uint8_t, __nv_bfloat16>()(
          static_cast<uint8_t>(val >> 8U));
      return res;
    }))

// fp8x4 -> bfloat164
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    uint32_t, dtype::bfloat164, DEVICE, STMTS_WRAPPER({
      dtype::bfloat164 res;
      res.x =
          CastFunctor<uint16_t, __nv_bfloat162>()(static_cast<uint16_t>(val));
      res.y = CastFunctor<uint16_t, __nv_bfloat162>()(
          static_cast<uint16_t>(val >> 16U));
      return res;
    }))

// fp8x8 -> bfloat168
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    uint2, dtype::bfloat168, DEVICE, STMTS_WRAPPER({
      dtype::bfloat164 tmp1, tmp2;
      tmp1 = CastFunctor<uint32_t, dtype::bfloat164>()(val.x);
      tmp2 = CastFunctor<uint32_t, dtype::bfloat164>()(val.y);
      dtype::bfloat168 res;
      res.x = tmp1.x;
      res.y = tmp1.y;
      res.z = tmp2.x;
      res.w = tmp2.y;
      return res;
    }))

// fp8 -> float
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    uint8_t, float, DEVICE, STMTS_WRAPPER({
      // fp8 -> half
      uint16_t tmp = CastFunctor<uint8_t, uint16_t>()(val);
      // half -> float
      float res;
      asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(res) : "h"(tmp));
      return res;
    }))

// fp8x2 -> float2
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    uint16_t, float2, DEVICE, STMTS_WRAPPER({
      // fp8x2 -> half2
      uint32_t tmp = CastFunctor<uint16_t, uint32_t>()(val);
      // half2 -> float2
      uint16_t lo, hi;
      asm volatile("mov.b32 {%0, %1}, %2;\n" : "=h"(lo), "=h"(hi) : "r"(tmp));
      float lof, hif;
      asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(lof) : "h"(lo));
      asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(hif) : "h"(hi));
      return make_float2(lof, hif);
    }))

// fp8x4 -> float4_
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    uint32_t, dtype::float4_, DEVICE, STMTS_WRAPPER({
      dtype::float4_ res;
      res.x = CastFunctor<uint16_t, float2>()(static_cast<uint16_t>(val));
      res.y =
          CastFunctor<uint16_t, float2>()(static_cast<uint16_t>(val >> 16U));
      return res;
    }))

// fp8x8 -> float8_
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    uint2, dtype::float8_, DEVICE, STMTS_WRAPPER({
      dtype::float4_ tmp1, tmp2;
      tmp1 = CastFunctor<uint32_t, dtype::float4_>()(val.x);
      tmp2 = CastFunctor<uint32_t, dtype::float4_>()(val.y);
      dtype::float8_ res;
      res.x = tmp1.x;
      res.y = tmp1.y;
      res.z = tmp2.x;
      res.w = tmp2.y;
      return res;
    }))

// half -> fp8
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(uint16_t, uint8_t, DEVICE, STMTS_WRAPPER({
                                       __half_raw tmp;
                                       tmp.x = val;
                                       __nv_fp8_storage_t res =
                                           __nv_cvt_halfraw_to_fp8(
                                               tmp, __NV_SATFINITE, __NV_E5M2);
                                       return static_cast<uint8_t>(res);
                                     }))

// bf16 -> fp8
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(__nv_bfloat16, uint8_t, DEVICE,
                                     STMTS_WRAPPER({
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
                                       assert(false);
#else
                                       __nv_fp8_storage_t res =
                                           __nv_cvt_bfloat16raw_to_fp8(
                                               __nv_bfloat16_raw(val),
                                               __NV_SATFINITE, __NV_E5M2);
                                       return static_cast<uint8_t>(res);
#endif
                                     }))

// float -> fp8
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(float, uint8_t, DEVICE, STMTS_WRAPPER({
                                       __nv_fp8_storage_t res =
                                           __nv_cvt_float_to_fp8(
                                               val, __NV_SATFINITE, __NV_E5M2);
                                       return static_cast<uint8_t>(res);
                                     }))

// fp8x4 -> float4
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    uint32_t, float4, DEVICE, STMTS_WRAPPER({
      dtype::float4_ tmp = CastFunctor<uint32_t, dtype::float4_>()(val);
      float4 res = make_float4(tmp.x.x, tmp.x.y, tmp.y.x, tmp.y.y);
      return res;
    }))

COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(float2, uint32_t, DEVICE, STMTS_WRAPPER({
                                       union {
                                         half2 float16;
                                         uint32_t uint32;
                                       };

                                       float16 = __float22half2_rn(val);
                                       return uint32;
                                     }))

COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(dtype::float4_, uint2, DEVICE,
                                     STMTS_WRAPPER({
                                       uint2 b;
                                       float2 c;
                                       c.x = val.x.x;
                                       c.y = val.x.y;
                                       b.x = CastFunctor<float2, uint32_t>()(c);

                                       c.x = val.y.x;
                                       c.y = val.y.y;
                                       b.y = CastFunctor<float2, uint32_t>()(c);

                                       return b;
                                     }))

// float4_ -> float4
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(dtype::float4_, float4, DEVICE,
                                     STMTS_WRAPPER({
                                       float4 b;
                                       b.x = val.x.x;
                                       b.y = val.x.y;
                                       b.z = val.y.x;
                                       b.w = val.y.y;
                                       return b;
                                     }))

COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    dtype::float8_, uint4, DEVICE, STMTS_WRAPPER({
      uint4 b;
      b.x = CastFunctor<float2, uint32_t>()(val.x);
      b.y = CastFunctor<float2, uint32_t>()(val.y);
      b.z = CastFunctor<float2, uint32_t>()(val.z);
      b.w = CastFunctor<float2, uint32_t>()(val.w);
      return b;
    }))

#endif /* defined(COLOSSAL_WITH_CUDA) */

#undef STMTS_WRAPPER
#undef COLOSSAL_CAST_FUNCTOR_SPECIALIZATION
}  // namespace funcs
}  // namespace colossalAI
