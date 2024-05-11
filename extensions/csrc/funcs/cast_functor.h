#pragma once

#if defined(COLOSSAL_WITH_CUDA)
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#endif

#include <assert.h>
#include <stdint.h>

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
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(dtype::half4, float4, DEVICE,
                                     STMTS_WRAPPER({
                                       float4 dst;
                                       dst.x = __half2float(val.x.x);
                                       dst.y = __half2float(val.x.y);
                                       dst.z = __half2float(val.y.x);
                                       dst.w = __half2float(val.y.y);
                                       return dst;
                                     }))
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(dtype::float8, dtype::half8, DEVICE,
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
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(__nv_bfloat16, float, DEVICE,
                                     STMTS_WRAPPER({
                                       return __bfloat162float(val);
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
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(dtype::bfloat164, float4, DEVICE,
                                     STMTS_WRAPPER({
                                       float4 dst;
                                       dst.x = __bfloat162float(val.x.x);
                                       dst.y = __bfloat162float(val.x.y);
                                       dst.z = __bfloat162float(val.y.x);
                                       dst.w = __bfloat162float(val.y.y);
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
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(dtype::float8, dtype::bfloat168, DEVICE,
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
    dtype::float8, dtype::bfloat168, DEVICE, STMTS_WRAPPER({
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

// half raw -> fp8
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(uint16_t, uint8_t, DEVICE, STMTS_WRAPPER({
                                       __half_raw tmp;
                                       tmp.x = val;
                                       __nv_fp8_storage_t res =
                                           __nv_cvt_halfraw_to_fp8(
                                               tmp, __NV_SATFINITE, __NV_E5M2);
                                       return static_cast<uint8_t>(res);
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

// half -> fp8
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(half, uint8_t, DEVICE, STMTS_WRAPPER({
                                       __half_raw tmp(val);
                                       __nv_fp8_storage_t res =
                                           __nv_cvt_halfraw_to_fp8(
                                               tmp, __NV_SATFINITE, __NV_E5M2);
                                       return static_cast<uint8_t>(res);
                                     }))

// fp8x2 -> half2
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(uint16_t, half2, DEVICE, STMTS_WRAPPER({
                                       __half2_raw res =
                                           __nv_cvt_fp8x2_to_halfraw2(
                                               val, __NV_E5M2);
                                       return half2(res);
                                     }))

// half2 -> fp8x2
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(half2, uint16_t, DEVICE, STMTS_WRAPPER({
                                       __half2_raw tmp(val);
                                       __nv_fp8x2_storage_t res =
                                           __nv_cvt_halfraw2_to_fp8x2(
                                               tmp, __NV_SATFINITE, __NV_E5M2);
                                       return static_cast<uint16_t>(res);
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

// half4 -> fp8x4
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    dtype::half4, uint32_t, DEVICE, STMTS_WRAPPER({
      half2 x, y;
      x = val.x;
      y = val.y;
      uint16_t lo, hi;
      lo = CastFunctor<half2, uint16_t>()(x);
      hi = CastFunctor<half2, uint16_t>()(y);
      uint32_t res;
      asm volatile("mov.b32 %0, {%1, %2};\n" : "=r"(res) : "h"(lo), "h"(hi));
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

// float -> fp8
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(float, uint8_t, DEVICE, STMTS_WRAPPER({
                                       __nv_fp8_storage_t res =
                                           __nv_cvt_float_to_fp8(
                                               val, __NV_SATFINITE, __NV_E5M2);
                                       return static_cast<uint8_t>(res);
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

// float2 -> fp8x2
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    float2, uint16_t, DEVICE, STMTS_WRAPPER({
      uint16_t tmp1 =
          static_cast<uint16_t>(CastFunctor<float, uint8_t>()(val.x));
      uint16_t tmp2 =
          static_cast<uint16_t>(CastFunctor<float, uint8_t>()(val.y));
      uint16_t res = (tmp2 << 8U) | tmp1;
      return res;
    }))

// float4 -> fp8x4
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(float4, uint32_t, DEVICE, STMTS_WRAPPER({
                                       uint32_t a, b, c, d;
                                       a = CastFunctor<float, uint8_t>()(val.x);
                                       b = CastFunctor<float, uint8_t>()(val.y);
                                       c = CastFunctor<float, uint8_t>()(val.z);
                                       d = CastFunctor<float, uint8_t>()(val.w);
                                       return (d << 24U) | (c << 16U) |
                                              (b << 8U) | a;
                                     }))

// fp8x4 -> float4
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    uint32_t, float4, DEVICE, STMTS_WRAPPER({
      float4 res;
      res.x = CastFunctor<uint8_t, float>()(static_cast<uint8_t>(val));
      res.y = CastFunctor<uint8_t, float>()(static_cast<uint8_t>(val >> 8U));
      res.z = CastFunctor<uint8_t, float>()(static_cast<uint8_t>(val >> 16U));
      res.w = CastFunctor<uint8_t, float>()(static_cast<uint8_t>(val >> 24U));
      return res;
    }))

// fp8x8 -> float8
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    uint2, dtype::float8, DEVICE, STMTS_WRAPPER({
      dtype::float8 res;
      res.x = CastFunctor<uint16_t, float2>()(static_cast<uint16_t>(val.x));
      res.y =
          CastFunctor<uint16_t, float2>()(static_cast<uint16_t>(val.x >> 16U));
      res.z = CastFunctor<uint16_t, float2>()(static_cast<uint16_t>(val.y));
      res.w =
          CastFunctor<uint16_t, float2>()(static_cast<uint16_t>(val.y >> 16U));
      return res;
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

// bf162 -> fp8x2
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    __nv_bfloat162, uint16_t, DEVICE, STMTS_WRAPPER({
      uint16_t a =
          static_cast<uint16_t>(CastFunctor<__nv_bfloat16, uint8_t>()(val.x));
      uint16_t b =
          static_cast<uint16_t>(CastFunctor<__nv_bfloat16, uint8_t>()(val.y));
      return (b << 8U) | a;
    }))

// bf164 -> fp8x4
COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    dtype::bfloat164, uint32_t, DEVICE, STMTS_WRAPPER({
      uint32_t res;
      uint16_t a, b;
      a = CastFunctor<__nv_bfloat162, uint16_t>()(val.x);
      b = CastFunctor<__nv_bfloat162, uint16_t>()(val.y);
      asm volatile("mov.b32 %0, {%1, %2};\n" : "=r"(res) : "h"(a), "h"(b));
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

COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(float4, uint2, DEVICE, STMTS_WRAPPER({
                                       uint2 b;
                                       float2 c;
                                       c.x = val.x;
                                       c.y = val.y;
                                       b.x = CastFunctor<float2, uint32_t>()(c);

                                       c.x = val.z;
                                       c.y = val.w;
                                       b.y = CastFunctor<float2, uint32_t>()(c);

                                       return b;
                                     }))

COLOSSAL_CAST_FUNCTOR_SPECIALIZATION(
    dtype::float8, uint4, DEVICE, STMTS_WRAPPER({
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
