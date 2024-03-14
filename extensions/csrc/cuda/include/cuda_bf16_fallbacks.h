/*
 * This code from NVIDIA FasterTransformer:
 * https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/utils/cuda_bf16_fallbacks.cuh
 */

inline __device__ __nv_bfloat162 bf16hadd2(const __nv_bfloat162 x,
                                           const __nv_bfloat162 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float fxl, fxh, fyl, fyh;
  fxl = __low2float(x);
  fxh = __high2float(x);
  fyl = __low2float(y);
  fyh = __high2float(y);
  return __floats2bfloat162_rn(fxl + fyl, fxh + fyh);
#else
  return __hadd2(x, y);
#endif
}

inline __device__ __nv_bfloat16 bf16hadd(const __nv_bfloat16 x,
                                         const __nv_bfloat16 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return __float2bfloat16(__bfloat162float(x) + __bfloat162float(y));
#else
  return __hadd(x, y);
#endif
}

inline __device__ __nv_bfloat162 bf16hmul2(const __nv_bfloat162 x,
                                           const __nv_bfloat162 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float fxl, fxh, fyl, fyh;
  fxl = __low2float(x);
  fxh = __high2float(x);
  fyl = __low2float(y);
  fyh = __high2float(y);
  return __floats2bfloat162_rn(fxl * fyl, fxh * fyh);
#else
  return __hmul2(x, y);
#endif
}

inline __device__ __nv_bfloat16 bf16hmul(const __nv_bfloat16 x,
                                         const __nv_bfloat16 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return __float2bfloat16(__bfloat162float(x) * __bfloat162float(y));
#else
  return __hmul(x, y);
#endif
}

inline __device__ __nv_bfloat16 bf16hmul(__nv_bfloat16 a, __nv_bfloat16 b,
                                         __nv_bfloat16 c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b) *
                          __bfloat162float(c));
#else
  return a * b * c;
#endif
}

inline __device__ __nv_bfloat162 bf16hmul2(__nv_bfloat162 a, __nv_bfloat162 b,
                                           __nv_bfloat162 c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float fal, fah, fbl, fbh, fcl, fch;
  fal = __low2float(a);
  fah = __high2float(a);
  fbl = __low2float(b);
  fbh = __high2float(b);
  fcl = __low2float(c);
  fch = __high2float(c);
  return __floats2bfloat162_rn(fal * fbl * fcl, fah * fbh * fch);
#else
  return a * b * c;
#endif
}
