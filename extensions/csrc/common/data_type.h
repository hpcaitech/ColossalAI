#pragma once

#if defined(COLOSSAL_WITH_CUDA)
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif

namespace colossalAI {
namespace dtype {

struct bfloat164 {
#ifdef COLOSSAL_WITH_CUDA
  __nv_bfloat162 x;
  __nv_bfloat162 y;
#endif
};

struct bfloat168 {
#ifdef COLOSSAL_WITH_CUDA
  __nv_bfloat162 x;
  __nv_bfloat162 y;
  __nv_bfloat162 z;
  __nv_bfloat162 w;
#endif
};

struct half4 {
#ifdef COLOSSAL_WITH_CUDA
  half2 x;
  half2 y;
#endif
};

struct half8 {
#ifdef COLOSSAL_WITH_CUDA
  half2 x;
  half2 y;
  half2 z;
  half2 w;
#endif
};

struct float8 {
#ifdef COLOSSAL_WITH_CUDA
  float2 x;
  float2 y;
  float2 z;
  float2 w;
#endif
};

}  // namespace dtype
}  // namespace colossalAI
