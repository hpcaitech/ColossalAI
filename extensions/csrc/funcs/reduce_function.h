#pragma once

#if defined(COLOSSAL_WITH_CUDA)
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "binary_functor.h"

namespace colossalAI {
namespace funcs {

const float kReduceFloatInfNeg = -100000000.f;
const float kReduceFloatInfPos = 100000000.f;
const unsigned int kWarpReduceMask = 0xffffffff;

enum class ReduceType { kMax = 0, kSum };

template <typename T, ReduceType rtype>
struct GetOpForReduceType;

template <typename T>
struct GetOpForReduceType<T, ReduceType::kMax> {
  using Op = funcs::BinaryOpFunctor<T, T, T, funcs::BinaryOpType::kMax>;
};

template <typename T>
struct GetOpForReduceType<T, ReduceType::kSum> {
  using Op = funcs::BinaryOpFunctor<T, T, T, funcs::BinaryOpType::kAdd>;
};

#define COLOSSAL_SHFL_FUNCTION(MASK, VAL_PTR, DELTA, WIDTH, OP, LANES) \
  _Pragma("unroll") for (int offset = 0; offset < LANES; ++offset) {   \
    *(VAL_PTR + offset) =                                              \
        OP(*(VAL_PTR + offset),                                        \
           __shfl_xor_sync(MASK, *(VAL_PTR + offset), DELTA, WIDTH));  \
  }

#define COLOSSAL_WARP_REDUCE_IMPL(MASK, VAL_PTR, WIDTH, OP, LANES)           \
  _Pragma("unroll") for (int DELTA = (WIDTH >> 1); DELTA > 0; DELTA >>= 1) { \
    COLOSSAL_SHFL_FUNCTION(MASK, VAL_PTR, DELTA, WIDTH, OP, LANES)           \
  }

#define COLOSSAL_BLOCK_REDUCE_IMPL(DTYPE, VAL_PTR, OP, LANES, DEFAULT_VALUE, \
                                   REDUCE_TYPE)                              \
  __shared__ T shm[LANES][32];                                               \
  int lane_id = threadIdx.x & 0x1f;                                          \
  int warp_id = threadIdx.x >> 5;                                            \
                                                                             \
  warp_reduce<DTYPE, REDUCE_TYPE, LANES>(VAL_PTR);                           \
  if (lane_id == 0) {                                                        \
    for (int offset = 0; offset < LANES; ++offset) {                         \
      shm[offset][warp_id] = *(VAL_PTR + offset);                            \
    }                                                                        \
  }                                                                          \
  __syncthreads();                                                           \
                                                                             \
  _Pragma("unroll") for (int offset = 0; offset < LANES; ++offset) {         \
    *(VAL_PTR + offset) = (threadIdx.x < (blockDim.x >> 5))                  \
                              ? shm[offset][lane_id]                         \
                              : static_cast<T>(DEFAULT_VALUE);               \
  }                                                                          \
  warp_reduce<DTYPE, REDUCE_TYPE, LANES>(VAL_PTR);

template <typename T, ReduceType rtype, int lanes, int width = 32>
__forceinline__ __device__ void warp_reduce(T* pval) {
  typename GetOpForReduceType<T, rtype>::Op op;
  COLOSSAL_WARP_REDUCE_IMPL(kWarpReduceMask, pval, width, op, lanes);
}

template <typename T, ReduceType rtype>
__forceinline__ __device__ constexpr T GetDefaultValueForBlockReduce() {
  if constexpr (rtype == ReduceType::kSum) {
    return static_cast<T>(0.0f);
  } else if constexpr (rtype == ReduceType::kMax) {
    return static_cast<T>(kReduceFloatInfNeg);
  }
}

template <typename T, ReduceType rtype, int lanes>
__forceinline__ __device__ void block_reduce(T* pval) {
  constexpr T kDefaultValue = GetDefaultValueForBlockReduce<T, rtype>();
  typename GetOpForReduceType<T, rtype>::Op op;
  COLOSSAL_BLOCK_REDUCE_IMPL(T, pval, op, lanes, kDefaultValue, rtype);
}

#undef COLOSSAL_SHFL_FUNCTION
#undef COLOSSAL_WARP_REDUCE_IMPL
#undef COLOSSAL_BLOCK_REDUCE_IMPL

}  // namespace funcs
}  // namespace colossalAI

#endif /* defined(COLOSSAL_WITH_CUDA) */
