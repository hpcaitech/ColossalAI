#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../funcs/op_functor.h"

namespace colossalAI {
namespace cuda {
namespace utils {

const float kReduceFloatInfNeg = -100000000.f;
const float kReduceFloatInfPos = 100000000.f;
const int kWarpSize = 32;
const unsigned int kWarpReduceMask = 0xffffffff;

enum class ReduceType { kMax = 0, kSum };

template <typename T, ReduceType rtype>
struct GetOpForReduceType;

template <typename T>
struct GetOpForReduceType<T, ReduceType::kMax> {
  using Op = funcs::BinaryOpFunctor<T, funcs::BinaryOpType::kMax>;
};

template <typename T>
struct GetOpForReduceType<T, ReduceType::kSum> {
  using Op = funcs::BinaryOpFunctor<T, funcs::BinaryOpType::kAdd>;
};

#define COLOSSAL_SHFL_FUNCTION(MASK, VAL_PTR, DELTA, WIDTH, OP, LANES) \
  for (int offset = 0; offset < LANES; ++offset) {                     \
    *(VAL_PTR + offset) =                                              \
        OP(*(VAL_PTR + offset),                                        \
           __shfl_xor_sync(MASK, *(VAL_PTR + offset), DELTA, WIDTH));  \
  }

#define COLOSSAL_WARP_REDUCE_IMPL(MASK, VAL_PTR, OP, LANES) \
  COLOSSAL_SHFL_FUNCTION(MASK, VAL_PTR, 16, 32, OP, LANES)  \
  COLOSSAL_SHFL_FUNCTION(MASK, VAL_PTR, 8, 32, OP, LANES)   \
  COLOSSAL_SHFL_FUNCTION(MASK, VAL_PTR, 4, 32, OP, LANES)   \
  COLOSSAL_SHFL_FUNCTION(MASK, VAL_PTR, 2, 32, OP, LANES)   \
  COLOSSAL_SHFL_FUNCTION(MASK, VAL_PTR, 1, 32, OP, LANES)

#define COLOSSAL_BLOCK_REDUCE_IMPL(DTYPE, MASK, VAL_PTR, OP, LANES, \
                                   DEFAULT_VALUE, REDUCE_TYPE)      \
  __shared__ T shm[LANES][32];                                      \
  int lane_id = threadIdx.x & 0x1f;                                 \
  int warp_id = threadIdx.x >> 5;                                   \
                                                                    \
  warp_reduce<DTYPE, REDUCE_TYPE, LANES>(VAL_PTR);                  \
  if (lane_id == 0) {                                               \
    for (int offset = 0; offset < LANES; ++offset) {                \
      shm[offset][warp_id] = *(VAL_PTR + offset);                   \
    }                                                               \
  }                                                                 \
  __syncthreads();                                                  \
                                                                    \
  for (int offset = 0; offset < LANES; ++offset) {                  \
    *(VAL_PTR + offset) = (threadIdx.x < (blockDim.x >> 5))         \
                              ? shm[offset][lane_id]                \
                              : static_cast<T>(DEFAULT_VALUE);      \
  }                                                                 \
  warp_reduce<DTYPE, REDUCE_TYPE, LANES>(VAL_PTR);

template <typename T, ReduceType rtype, int lanes>
__forceinline__ __device__ void warp_reduce(T* pval) {
  typename GetOpForReduceType<T, rtype>::Op op;
  COLOSSAL_WARP_REDUCE_IMPL(kWarpReduceMask, pval, op, lanes);
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
  COLOSSAL_BLOCK_REDUCE_IMPL(T, kWarpReduceMask, pval, op, lanes, kDefaultValue,
                             rtype);
}

#undef COLOSSAL_SHFL_FUNCTION
#undef COLOSSAL_WARP_REDUCE_IMPL
#undef COLOSSAL_BLOCK_REDUCE_IMPL

template <typename T>
__device__ __forceinline__ T reduce_block_into_lanes(
    T* x, T val, int lanes = 1,
    bool share_result = false)  // lanes is intended to be <= 32.
{
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int blockSize =
      blockDim.x * blockDim.y;  // blockSize is intended to be a multiple of 32.

  if (blockSize >= 64) {
    x[tid] = val;
    __syncthreads();
  }

#pragma unroll
  for (int i = (blockSize >> 1); i >= 64; i >>= 1) {
    if (tid < i) x[tid] = x[tid] + x[tid + i];
    __syncthreads();
  }

  T final;

  if (tid < 32) {
    if (blockSize >= 64)
      final = x[tid] + x[tid + 32];
    else
      final = val;
      // __SYNCWARP();

#pragma unroll
    for (int i = 16; i >= lanes; i >>= 1)
      final = final + __shfl_down_sync(0xffffffff, final, i);
  }

  if (share_result) {
    if (tid < lanes) x[tid] = final;  // EpilogueOp
    // Make sure the smem result is visible to all warps.
    __syncthreads();
  }

  return final;
}

template <typename T>
__device__ __forceinline__ T reduce_block_into_lanes_max_op(
    T* x, T val, int lanes = 1,
    bool share_result = false)  // lanes is intended to be <= 32.
{
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int blockSize =
      blockDim.x * blockDim.y;  // blockSize is intended to be a multiple of 32.

  if (blockSize >= 64) {
    x[tid] = val;
    __syncthreads();
  }

#pragma unroll
  for (int i = (blockSize >> 1); i >= 64; i >>= 1) {
    if (tid < i) x[tid] = fmaxf(fabsf(x[tid]), fabsf(x[tid + i]));
    __syncthreads();
  }

  T final;

  if (tid < 32) {
    if (blockSize >= 64)
      final = fmaxf(fabsf(x[tid]), fabsf(x[tid + 32]));
    else
      final = val;
      // __SYNCWARP();

#pragma unroll
    for (int i = 16; i >= lanes; i >>= 1)
      final =
          fmaxf(fabsf(final), fabsf(__shfl_down_sync(0xffffffff, final, i)));
  }

  if (share_result) {
    if (tid < lanes) x[tid] = final;  // EpilogueOp
    // Make sure the smem result is visible to all warps.
    __syncthreads();
  }

  return final;
}

}  // namespace utils
}  // namespace cuda
}  // namespace colossalAI
