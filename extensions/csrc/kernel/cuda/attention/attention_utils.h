/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * Copyright (c) 2024, The Colossal-AI team.
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <float.h>

#include "common/vec_type_traits.h"
#include "funcs/binary_functor.h"
#include "funcs/cast_functor.h"
#include "funcs/ternary_functor.h"
#include "funcs/unary_functor.h"

namespace colossalAI {
namespace cuda {
namespace attention {

#define WARP_SIZE 32
#define VEC_SIZE_8 8

#define SHFL_XOR_SYNC(var, lane_mask) \
  __shfl_xor_sync(uint32_t(-1), var, lane_mask)
#define SHFL_SYNC(var, src_lane) __shfl_sync(uint32_t(-1), var, src_lane)

// Q*K^T operation.
template <int NUM_THREADS_PER_ROUNDS, int NUM_THREADS_PER_X, typename VecT,
          int N>
inline __device__ float qk_dot_(const VecT (&q)[N], const VecT (&k)[N]) {
  using A_vec = typename common::FloatVecTypeTrait<VecT>::Type;
  // Compute the parallel products for Q*K^T (treat vector lanes separately).
  funcs::BinaryOpFunctor<VecT, VecT, A_vec, funcs::BinaryOpType::kMul> mul_vect;
  funcs::UnaryOpFunctor<A_vec, float, funcs::UnaryOpType::kSum> sum_vect;
  funcs::TernaryOpFunctor<VecT, VecT, A_vec, funcs::TernaryOpType::kFma> fma;

  A_vec qk_vec = mul_vect(q[0], k[0]);
#pragma unroll
  for (int ii = 1; ii < N; ii++) {
    qk_vec = fma(q[ii], k[ii], qk_vec);
  }

  // Finalize the reduction across lanes.
  float qk = sum_vect(qk_vec);
#pragma unroll
  for (int mask = (WARP_SIZE >> 1); mask >= NUM_THREADS_PER_ROUNDS;
       mask >>= 1) {
    qk += SHFL_XOR_SYNC(qk, mask);
  }

#pragma unroll
  for (int mask = (NUM_THREADS_PER_X >> 1); mask > 0; mask >>= 1) {
    qk += SHFL_XOR_SYNC(qk, mask);
  }
  return qk;
}

template <typename T, int NUM_THREADS_PER_ROUNDS, int NUM_THREADS_PER_X>
struct Qk_dot {
  template <typename VecT, int N>
  static inline __device__ float dot(const VecT (&q)[N], const VecT (&k)[N]) {
    return qk_dot_<NUM_THREADS_PER_ROUNDS, NUM_THREADS_PER_X>(q, k);
  }
};

template <int NUM_WARPS, int NUM_THREADS_PER_ROUNDS, int NUM_THREADS_PER_X>
inline __device__ float block_max(float* red_smem, float max) {
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 0x1f;

// Perform reduction across the threads in the same warp to get the max value
// for each warp, the 1st out of NUM_THREADS_PER_TOKEN thread already has the
// max value among every NUM_THREADS_PER_TOKEN threads.
#pragma unroll
  for (int mask = (NUM_THREADS_PER_ROUNDS >> 1); mask >= NUM_THREADS_PER_X;
       mask >>= 1) {
    max = fmaxf(max, SHFL_XOR_SYNC(max, mask));
  }

  if (lane == 0) red_smem[warp] = max;
  __syncthreads();

  // The warps compute the final maxs.
  max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;

// Parallel reduction of all tokens from the same sequence inside the warp.
#pragma unroll
  for (int mask = (NUM_WARPS >> 1); mask > 0; mask >>= 1) {
    max = fmaxf(max, SHFL_XOR_SYNC(max, mask));
  }

  // Broadcast to other threads.
  return SHFL_SYNC(max, 0);
}

// here we need another block_sum instead of using block_reduce
// since we need manage shared memory in a explicit way
template <int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float sum) {
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 0x1f;

// Compute the sum per warp.
#pragma unroll
  for (int mask = (WARP_SIZE >> 1); mask > 0; mask >>= 1) {
    sum += SHFL_XOR_SYNC(sum, mask);
  }

  if (lane == 0) red_smem[warp] = sum;
  __syncthreads();

  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

// Parallel reduction of all tokens from the same sequence inside the warp.
#pragma unroll
  for (int mask = (NUM_WARPS >> 1); mask > 0; mask >>= 1) {
    sum += SHFL_XOR_SYNC(sum, mask);
  }

  // Broadcast to other threads.
  return SHFL_SYNC(sum, 0);
}

// here VecT is a vector of float, whose size is N
template <typename VecT, int NUM_WARPS, int NUM_THREADS_PER_GROUP, int N>
inline __device__ void block_sum(float* red_smem, VecT& acc) {
  float* acc_ptr = reinterpret_cast<float*>(&acc);
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 0x1f;

#pragma unroll
  for (int i = 0; i < N; i++) {
#pragma unroll
    for (int mask = (WARP_SIZE >> 1); mask >= NUM_THREADS_PER_GROUP;
         mask >>= 1) {
      acc_ptr[i] += SHFL_XOR_SYNC(acc_ptr[i], mask);
    }
  }

#pragma unroll
  for (int limit = NUM_WARPS; limit > 1; limit >>= 1) {
    int mid = limit >> 1;
    if (warp >= mid && warp < limit) {
      float* dst = red_smem + (warp - mid) * N * NUM_THREADS_PER_GROUP;
      if (lane < NUM_THREADS_PER_GROUP) {
        if constexpr (N == VEC_SIZE_8) {
          VecT* vdst = &((reinterpret_cast<VecT*>(dst))[lane]);
          const int idx0 = (lane >> 2) & 0x1;
          const int idx1 = idx0 ^ 0x1;
          (reinterpret_cast<float4*>(vdst))[idx0] =
              (reinterpret_cast<float4*>(acc_ptr))[idx0];
          (reinterpret_cast<float4*>(vdst))[idx1] =
              (reinterpret_cast<float4*>(acc_ptr))[idx1];
        } else {
          (reinterpret_cast<VecT*>(dst))[lane] = acc;
        }
      }
    }
    __syncthreads();

    if (warp < mid) {
      float* src = red_smem + warp * N * NUM_THREADS_PER_GROUP;
      VecT src_reg;
      if (lane < NUM_THREADS_PER_GROUP) {
        float* src_ptr = reinterpret_cast<float*>(&src_reg);
        if constexpr (N == VEC_SIZE_8) {
          VecT* vsrc = &((reinterpret_cast<VecT*>(src))[lane]);
          const int idx0 = (lane >> 2) & 0x1;
          const int idx1 = idx0 ^ 0x1;
          (reinterpret_cast<float4*>(src_ptr))[idx0] =
              (reinterpret_cast<float4*>(vsrc))[idx0];
          (reinterpret_cast<float4*>(src_ptr))[idx1] =
              (reinterpret_cast<float4*>(vsrc))[idx1];
        } else {
          src_reg = (reinterpret_cast<VecT*>(src))[lane];
        }
#pragma unroll
        for (int j = 0; j < N; j++) {
          acc_ptr[j] += src_ptr[j];
        }
      }
    }
    __syncthreads();
  }
}

#undef SHFL_SYNC
#undef SHFL_XOR_SYNC

}  // namespace attention
}  // namespace cuda
}  // namespace colossalAI
