/*This code from FasterTransformer:
 *     https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/kernels/layernorm_kernels.cu
 *     with minor changes. */

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>


#include "common/micros.h"
#include "funcs/cast_functor.h"
#include "funcs/binary_functor.h"
#include "funcs/reduce_function.h"
#include "common/vec_type_traits.h"

using colossalAI::funcs::block_reduce;
using colossalAI::funcs::ReduceType;
using colossalAI::funcs::CastFunctor;
using colossalAI::funcs::BinaryOpFunctor;
using colossalAI::funcs::BinaryOpType;
using colossalAI::common::VecTypeTrait;

#define RMSNORM_LAUNCHER(UNROLL_FACTOR, THREADDIM)                                 \
  DISPATCH_RMSNORM_FLOAT_HALF_AND_BFLOAT(                                          \
    input.element_size(),                                                          \
    input.scalar_type(),                                                           \
    "rms_layernorm_kernel",                                                        \
    rms_layernorm_kernel<scalar_t, UNROLL_FACTOR><<<grid, THREADDIM, 0, stream>>>( \
      out.data_ptr<scalar_t>(),                                                    \
      input.data_ptr<scalar_t>(),                                                  \
      weight.data_ptr<scalar_t>(),                                                 \
      epsilon,                                                                     \
      num_tokens,                                                                  \
      hidden_size);)                                                               \

#define FUSED_ADD_RMSNORM_LAUNCHER(UNROLL_FACTOR, THREADDIM)                                  \
  DISPATCH_RMSNORM_FLOAT_HALF_AND_BFLOAT(                                                     \
    input.element_size(),                                                                     \
    input.scalar_type(),                                                                      \
    "fused_add_rms_layernorm_kernel",                                                         \
    fused_add_rms_layernorm_kernel<scalar_t, UNROLL_FACTOR><<<grid, THREADDIM, 0, stream>>>(  \
      input.data_ptr<scalar_t>(),                                                             \
      residual.data_ptr<scalar_t>(),                                                          \
      weight.data_ptr<scalar_t>(),                                                            \
      epsilon,                                                                                \
      num_tokens,                                                                             \
      hidden_size);)                                                                          \

// optimized for half and bf16
template<typename scalar_t, int unroll_factor>
__global__ void rms_layernorm_kernel(
  scalar_t* __restrict__ out,             // [..., hidden_size]
  const scalar_t* __restrict__ input,     // [..., hidden_size]
  const scalar_t* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int hidden_size) {
  using scalar2_t = typename VecTypeTrait<scalar_t, 2>::Type;
  BinaryOpFunctor<scalar2_t, scalar2_t, scalar2_t, BinaryOpType::kMul> mul_scalar2t;
  __shared__ float s_variance;

  /*
   * since the open-sourced LLM's hidden dimensions mainly range from
   * 4096 (LLAMA-7B) to 8192 (LLAMA-65B), we thus set the supported
   * hidden dimension limit to 8192, and each thread's capacity
   * for caching input tensors to 8 (8192 = 8 * 1024) which
   * will cause problems for extremely large models, such as
   * Megatron-Turing NLG 530B with hidden dimensions up to 20480
   */
  scalar2_t x_local[4];

  scalar2_t* out_ptr = (scalar2_t*)out;
  const scalar2_t* input_ptr = (scalar2_t*)input;
  const scalar2_t* weight_ptr = (const scalar2_t*)weight;

  float variance = 0.0f;
  int row_offset = blockIdx.x * hidden_size / 2;


#pragma unroll unroll_factor
  for (int idx = threadIdx.x, cnt = 0; idx < hidden_size / 2; idx += blockDim.x, cnt++) {
    int id = row_offset + idx;
    x_local[cnt] = input_ptr[id];
    float v1 = CastFunctor<scalar_t,float>()(x_local[cnt].x);
    float v2 = CastFunctor<scalar_t,float>()(x_local[cnt].y);
    variance += v1 * v1 + v2 * v2;
  }
  block_reduce<float, ReduceType::kSum,1>(&variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  scalar2_t s_variance_2 = CastFunctor<float,scalar2_t>()(s_variance);
#pragma unroll unroll_factor
  for (int idx = threadIdx.x, cnt = 0; idx < hidden_size / 2; idx += blockDim.x, cnt++) {
    int id = row_offset + idx;
    out_ptr[id] = mul_scalar2t(mul_scalar2t(x_local[cnt], s_variance_2), weight_ptr[idx]);
  }
}

template<typename scalar_t, int unroll_factor>
__global__ void general_rms_layernorm_kernel(
  scalar_t* __restrict__ out,             // [..., hidden_size]
  const scalar_t* __restrict__ input,     // [..., hidden_size]
  const scalar_t* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;
  float x_local[8];

  int row_offset = blockIdx.x * hidden_size;

#pragma unroll unroll_factor
  for (int idx = threadIdx.x, cnt = 0; idx < hidden_size; idx += blockDim.x, cnt++) {
    int id = row_offset + idx;
    x_local[cnt] = (float) input[id];
    variance += x_local[cnt] * x_local[cnt];
  }
  block_reduce<float, ReduceType::kSum,1>(&variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

#pragma unroll unroll_factor
  for (int idx = threadIdx.x, cnt = 0; idx < hidden_size; idx += blockDim.x, cnt++) {
    int id = row_offset + idx;
    out[id] = ((scalar_t) (x_local[cnt] * s_variance)) * weight[idx];
  }
}

// optimized for half and bf16
template<typename scalar_t, int unroll_factor>
__global__ void fused_add_rms_layernorm_kernel(
  scalar_t* __restrict__ input,           // [..., hidden_size]
  scalar_t* __restrict__ residual,        // [..., hidden_size]
  const scalar_t* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int hidden_size) {
  using scalar2_t = typename VecTypeTrait<scalar_t, 2>::Type;
  BinaryOpFunctor<scalar2_t, scalar2_t, scalar2_t, BinaryOpType::kAdd> add_scalar2t;
  BinaryOpFunctor<scalar2_t, scalar2_t, scalar2_t, BinaryOpType::kMul> mul_scalar2t;

  __shared__ float s_variance;
  scalar2_t x_local[4];

  scalar2_t* input_ptr = (scalar2_t*)input;
  scalar2_t* residual_ptr = (scalar2_t*)residual;
  const scalar2_t* weight_ptr = (const scalar2_t*)weight;

  float variance = 0.0f;
  int row_offset = blockIdx.x * hidden_size / 2;

#pragma unroll unroll_factor
  for (int idx = threadIdx.x, cnt = 0; idx < hidden_size / 2; idx += blockDim.x, cnt++) {
    int id = row_offset + idx;
    x_local[cnt] = input_ptr[id];
    x_local[cnt] = add_scalar2t(x_local[cnt], residual_ptr[id]);
    float v1 = CastFunctor<scalar_t,float>()(x_local[cnt].x);
    float v2 = CastFunctor<scalar_t,float>()(x_local[cnt].y);
    variance += v1 * v1 + v2 * v2;
    residual_ptr[id] = x_local[cnt];
  }
  block_reduce<float, ReduceType::kSum,1>(&variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  scalar2_t s_variance_2 = CastFunctor<float, scalar2_t>()(s_variance);

#pragma unroll unroll_factor
  for (int idx = threadIdx.x, cnt = 0; idx < hidden_size / 2; idx += blockDim.x, cnt++) {
    int id = row_offset + idx;
    input_ptr[id] = mul_scalar2t(mul_scalar2t(x_local[cnt], s_variance_2), weight_ptr[idx]);
  }
}

template<typename scalar_t, int unroll_factor>
__global__ void general_fused_add_rms_layernorm_kernel(
  scalar_t* __restrict__ input,           // [..., hidden_size]
  scalar_t* __restrict__ residual,        // [..., hidden_size]
  const scalar_t* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;
  float x_local[8];

  int row_offset = blockIdx.x * hidden_size;

#pragma unroll unroll_factor
  for (int idx = threadIdx.x, cnt = 0; idx < hidden_size; idx += blockDim.x, cnt++) {
    int id = row_offset + idx;
    x_local[cnt] = (float) input[id];
    x_local[cnt] += (float) residual[id];
    variance += x_local[cnt] * x_local[cnt];
    residual[id] = (scalar_t) x_local[cnt];
  }
  block_reduce<float, ReduceType::kSum,1>(&variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

#pragma unroll unroll_factor
  for (int idx = threadIdx.x, cnt = 0; idx < hidden_size; idx += blockDim.x, cnt++) {
    int id = row_offset + idx;
    input[id] = ((scalar_t) (x_local[cnt] * s_variance)) * weight[idx];
  }
}


#define DISPATCH_RMSNORM_FLOAT_HALF_AND_BFLOAT(DATA_SIZE, TYPE, NAME, ...)  \
  if (DATA_SIZE == 2) {                                                     \
    switch (TYPE) {                                                         \
      case at::ScalarType::Half: {                                          \
        using scalar_t = at::Half;                                          \
        __VA_ARGS__;                                                        \
        break;                                                              \
      }                                                                     \
      case at::ScalarType::BFloat16: {                                      \
        using scalar_t = at::BFloat16;                                      \
        __VA_ARGS__;                                                        \
        break;                                                              \
      }                                                                     \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");     \
    }                                                                       \
  } else {                                                                  \
    switch (TYPE) {                                                         \
      case at::ScalarType::Float: {                                         \
        using scalar_t = float;                                             \
        general_##__VA_ARGS__;                                              \
        break;                                                              \
      }                                                                     \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");     \
    }                                                                       \
  }                                                                         \


void rms_layernorm(
  torch::Tensor& out,      // [..., hidden_size]
  torch::Tensor& input,    // [..., hidden_size]
  torch::Tensor& weight,   // [hidden_size]
  float epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (num_tokens >= 512) {
    if (input.scalar_type() == at::ScalarType::Float) {
      RMSNORM_LAUNCHER(8, hidden_size / 8);
    } else {
      RMSNORM_LAUNCHER(4, hidden_size / 8);
    }
  } else {
    int unroll_factor = (hidden_size + block.x - 1) / block.x;
    if (input.scalar_type() != at::ScalarType::Float) {
      block.x = std::min(hidden_size / 2, 1024);
      unroll_factor = (hidden_size / 2 + block.x - 1) / block.x;
    }
    switch (unroll_factor) {
      case 1:
        RMSNORM_LAUNCHER(1, block);
        break;
      case 2:
        RMSNORM_LAUNCHER(2, block);
        break;
      case 3:
        RMSNORM_LAUNCHER(3, block);
        break;
      case 4:
        RMSNORM_LAUNCHER(4, block);
        break;
      case 5:
        RMSNORM_LAUNCHER(5, block);
        break;
      case 8:
        RMSNORM_LAUNCHER(8, block);
        break;
      default:
        AT_ERROR("unroll_factor must be 1, 2, 3, 4, 5 or 8");
    }
  }
}

void fused_add_rms_layernorm(
  torch::Tensor& input,    // [..., hidden_size]
  torch::Tensor& residual, // [..., hidden_size]
  torch::Tensor& weight,   // [hidden_size]
  float epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (num_tokens >= 512) {
    if (input.scalar_type() == at::ScalarType::Float) {
      FUSED_ADD_RMSNORM_LAUNCHER(8, hidden_size / 8);
    } else {
      FUSED_ADD_RMSNORM_LAUNCHER(4, hidden_size / 8);
    }
  } else {
    int unroll_factor = (hidden_size + block.x - 1) / block.x;
    if (input.scalar_type() != at::ScalarType::Float) {
      block.x = std::min(hidden_size / 2, 1024);
      unroll_factor = (hidden_size / 2 + block.x - 1) / block.x;
    }
    switch (unroll_factor) {
      case 1:
        FUSED_ADD_RMSNORM_LAUNCHER(1, block);
        break;
      case 2:
        FUSED_ADD_RMSNORM_LAUNCHER(2, block);
        break;
      case 3:
        FUSED_ADD_RMSNORM_LAUNCHER(3, block);
        break;
      case 4:
        FUSED_ADD_RMSNORM_LAUNCHER(4, block);
        break;
      case 5:
        FUSED_ADD_RMSNORM_LAUNCHER(5, block);
        break;
      case 8:
        FUSED_ADD_RMSNORM_LAUNCHER(8, block);
        break;
      default:
        AT_ERROR("unroll_factor must be 1, 2, 3, 4, 5 or 8");
    }
  }
}

#undef DISPATCH_RMSNORM_FLOAT_HALF_AND_BFLOAT
