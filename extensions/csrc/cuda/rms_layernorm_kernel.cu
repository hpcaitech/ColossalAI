/*This code from VLLM:
 *     https://github.com/vllm-project/vllm/
 *     with minor changes. */

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdio.h>


#include "block_reduce.h"
#include "../common/micros.h"

template<typename scalar_t>
__global__ void rms_layernorm_kernel(
  scalar_t* __restrict__ out,             // [..., hidden_size]
  const scalar_t* __restrict__ input,     // [..., hidden_size]
  const scalar_t* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;
  /*
   * since the open-sourced LLM's hidden dimensions mainly range from
   * 4096 (LLAMA-7B) to 8192 (LLAMA-65B), we thus set the supported
   * hidden dimension limit to 8192, and each thread's capacity
   * for caching input tensors to 8 (8192 = 8 * 1024) which
   * will cause problems for extremely large models, such as
   * Megatron-Turing NLG 530B with hidden dimensions up to 20480
   */
  float x_local[8];

  for (int idx = threadIdx.x, cnt = 0; idx < hidden_size; idx += blockDim.x, cnt++) {
    x_local[cnt] = (float) input[blockIdx.x * hidden_size + idx];
    variance += x_local[cnt] * x_local[cnt];
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x, cnt = 0; idx < hidden_size; idx += blockDim.x, cnt++) {
    out[blockIdx.x * hidden_size + idx] = ((scalar_t) (x_local[cnt] * s_variance)) * weight[idx];
  }
}

template<typename scalar_t>
__global__ void fused_add_rms_layernorm_kernel(
  scalar_t* __restrict__ input,           // [..., hidden_size]
  scalar_t* __restrict__ residual,        // [..., hidden_size]
  const scalar_t* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;
  float x_local[8];

  for (int idx = threadIdx.x, cnt = 0; idx < hidden_size; idx += blockDim.x, cnt++) {
    x_local[cnt] = (float) input[blockIdx.x * hidden_size + idx];
    x_local[cnt] += (float) residual[blockIdx.x * hidden_size + idx];
    variance += x_local[cnt] * x_local[cnt];
    residual[blockIdx.x * hidden_size + idx] = (scalar_t) x_local[cnt];
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x, cnt = 0; idx < hidden_size; idx += blockDim.x, cnt++) {
    input[blockIdx.x * hidden_size + idx] = ((scalar_t) (x_local[cnt] * s_variance)) * weight[idx];
  }
}

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

  DISPATCH_FLOAT_HALF_AND_BFLOAT(
    input.scalar_type(),
    "rms_layernorm_kernel",
    rms_layernorm_kernel<scalar_t><<<grid, block, 0, stream>>>(
      out.data_ptr<scalar_t>(),
      input.data_ptr<scalar_t>(),
      weight.data_ptr<scalar_t>(),
      epsilon,
      num_tokens,
      hidden_size);)
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

  DISPATCH_FLOAT_HALF_AND_BFLOAT(
    input.scalar_type(),
    "fused_add_rms_layernorm_kernel",
    fused_add_rms_layernorm_kernel<scalar_t><<<grid, block, 0, stream>>>(
      input.data_ptr<scalar_t>(),
      residual.data_ptr<scalar_t>(),
      weight.data_ptr<scalar_t>(),
      epsilon,
      num_tokens,
      hidden_size);)
}
