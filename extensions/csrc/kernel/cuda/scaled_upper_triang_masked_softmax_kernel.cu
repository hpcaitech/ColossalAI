/*This code from NVIDIA Megatron:
 *     with minor changes. */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <assert.h>
#include <c10/macros/Macros.h>
#include <stdint.h>
#include <cfloat>
#include <limits>

#include "common/micros.h"
#include "utils/vec_copy.h"
#include "funcs/reduce_function.h"
#include "funcs/unary_functor.h"

using colossalAI::funcs::UnaryOpFunctor;
using colossalAI::funcs::UnaryOpType;
using colossalAI::funcs::warp_reduce;
using colossalAI::funcs::ReduceType;
using colossalAI::cuda::utils::copy;
using colossalAI::cuda::utils::copy_zero;

/*
 * Extended softmax (from native aten pytorch) with following additional
 * features 1) input scaling 2) Implicit time (diagonal masking)
 */
template <typename input_t, typename output_t, typename acc_t,
          int log2_elements>
__global__ void scaled_upper_triang_masked_softmax_warp_forward(
    output_t *dst, const input_t *src, const acc_t scale, int micro_batch_size,
    int stride, int element_count) {
  // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and
  // warp_size of method warp_softmax_forward_kernel.
  constexpr int next_power_of_two = 1 << log2_elements;
  constexpr int WARP_SIZE =
      (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
  constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;
  constexpr int ELEMENTS_PER_LDG_STG = (WARP_ITERATIONS < 4) ? 1 : 4;

  int first_batch =
      (blockDim.y * blockIdx.y + threadIdx.y) * gridDim.x * WARP_BATCH +
      blockIdx.x;
  int local_seq = blockIdx.x + 1;
  int warp_iteration_limit =
      (local_seq + ELEMENTS_PER_LDG_STG * WARP_SIZE - 1) / WARP_SIZE;

  // micro_batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = micro_batch_size - first_batch;
  if (local_batches > WARP_BATCH) local_batches = WARP_BATCH;

  // there might be multiple batches per warp. compute the index within the
  // batch
  int local_idx = threadIdx.x;

  src += first_batch * stride + ELEMENTS_PER_LDG_STG * local_idx;
  dst += first_batch * stride + ELEMENTS_PER_LDG_STG * local_idx;

  // load data from global memory
  acc_t elements[WARP_BATCH][WARP_ITERATIONS];
  input_t temp_data[ELEMENTS_PER_LDG_STG];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : local_seq;

#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
      int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;

      if (element_index < batch_element_count) {
        copy<input_t, ELEMENTS_PER_LDG_STG>(
            src + i * element_count * stride + it * WARP_SIZE, temp_data);

#pragma unroll
        for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
          if ((element_index + element) < batch_element_count) {
            elements[i][it + element] = (acc_t)temp_data[element] * scale;
          } else {
            elements[i][it + element] = -std::numeric_limits<acc_t>::infinity();
          }
        }
      } else {
#pragma unroll
        for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
          elements[i][it + element] = -std::numeric_limits<acc_t>::infinity();
        }
      }
    }
  }

  // compute max_value
  acc_t max_value[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    max_value[i] = elements[i][0];
#pragma unroll
    for (int it = 1; it < WARP_ITERATIONS; ++it) {
      max_value[i] =
          (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
    }
  }
  warp_reduce<acc_t,ReduceType::kMax,WARP_BATCH,WARP_SIZE>(max_value);

  acc_t sum[WARP_BATCH]{0.0f};
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      if (it < warp_iteration_limit) {
        elements[i][it] = std::exp((elements[i][it] - max_value[i]));
        sum[i] += elements[i][it];
      }
    }
  }
  warp_reduce<acc_t,ReduceType::kSum,WARP_BATCH,WARP_SIZE>(sum);


  // store result
  output_t out[ELEMENTS_PER_LDG_STG];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches) break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
      int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;

      if (element_index < local_seq) {
#pragma unroll
        for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
          if (element_index + element < local_seq) {
            out[element] = elements[i][it + element] / sum[i];
          } else {
            out[element] = 0;
          }
        }
        copy<output_t, ELEMENTS_PER_LDG_STG>(
            out, dst + i * element_count * stride + it * WARP_SIZE);
      } else if (element_index < element_count) {
        copy_zero<output_t, ELEMENTS_PER_LDG_STG>(
            dst + i * element_count * stride + it * WARP_SIZE);
      } else {
        break;
      }
    }
  }
}

template <typename input_t, typename output_t, typename acc_t,
          int log2_elements>
__global__ void scaled_upper_triang_masked_softmax_warp_backward(
    output_t *gradInput, input_t *grad, const input_t *output, acc_t scale,
    int micro_batch_size, int stride, int element_count) {
  // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and
  // warp_size of method warp_softmax_backward_kernel.
  constexpr int next_power_of_two = 1 << log2_elements;
  constexpr int WARP_SIZE =
      (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
  constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;
  constexpr int ELEMENTS_PER_LDG_STG = (WARP_ITERATIONS < 4) ? 1 : 4;

  int first_batch =
      (blockDim.y * blockIdx.y + threadIdx.y) * gridDim.x * WARP_BATCH +
      blockIdx.x;
  int local_seq = blockIdx.x + 1;

  // micro_batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = micro_batch_size - first_batch;
  if (local_batches > WARP_BATCH) local_batches = WARP_BATCH;

  // there might be multiple batches per warp. compute the index within the
  // batch
  int local_idx = threadIdx.x;

  // the first element to process by the current thread
  int thread_offset = first_batch * stride + ELEMENTS_PER_LDG_STG * local_idx;
  grad += thread_offset;
  output += thread_offset;
  gradInput += thread_offset;

  // load data from global memory
  acc_t grad_reg[WARP_BATCH][WARP_ITERATIONS]{0.0f};
  acc_t output_reg[WARP_BATCH][WARP_ITERATIONS]{0.0f};
  input_t temp_grad[ELEMENTS_PER_LDG_STG];
  input_t temp_output[ELEMENTS_PER_LDG_STG];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : local_seq;

#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
      int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
      if (element_index < batch_element_count) {
        copy<input_t, ELEMENTS_PER_LDG_STG>(
            grad + i * element_count * stride + it * WARP_SIZE, temp_grad);
        copy<input_t, ELEMENTS_PER_LDG_STG>(
            output + i * element_count * stride + it * WARP_SIZE, temp_output);

#pragma unroll
        for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
          if (element_index + element < batch_element_count) {
            output_reg[i][it + element] = (acc_t)temp_output[element];
          }
        }
#pragma unroll
        for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
          if (element_index + element < batch_element_count) {
            grad_reg[i][it + element] =
                (acc_t)temp_grad[element] * output_reg[i][it + element];
          }
        }
      }
    }
  }

  acc_t sum[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    sum[i] = grad_reg[i][0];
#pragma unroll
    for (int it = 1; it < WARP_ITERATIONS; ++it) {
      sum[i] += grad_reg[i][it];
    }
  }
  warp_reduce<acc_t,ReduceType::kSum,WARP_BATCH,WARP_SIZE>(sum);

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches) break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
      int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
        // compute gradients
        output_t out[ELEMENTS_PER_LDG_STG];
#pragma unroll
        for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
          out[element] =
              (output_t)(scale * (grad_reg[i][it + element] -
                                  output_reg[i][it + element] * sum[i]));
        }
        copy<output_t, ELEMENTS_PER_LDG_STG>(
            out, gradInput + i * element_count * stride + it * WARP_SIZE);
      }
    }
  }
}

template <typename input_t, typename output_t, typename acc_t>
void dispatch_scaled_upper_triang_masked_softmax_forward(
    output_t *dst, const input_t *src, const input_t scale,
    int softmax_elements, int softmax_elements_stride, int attn_batches) {
  TORCH_INTERNAL_ASSERT(softmax_elements >= 0 && softmax_elements <= 2048);
  if (softmax_elements == 0) {
    return;
  } else {
    int log2_elements = UnaryOpFunctor<int, int, UnaryOpType::kLog2Ceil>()(softmax_elements);
    const int next_power_of_two = 1 << log2_elements;
    int seq_len = softmax_elements;
    int batch_count = attn_batches * seq_len;

    // This value must match the WARP_SIZE constexpr value computed inside
    // softmax_warp_forward.
    int warp_size =
        (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;

    // This value must match the WARP_BATCH constexpr value computed inside
    // softmax_warp_forward.
    int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    TORCH_INTERNAL_ASSERT(attn_batches % batches_per_block == 0);

    int blocks_per_seq = attn_batches / batches_per_block;
    dim3 blocks(seq_len, blocks_per_seq, 1);
    dim3 threads(warp_size, warps_per_block, 1);
    // Launch code would be more elegant if C++ supported FOR CONSTEXPR
    switch (log2_elements) {
      case 0:  // 1
        scaled_upper_triang_masked_softmax_warp_forward<input_t, output_t,
                                                        acc_t, 0>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dst, src, scale, batch_count, softmax_elements_stride,
                softmax_elements);
        break;
      case 1:  // 2
        scaled_upper_triang_masked_softmax_warp_forward<input_t, output_t,
                                                        acc_t, 1>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dst, src, scale, batch_count, softmax_elements_stride,
                softmax_elements);
        break;
      case 2:  // 4
        scaled_upper_triang_masked_softmax_warp_forward<input_t, output_t,
                                                        acc_t, 2>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dst, src, scale, batch_count, softmax_elements_stride,
                softmax_elements);
        break;
      case 3:  // 8
        scaled_upper_triang_masked_softmax_warp_forward<input_t, output_t,
                                                        acc_t, 3>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dst, src, scale, batch_count, softmax_elements_stride,
                softmax_elements);
        break;
      case 4:  // 16
        scaled_upper_triang_masked_softmax_warp_forward<input_t, output_t,
                                                        acc_t, 4>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dst, src, scale, batch_count, softmax_elements_stride,
                softmax_elements);
        break;
      case 5:  // 32
        scaled_upper_triang_masked_softmax_warp_forward<input_t, output_t,
                                                        acc_t, 5>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dst, src, scale, batch_count, softmax_elements_stride,
                softmax_elements);
        break;
      case 6:  // 64
        scaled_upper_triang_masked_softmax_warp_forward<input_t, output_t,
                                                        acc_t, 6>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dst, src, scale, batch_count, softmax_elements_stride,
                softmax_elements);
        break;
      case 7:  // 128
        scaled_upper_triang_masked_softmax_warp_forward<input_t, output_t,
                                                        acc_t, 7>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dst, src, scale, batch_count, softmax_elements_stride,
                softmax_elements);
        break;
      case 8:  // 256
        scaled_upper_triang_masked_softmax_warp_forward<input_t, output_t,
                                                        acc_t, 8>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dst, src, scale, batch_count, softmax_elements_stride,
                softmax_elements);
        break;
      case 9:  // 512
        scaled_upper_triang_masked_softmax_warp_forward<input_t, output_t,
                                                        acc_t, 9>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dst, src, scale, batch_count, softmax_elements_stride,
                softmax_elements);
        break;
      case 10:  // 1024
        scaled_upper_triang_masked_softmax_warp_forward<input_t, output_t,
                                                        acc_t, 10>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dst, src, scale, batch_count, softmax_elements_stride,
                softmax_elements);
        break;
      case 11:  // 2048
        scaled_upper_triang_masked_softmax_warp_forward<input_t, output_t,
                                                        acc_t, 11>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dst, src, scale, batch_count, softmax_elements_stride,
                softmax_elements);
        break;
      default:
        break;
    }
  }
}

template <typename input_t, typename output_t, typename acc_t>
void dispatch_scaled_upper_triang_masked_softmax_backward(
    output_t *grad_input, input_t *grad, const input_t *output,
    const acc_t scale, int softmax_elements, int softmax_elements_stride,
    int attn_batches) {
  TORCH_INTERNAL_ASSERT(softmax_elements >= 0 && softmax_elements <= 2048);
  if (softmax_elements == 0) {
    return;
  } else {
    int log2_elements = UnaryOpFunctor<int, int, UnaryOpType::kLog2Ceil>()(softmax_elements);
    const int next_power_of_two = 1 << log2_elements;
    int seq_len = softmax_elements;
    int batch_count = attn_batches * seq_len;

    // This value must match the WARP_SIZE constexpr value computed inside
    // softmax_warp_backward.
    int warp_size =
        (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;

    // This value must match the WARP_BATCH constexpr value computed inside
    // softmax_warp_backward.
    int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    TORCH_INTERNAL_ASSERT(attn_batches % batches_per_block == 0);

    int blocks_per_seq = attn_batches / batches_per_block;
    dim3 blocks(seq_len, blocks_per_seq, 1);
    dim3 threads(warp_size, warps_per_block, 1);
    // Launch code would be more elegant if C++ supported FOR CONSTEXPR
    switch (log2_elements) {
      case 0:  // 1
        scaled_upper_triang_masked_softmax_warp_backward<input_t, output_t,
                                                         acc_t, 0>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_input, grad, output, scale, batch_count,
                softmax_elements_stride, softmax_elements);
        break;
      case 1:  // 2
        scaled_upper_triang_masked_softmax_warp_backward<input_t, output_t,
                                                         acc_t, 1>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_input, grad, output, scale, batch_count,
                softmax_elements_stride, softmax_elements);
        break;
      case 2:  // 4
        scaled_upper_triang_masked_softmax_warp_backward<input_t, output_t,
                                                         acc_t, 2>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_input, grad, output, scale, batch_count,
                softmax_elements_stride, softmax_elements);
        break;
      case 3:  // 8
        scaled_upper_triang_masked_softmax_warp_backward<input_t, output_t,
                                                         acc_t, 3>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_input, grad, output, scale, batch_count,
                softmax_elements_stride, softmax_elements);
        break;
      case 4:  // 16
        scaled_upper_triang_masked_softmax_warp_backward<input_t, output_t,
                                                         acc_t, 4>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_input, grad, output, scale, batch_count,
                softmax_elements_stride, softmax_elements);
        break;
      case 5:  // 32
        scaled_upper_triang_masked_softmax_warp_backward<input_t, output_t,
                                                         acc_t, 5>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_input, grad, output, scale, batch_count,
                softmax_elements_stride, softmax_elements);
        break;
      case 6:  // 64
        scaled_upper_triang_masked_softmax_warp_backward<input_t, output_t,
                                                         acc_t, 6>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_input, grad, output, scale, batch_count,
                softmax_elements_stride, softmax_elements);
        break;
      case 7:  // 128
        scaled_upper_triang_masked_softmax_warp_backward<input_t, output_t,
                                                         acc_t, 7>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_input, grad, output, scale, batch_count,
                softmax_elements_stride, softmax_elements);
        break;
      case 8:  // 256
        scaled_upper_triang_masked_softmax_warp_backward<input_t, output_t,
                                                         acc_t, 8>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_input, grad, output, scale, batch_count,
                softmax_elements_stride, softmax_elements);
        break;
      case 9:  // 512
        scaled_upper_triang_masked_softmax_warp_backward<input_t, output_t,
                                                         acc_t, 9>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_input, grad, output, scale, batch_count,
                softmax_elements_stride, softmax_elements);
        break;
      case 10:  // 1024
        scaled_upper_triang_masked_softmax_warp_backward<input_t, output_t,
                                                         acc_t, 10>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_input, grad, output, scale, batch_count,
                softmax_elements_stride, softmax_elements);
        break;
      case 11:  // 2048
        scaled_upper_triang_masked_softmax_warp_backward<input_t, output_t,
                                                         acc_t, 11>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_input, grad, output, scale, batch_count,
                softmax_elements_stride, softmax_elements);
        break;
      default:
        break;
    }
  }
}




torch::Tensor fwd_cuda(torch::Tensor const& input, float scale_factor) {
  // input is a 3d tensor with dimensions [attn_batches, seq_len, seq_len]
  const int attn_batches = input.size(0);
  const int seq_len = input.size(1);
  TORCH_INTERNAL_ASSERT(seq_len <= 2048);

  // Output
  auto act_options = input.options().requires_grad(false);
  torch::Tensor softmax_results =
      torch::empty({attn_batches, seq_len, seq_len}, act_options);

  // Softmax Intermediate Result Ptr
  void* input_ptr = static_cast<void*>(input.data_ptr());
  void* softmax_results_ptr = static_cast<void*>(softmax_results.data_ptr());

  DISPATCH_HALF_AND_BFLOAT(
      input.scalar_type(),
      "dispatch_scaled_upper_triang_masked_softmax_forward",
      dispatch_scaled_upper_triang_masked_softmax_forward<scalar_t, scalar_t,
                                                          float>(
          reinterpret_cast<scalar_t*>(softmax_results_ptr),
          reinterpret_cast<const scalar_t*>(input_ptr), scale_factor, seq_len,
          seq_len, attn_batches););
  return softmax_results;
}

torch::Tensor bwd_cuda(torch::Tensor const& output_grads_,
                       torch::Tensor const& softmax_results_,
                       float scale_factor) {
  auto output_grads = output_grads_.contiguous();
  auto softmax_results = softmax_results_.contiguous();

  // output grads is a 3d tensor with dimensions [attn_batches, seq_len,
  // seq_len]
  const int attn_batches = output_grads.size(0);
  const int seq_len = output_grads.size(1);
  TORCH_INTERNAL_ASSERT(output_grads.size(1) == output_grads.size(2));

  void* output_grads_ptr = static_cast<void*>(output_grads.data_ptr());

  // Softmax Grad
  DISPATCH_HALF_AND_BFLOAT(
      output_grads_.scalar_type(),
      "dispatch_scaled_upper_triang_masked_softmax_backward",
      dispatch_scaled_upper_triang_masked_softmax_backward<scalar_t, scalar_t,
                                                           float>(
          reinterpret_cast<scalar_t*>(output_grads_ptr),
          reinterpret_cast<scalar_t*>(output_grads_ptr),
          reinterpret_cast<scalar_t const*>(softmax_results.data_ptr()),
          scale_factor, seq_len, seq_len, attn_batches););

  // backward pass is completely in-place
  return output_grads;
}
