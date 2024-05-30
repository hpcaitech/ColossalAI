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


/*
 * Extended softmax (from native aten pytorch) with following additional
 * features 1) input scaling 2) Explicit masking
 */
template <typename input_t, typename output_t, typename acc_t,
          int log2_elements>
__global__ void scaled_masked_softmax_warp_forward(
    output_t *dst, const input_t *src, const uint8_t *mask, const acc_t scale,
    int micro_batch_size, int element_count, int pad_batches) {
  // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and
  // warp_size of method warp_softmax_forward_kernel.
  constexpr int next_power_of_two = 1 << log2_elements;
  constexpr int WARP_SIZE =
      (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
  constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;
  constexpr int ELEMENTS_PER_LDG_STG = (WARP_ITERATIONS < 4) ? 1 : 4;

  // blockDim/threadIdx = (WARP_SIZE, WARPS_PER_BLOCK, )
  // gridDim/blockIdx = (seq_len, attn_heads, batches)
  int first_batch =
      (blockDim.y *
           (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) +
       threadIdx.y) *
      WARP_BATCH;
  int pad_first_batch = 0;
  if (pad_batches != 1) {  // bert style
    pad_first_batch =
        (blockDim.y * (blockIdx.x + gridDim.x * blockIdx.z) + threadIdx.y) *
        WARP_BATCH;
  } else {  // gpt2 style
    pad_first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;
  }

  // micro_batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = micro_batch_size - first_batch;
  if (local_batches > WARP_BATCH) local_batches = WARP_BATCH;

  // there might be multiple batches per warp. compute the index within the
  // batch
  int local_idx = threadIdx.x;

  src += first_batch * element_count + ELEMENTS_PER_LDG_STG * local_idx;
  dst += first_batch * element_count + ELEMENTS_PER_LDG_STG * local_idx;
  mask += pad_first_batch * element_count + ELEMENTS_PER_LDG_STG * local_idx;

  // load data from global memory
  acc_t elements[WARP_BATCH][WARP_ITERATIONS];
  input_t temp_data[ELEMENTS_PER_LDG_STG];
  uint8_t temp_mask[ELEMENTS_PER_LDG_STG];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;

#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
      int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;

      if (element_index < batch_element_count) {
        int itr_idx = i * element_count + it * WARP_SIZE;
        copy<input_t, ELEMENTS_PER_LDG_STG>(src + itr_idx, temp_data);
        copy<uint8_t, ELEMENTS_PER_LDG_STG>(mask + itr_idx, temp_mask);

#pragma unroll
        for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
          if (temp_mask[element] != 1) {
            elements[i][it + element] = (acc_t)temp_data[element] * scale;
          } else {
            elements[i][it + element] = -10000.0;
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
      elements[i][it] = std::exp((elements[i][it] - max_value[i]));
      sum[i] += elements[i][it];
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
      if (element_index < element_count) {
#pragma unroll
        for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
          out[element] = elements[i][it + element] / sum[i];
        }
        copy<output_t, ELEMENTS_PER_LDG_STG>(
          out,  dst + i * element_count + it * WARP_SIZE);
      } else {
        break;
      }
    }
  }
}

template <typename input_t, typename output_t, typename acc_t,
          int log2_elements>
__global__ void scaled_masked_softmax_warp_backward(
    output_t *gradInput, input_t *grad, const input_t *output, acc_t scale,
    int micro_batch_size, int element_count) {
  // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and
  // warp_size of method warp_softmax_backward_kernel.
  constexpr int next_power_of_two = 1 << log2_elements;
  constexpr int WARP_SIZE =
      (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
  constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;
  constexpr int ELEMENTS_PER_LDG_STG = (WARP_ITERATIONS < 4) ? 1 : 4;

  // blockDim/threadIdx = (WARP_SIZE, WARPS_PER_BLOCK, )
  // gridDim/blockIdx = (seq_len, attn_heads, batches)
  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  // micro_batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = micro_batch_size - first_batch;
  if (local_batches > WARP_BATCH) local_batches = WARP_BATCH;

  // there might be multiple batches per warp. compute the index within the
  // batch
  int local_idx = threadIdx.x;

  // the first element to process by the current thread
  int thread_offset =
      first_batch * element_count + ELEMENTS_PER_LDG_STG * local_idx;
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
    int batch_element_count = (i >= local_batches) ? 0 : element_count;

#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
      int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
      if (element_index < batch_element_count) {
        copy<input_t, ELEMENTS_PER_LDG_STG>(
            grad + i * element_count + it * WARP_SIZE, temp_grad);
        copy<input_t, ELEMENTS_PER_LDG_STG>(
            output + i * element_count + it * WARP_SIZE, temp_output);

#pragma unroll
        for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
          output_reg[i][it + element] = (acc_t)temp_output[element];
        }
#pragma unroll
        for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
          grad_reg[i][it + element] =
              (acc_t)temp_grad[element] * output_reg[i][it + element];
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
          out, gradInput + i * element_count + it * WARP_SIZE);
      }
    }
  }
}


int get_batch_per_block(int query_seq_len, int key_seq_len, int batches,
                        int attn_heads) {
  int log2_elements = UnaryOpFunctor<int, int, UnaryOpType::kLog2Ceil>()(key_seq_len);
  const int next_power_of_two = 1 << log2_elements;

  int warp_size =
      (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
  int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

  constexpr int threads_per_block = 128;
  int warps_per_block = (threads_per_block / warp_size);
  int batches_per_block = warps_per_block * batches_per_warp;

  return batches_per_block;
}

template <typename input_t, typename output_t, typename acc_t>
void dispatch_scaled_masked_softmax_forward(output_t *dst, const input_t *src,
                                            const uint8_t *mask,
                                            const input_t scale,
                                            int query_seq_len, int key_seq_len,
                                            int batches, int attn_heads,
                                            int pad_batches) {
  TORCH_INTERNAL_ASSERT(key_seq_len >= 0 && key_seq_len <= 2048);
  if (key_seq_len == 0) {
    return;
  } else {
    int log2_elements = UnaryOpFunctor<int, int, UnaryOpType::kLog2Ceil>()(key_seq_len);
    const int next_power_of_two = 1 << log2_elements;
    int batch_count = batches * attn_heads * query_seq_len;

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
    TORCH_INTERNAL_ASSERT(query_seq_len % batches_per_block == 0);
    dim3 blocks(query_seq_len / batches_per_block, attn_heads, batches);
    dim3 threads(warp_size, warps_per_block, 1);
    // Launch code would be more elegant if C++ supported FOR CONSTEXPR
    switch (log2_elements) {
      case 0:  // 1
        scaled_masked_softmax_warp_forward<input_t, output_t, acc_t, 0>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dst, src, mask, scale, batch_count, key_seq_len, pad_batches);
        break;
      case 1:  // 2
        scaled_masked_softmax_warp_forward<input_t, output_t, acc_t, 1>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dst, src, mask, scale, batch_count, key_seq_len, pad_batches);
        break;
      case 2:  // 4
        scaled_masked_softmax_warp_forward<input_t, output_t, acc_t, 2>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dst, src, mask, scale, batch_count, key_seq_len, pad_batches);
        break;
      case 3:  // 8
        scaled_masked_softmax_warp_forward<input_t, output_t, acc_t, 3>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dst, src, mask, scale, batch_count, key_seq_len, pad_batches);
        break;
      case 4:  // 16
        scaled_masked_softmax_warp_forward<input_t, output_t, acc_t, 4>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dst, src, mask, scale, batch_count, key_seq_len, pad_batches);
        break;
      case 5:  // 32
        scaled_masked_softmax_warp_forward<input_t, output_t, acc_t, 5>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dst, src, mask, scale, batch_count, key_seq_len, pad_batches);
        break;
      case 6:  // 64
        scaled_masked_softmax_warp_forward<input_t, output_t, acc_t, 6>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dst, src, mask, scale, batch_count, key_seq_len, pad_batches);
        break;
      case 7:  // 128
        scaled_masked_softmax_warp_forward<input_t, output_t, acc_t, 7>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dst, src, mask, scale, batch_count, key_seq_len, pad_batches);
        break;
      case 8:  // 256
        scaled_masked_softmax_warp_forward<input_t, output_t, acc_t, 8>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dst, src, mask, scale, batch_count, key_seq_len, pad_batches);
        break;
      case 9:  // 512
        scaled_masked_softmax_warp_forward<input_t, output_t, acc_t, 9>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dst, src, mask, scale, batch_count, key_seq_len, pad_batches);
        break;
      case 10:  // 1024
        scaled_masked_softmax_warp_forward<input_t, output_t, acc_t, 10>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dst, src, mask, scale, batch_count, key_seq_len, pad_batches);
        break;
      case 11:  // 2048
        scaled_masked_softmax_warp_forward<input_t, output_t, acc_t, 11>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dst, src, mask, scale, batch_count, key_seq_len, pad_batches);
        break;
      default:
        break;
    }
  }
}

template <typename input_t, typename output_t, typename acc_t>
void dispatch_scaled_masked_softmax_backward(output_t *grad_input,
                                             input_t *grad,
                                             const input_t *output,
                                             const acc_t scale,
                                             int query_seq_len, int key_seq_len,
                                             int batches, int attn_heads) {
  TORCH_INTERNAL_ASSERT(key_seq_len >= 0 && key_seq_len <= 2048);
  if (key_seq_len == 0) {
    return;
  } else {
    int log2_elements = UnaryOpFunctor<int, int, UnaryOpType::kLog2Ceil>()(key_seq_len);
    const int next_power_of_two = 1 << log2_elements;
    int batch_count = batches * attn_heads * query_seq_len;

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
    int blocks = batch_count / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);
    // Launch code would be more elegant if C++ supported FOR CONSTEXPR
    switch (log2_elements) {
      case 0:  // 1
        scaled_masked_softmax_warp_backward<input_t, output_t, acc_t, 0>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_input, grad, output, scale, batch_count, key_seq_len);
        break;
      case 1:  // 2
        scaled_masked_softmax_warp_backward<input_t, output_t, acc_t, 1>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_input, grad, output, scale, batch_count, key_seq_len);
        break;
      case 2:  // 4
        scaled_masked_softmax_warp_backward<input_t, output_t, acc_t, 2>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_input, grad, output, scale, batch_count, key_seq_len);
        break;
      case 3:  // 8
        scaled_masked_softmax_warp_backward<input_t, output_t, acc_t, 3>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_input, grad, output, scale, batch_count, key_seq_len);
        break;
      case 4:  // 16
        scaled_masked_softmax_warp_backward<input_t, output_t, acc_t, 4>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_input, grad, output, scale, batch_count, key_seq_len);
        break;
      case 5:  // 32
        scaled_masked_softmax_warp_backward<input_t, output_t, acc_t, 5>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_input, grad, output, scale, batch_count, key_seq_len);
        break;
      case 6:  // 64
        scaled_masked_softmax_warp_backward<input_t, output_t, acc_t, 6>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_input, grad, output, scale, batch_count, key_seq_len);
        break;
      case 7:  // 128
        scaled_masked_softmax_warp_backward<input_t, output_t, acc_t, 7>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_input, grad, output, scale, batch_count, key_seq_len);
        break;
      case 8:  // 256
        scaled_masked_softmax_warp_backward<input_t, output_t, acc_t, 8>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_input, grad, output, scale, batch_count, key_seq_len);
        break;
      case 9:  // 512
        scaled_masked_softmax_warp_backward<input_t, output_t, acc_t, 9>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_input, grad, output, scale, batch_count, key_seq_len);
        break;
      case 10:  // 1024
        scaled_masked_softmax_warp_backward<input_t, output_t, acc_t, 10>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_input, grad, output, scale, batch_count, key_seq_len);
        break;
      case 11:  // 2048
        scaled_masked_softmax_warp_backward<input_t, output_t, acc_t, 11>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_input, grad, output, scale, batch_count, key_seq_len);
        break;
      default:
        break;
    }
  }
}

torch::Tensor fwd_cuda(torch::Tensor const& input, torch::Tensor const& mask,
                       float scale_factor) {
  // input is a 4d tensor with dimensions [batches, attn_heads, seq_len,
  // seq_len]
  const int batches = input.size(0);
  const int pad_batches = mask.size(0);
  const int attn_heads = input.size(1);
  const int query_seq_len = input.size(2);
  const int key_seq_len = input.size(3);
  TORCH_INTERNAL_ASSERT(key_seq_len <= 2048);
  TORCH_INTERNAL_ASSERT(query_seq_len > 1);
  TORCH_INTERNAL_ASSERT(pad_batches == 1 || pad_batches == batches);
  TORCH_INTERNAL_ASSERT(mask.size(1) == 1);
  TORCH_INTERNAL_ASSERT(mask.size(2) == query_seq_len);
  TORCH_INTERNAL_ASSERT(mask.size(3) == key_seq_len);

  // Output
  auto act_options = input.options().requires_grad(false);
  torch::Tensor softmax_results = torch::empty(
      {batches, attn_heads, query_seq_len, key_seq_len}, act_options);

  // Softmax Intermediate Result Ptr
  void* input_ptr = static_cast<void*>(input.data_ptr());
  void* mask_ptr = static_cast<void*>(mask.data_ptr());
  void* softmax_results_ptr = static_cast<void*>(softmax_results.data_ptr());

  DISPATCH_HALF_AND_BFLOAT(
      input.scalar_type(), "dispatch_scaled_masked_softmax_forward",
      dispatch_scaled_masked_softmax_forward<scalar_t, scalar_t, float>(
          reinterpret_cast<scalar_t*>(softmax_results_ptr),
          reinterpret_cast<const scalar_t*>(input_ptr),
          reinterpret_cast<const uint8_t*>(mask_ptr), scale_factor,
          query_seq_len, key_seq_len, batches, attn_heads, pad_batches););
  return softmax_results;
}

torch::Tensor bwd_cuda(torch::Tensor const& output_grads_,
                       torch::Tensor const& softmax_results_,
                       float scale_factor) {
  auto output_grads = output_grads_.contiguous();
  auto softmax_results = softmax_results_.contiguous();

  // output grads is a 4d tensor with dimensions [batches, attn_heads, seq_len,
  // seq_len]
  const int batches = output_grads.size(0);
  const int attn_heads = output_grads.size(1);
  const int query_seq_len = output_grads.size(2);
  const int key_seq_len = output_grads.size(3);

  void* output_grads_ptr = static_cast<void*>(output_grads.data_ptr());

  // Softmax Grad
  DISPATCH_HALF_AND_BFLOAT(
      output_grads_.scalar_type(), "dispatch_scaled_masked_softmax_backward",
      dispatch_scaled_masked_softmax_backward<scalar_t, scalar_t, float>(
          reinterpret_cast<scalar_t*>(output_grads_ptr),
          reinterpret_cast<scalar_t*>(output_grads_ptr),
          reinterpret_cast<scalar_t const*>(softmax_results.data_ptr()),
          scale_factor, query_seq_len, key_seq_len, batches, attn_heads););

  // backward pass is completely in-place
  return output_grads;
}
