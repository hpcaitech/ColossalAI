// modified from
// https://github.com/NVIDIA/apex/blob/master/csrc/multi_tensor_sgd_kernel.cu
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "compat.h"
#include "multi_tensor_apply.cuh"

#define BLOCK_SIZE 512
#define ILP 4

/**
 * Perform fused SGD on multiple buffers
 * N: number of tensors
 * tl[0] : gradients
 * tl[1] : weights
 * tl[2] : momentum buffers
 * tl[3] : fp16 weights (if appropriate)
 * wd : weight_decay (scalar)
 * momentum : momentum (scalar)
 * dampening : momentum dampening (scalar)
 * lr : learning rate (scalar)
 * nesterov : enable nesterov (bool)
 * first run : necessary for proper momentum handling & init
 * wd_after_momentum : apply weight decay _after_ momentum instead of before
 **/
template <typename T_grad, typename T_weight>
struct SGDFunctor {
  __device__ __forceinline__ void operator()(
      int chunk_size, volatile int *noop_gmem, TensorListMetadata<3> &tl,
      float wd, float momentum, float dampening, float lr, bool nesterov,
      bool first_run, bool wd_after_momentum, float scale) {
    // Early exit if we don't need to do anything
    if (*noop_gmem) return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    T_grad *grad_in = (T_grad *)tl.addresses[0][tensor_loc];
    grad_in += chunk_idx * chunk_size;

    T_weight *weight_in = (T_weight *)tl.addresses[1][tensor_loc];
    weight_in += chunk_idx * chunk_size;

    T_weight *mom_in = (T_weight *)tl.addresses[2][tensor_loc];
    mom_in += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    // Non-divergent exit condition for the __syncthreads
    float incoming_grads[ILP];
    float incoming_weights[ILP];
    float incoming_moms[ILP];
    for (int i_start = 0; i_start < n && i_start < chunk_size;
         i_start += blockDim.x * ILP) {
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        incoming_grads[ii] = 0;
        incoming_weights[ii] = 0;
        incoming_moms[ii] = 0;
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          incoming_grads[ii] = static_cast<float>(grad_in[i]) * scale;
          incoming_weights[ii] = static_cast<float>(weight_in[i]);
          incoming_moms[ii] = static_cast<float>(mom_in[i]);
        }
      }

// note for clarification to future michael:
// From a pure memory dependency perspective, there's likely no point unrolling
// the write loop, since writes just fire off once their LDGs arrive.
// Put another way, the STGs are dependent on the LDGs, but not on each other.
// There is still compute ILP benefit from unrolling the loop though.
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          // apply weight decay before momentum if necessary
          if (wd != 0.f && !wd_after_momentum)
            incoming_grads[ii] += wd * incoming_weights[ii];

          if (momentum != 0.f) {
            if (!first_run)
              incoming_moms[ii] = incoming_moms[ii] * momentum +
                                  (1.f - dampening) * incoming_grads[ii];
            else  // initialize momentums to current incoming grads
              incoming_moms[ii] = incoming_grads[ii];

            if (nesterov)
              incoming_grads[ii] += momentum * incoming_moms[ii];
            else
              incoming_grads[ii] = incoming_moms[ii];
          }

          // Apply WD after momentum if desired
          if (wd != 0.f && wd_after_momentum)
            incoming_grads[ii] += wd * incoming_weights[ii];

          // adjust the weight and write out
          weight_in[i] += (-lr * incoming_grads[ii]);

          // also write out the new momentum
          if (momentum != 0.f) mom_in[i] = incoming_moms[ii];
        }
      }
    }
  }
};

void multi_tensor_sgd_cuda(int chunk_size, at::Tensor noop_flag,
                           std::vector<std::vector<at::Tensor>> tensor_lists,
                           float wd, float momentum, float dampening, float lr,
                           bool nesterov, bool first_run,
                           bool wd_after_momentum, float scale) {
  auto num_tensors = tensor_lists.size();
  auto grad_type = tensor_lists[0][0].scalar_type();
  auto weight_type = tensor_lists[1][0].scalar_type();

  TORCH_CHECK(noop_flag.device() == tensor_lists[0][0].device(),
              "expected noop flag to be on the same device as tensors");

  // We have 3 possibilities to handle here, in terms of
  // grad_type, param_type, momentum_type
  // 1. fp16, fp16, fp16
  // 2. fp32, fp32, fp32
  // 3. fp16, fp32, fp32
  // It's easier to hardcode these possibilities than to use
  // switches etc. to handle the cross-product of cases where
  // we don't want the majority of them.

  // Case 1. fp16, fp16, fp16, No
  if (grad_type == at::ScalarType::Half &&
      weight_type == at::ScalarType::Half && num_tensors == 3) {
    multi_tensor_apply<3>(BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
                          SGDFunctor<at::Half, at::Half>(), wd, momentum,
                          dampening, lr, nesterov, first_run, wd_after_momentum,
                          scale);
  }
  // Case 2. fp32, fp32, fp32
  else if (grad_type == at::ScalarType::Float &&
           weight_type == at::ScalarType::Float && num_tensors == 3) {
    multi_tensor_apply<3>(BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
                          SGDFunctor<float, float>(), wd, momentum, dampening,
                          lr, nesterov, first_run, wd_after_momentum, scale);
  }
  // Case 3. fp16, fp32, fp32
  else if (grad_type == at::ScalarType::Half &&
           weight_type == at::ScalarType::Float && num_tensors == 3) {
    multi_tensor_apply<3>(BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
                          SGDFunctor<at::Half, float>(), wd, momentum,
                          dampening, lr, nesterov, first_run, wd_after_momentum,
                          scale);
  } else {
    AT_ERROR(
        "multi_tensor_sgd only supports some combinations of gradient & weight "
        "types. Given: ",
        "gradient: ", grad_type, ", weight: ", weight_type,
        ", num_lists: ", num_tensors);
  }

  AT_CUDA_CHECK(cudaGetLastError());
}