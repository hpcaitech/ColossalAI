/*This code from NVIDIA Megatron:
 *     with minor changes. */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "scaled_upper_triang_masked_softmax.h"
#include "type_shim.h"

namespace multihead_attn {
namespace fused_softmax {
namespace scaled_upper_triang_masked_softmax {

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
}  // namespace scaled_upper_triang_masked_softmax
}  // namespace fused_softmax
}  // namespace multihead_attn
