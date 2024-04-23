/*This code from NVIDIA Megatron:
 *     with minor changes. */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "scaled_masked_softmax.h"
#include "type_shim.h"

namespace multihead_attn {
namespace fused_softmax {
namespace scaled_masked_softmax {

int get_batch_per_block_cuda(int query_seq_len, int key_seq_len, int batches,
                             int attn_heads) {
  return get_batch_per_block(query_seq_len, key_seq_len, batches, attn_heads);
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
}  // namespace scaled_masked_softmax
}  // namespace fused_softmax
}  // namespace multihead_attn
