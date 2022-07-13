/*This code from NVIDIA Megatron:
 *     with minor changes. */

#include <cuda_fp16.h>
#include <torch/extension.h>
#include <vector>

namespace multihead_attn {
namespace fused_softmax {
namespace scaled_masked_softmax {

torch::Tensor fwd_cuda(
    torch::Tensor const& input, 
    torch::Tensor const& mask,
    float scale_factor);

torch::Tensor bwd_cuda(
    torch::Tensor const& output_grads, 
    torch::Tensor const& softmax_results,
    float scale_factor);

int get_batch_per_block_cuda(
    int query_seq_len,
    int key_seq_len,
    int batches,
    int attn_heads);

torch::Tensor fwd(
    torch::Tensor const& input,
    torch::Tensor const& mask,
    float scale_factor) {
  AT_ASSERTM(input.dim() == 4, "expected 4D tensor");
  AT_ASSERTM((input.scalar_type() == at::ScalarType::Half) ||
	     (input.scalar_type() == at::ScalarType::BFloat16), 
      "Only fp16 and bf16 are supported");
  AT_ASSERTM(mask.dim() == 4, "expected 4D tensor");

  return fwd_cuda(input, mask, scale_factor);
}

torch::Tensor bwd(
    torch::Tensor const& output_grads, 
    torch::Tensor const& softmax_results,
    float scale_factor) {

  AT_ASSERTM(output_grads.dim() == 4, "expected 3D tensor");
  AT_ASSERTM(softmax_results.dim() == 4, "expected 3D tensor");

  AT_ASSERTM((output_grads.scalar_type() == at::ScalarType::Half) ||
	     (output_grads.scalar_type() == at::ScalarType::BFloat16), 
      "Only fp16 and bf16 are supported");
  AT_ASSERTM((softmax_results.scalar_type() == at::ScalarType::Half) ||
	     (softmax_results.scalar_type() == at::ScalarType::BFloat16), 
      "Only fp16 and bf16 are supported");

  return bwd_cuda(output_grads, softmax_results, scale_factor);
}

int get_batch_per_block(
    int query_seq_len,
    int key_seq_len,
    int batches,
    int attn_heads) {
    return get_batch_per_block_cuda(query_seq_len, key_seq_len, batches, attn_heads);
}

} // end namespace scaled_masked_softmax
} // end namespace fused_softmax
} // end namespace multihead_attn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", 
        &multihead_attn::fused_softmax::scaled_masked_softmax::fwd, 
	"Self Multihead Attention scaled, time masked softmax -- Forward.");

  m.def("backward",
        &multihead_attn::fused_softmax::scaled_masked_softmax::bwd,
	"Self Multihead Attention scaled, time masked softmax -- Backward.");

  m.def("get_batch_per_block",
        &multihead_attn::fused_softmax::scaled_masked_softmax::get_batch_per_block,
        "Return Batch per block size."
  );
}
