#include <torch/extension.h>

void decode_kv_cache_memcpy(
    torch::Tensor& key,        // [num_tokens, num_heads, head_size]
    torch::Tensor& value,      // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,  // [num_blocks, num_heads, block_size, head_size]
    torch::Tensor&
        value_cache,  // [num_blocks, num_heads, block_size, head_size]
    torch::Tensor& sequence_lengths,  // [batch_size]
    torch::Tensor& block_tables);     // [batch_size, max_seq_len]

torch::Tensor silu_and_mul(const torch::Tensor& ins);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode_kv_cache_memcpy", &decode_kv_cache_memcpy,
        "Copy the GPU memory of kvcache during the decode stage.");
  m.def("silu_and_mul", &silu_and_mul, "Silu with a following multiply");
}
