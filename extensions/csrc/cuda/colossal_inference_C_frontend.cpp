#include <torch/extension.h>

void decode_kv_cache_memcpy(
    torch::Tensor& key,        // [num_tokens, num_heads, head_size]
    torch::Tensor& value,      // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,  // [num_blocks, num_heads, block_size, head_size]
    torch::Tensor&
        value_cache,  // [num_blocks, num_heads, block_size, head_size]
    torch::Tensor& sequence_lengths,  // [batch_size]
    torch::Tensor& block_tables);     // [batch_size, max_seq_len]

void rotary_embedding(
    torch::Tensor& query,   // [total_tokens, head_num, head_dim]
    torch::Tensor& key,     // [total_tokens, kv_head_num, head_dim]      
    torch::Tensor& cos,     // [total_tokens, head_dim]
    torch::Tensor& sin);    // [total_tokens, head_dim]

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode_kv_cache_memcpy", &decode_kv_cache_memcpy,
        "Copy the GPU memory of kvcache during the decode stage.");
  m.def("rotary_embedding", &rotary_embedding,
        "performing Rotary Embedding-related calculations.");
}
