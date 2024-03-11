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

void rms_layernorm(torch::Tensor& out,     // [..., hidden_size]
                   torch::Tensor& input,   // [..., hidden_size]
                   torch::Tensor& weight,  // [hidden_size]
                   float epsilon);

void fused_add_rms_layernorm(torch::Tensor& input,     // [..., hidden_size]
                             torch::Tensor& residual,  // [..., hidden_size]
                             torch::Tensor& weight,    // [hidden_size]
                             float epsilon);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode_kv_cache_memcpy", &decode_kv_cache_memcpy,
        "Copy the GPU memory of kvcache during the decode stage.");

  m.def("silu_and_mul", &silu_and_mul, "Silu with a following multiply");

  m.def("rms_layernorm", &rms_layernorm,
        "Apply Root Mean Square (RMS) Normalization to the input tensor.");

  m.def("fused_add_rms_layernorm", &fused_add_rms_layernorm,
        "In-place fused Add and RMS Normalization.");
}
