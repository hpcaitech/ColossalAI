#include <torch/extension.h>

void decode_kv_cache_memcpy(
    torch::Tensor& key,    // [num_tokens, num_heads, head_size]
    torch::Tensor& value,  // [num_tokens, num_heads, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, head_num, head_dim/x, block_size, x]
    torch::Tensor&
        value_cache,  // [num_blocks, num_heads, block_size, head_size]
    torch::Tensor& sequence_lengths,  // [batch_size]
    torch::Tensor& block_tables);     // [batch_size, max_seq_len]

void context_kv_cache_memcpy(
    at::Tensor& key,        // [num_tokens, head_num, head_dim]
    at::Tensor& value,      // [num_tokens, head_num, head_dim]
    at::Tensor& key_cache,  // [num_blocks, head_num, head_dim/x, block_size, x]
    at::Tensor& value_cache,  // [num_blocks, head_num, block_size, head_dim]
    at::Tensor& sequence_lengths,  // [batch_size]
    at::Tensor& cu_seqlens,        // [batch_size + 1]
    at::Tensor& block_tables,      // [batch_size, max_seq_len]
    int max_seq_len_in_batch);

void rotary_embedding(
    torch::Tensor& query,  // [total_tokens, head_num, head_dim]
    torch::Tensor& key,    // [total_tokens, kv_head_num, head_dim]
    torch::Tensor& cos,    // [total_tokens, head_dim]
    torch::Tensor& sin,    // [total_tokens, head_dim]
    bool high_precision);

void rotary_embedding_and_cache_copy(
    torch::Tensor& query,  // [num_tokens, head_num, head_dim]
    torch::Tensor& key,    // [num_tokens, kv_head_num, head_dim]
    torch::Tensor& value,  // [num_tokens, num_heads, head_dim]
    torch::Tensor& cos,    // [num_tokens, head_dim]
    torch::Tensor& sin,    // [num_tokens, head_dim]
    torch::Tensor&
        key_cache,  // [num_blocks, head_num, head_dim/x, block_size, x]
    torch::Tensor&
        value_cache,  // [num_blocks, num_heads, block_size, head_dim]
    torch::Tensor& sequence_lengths,  // [batch_size]
    torch::Tensor& block_tables,      // [batch_size, max_seq_len]
    bool high_precision);

torch::Tensor silu_and_mul(const torch::Tensor& ins);

void rms_layernorm(torch::Tensor& out,     // [..., hidden_size]
                   torch::Tensor& input,   // [..., hidden_size]
                   torch::Tensor& weight,  // [hidden_size]
                   float epsilon);

void fused_add_rms_layernorm(torch::Tensor& input,     // [..., hidden_size]
                             torch::Tensor& residual,  // [..., hidden_size]
                             torch::Tensor& weight,    // [hidden_size]
                             float epsilon);

void get_cos_and_sin(at::Tensor& cos_cache,  // [max_rotary_position, head_dim]
                     at::Tensor& sin_cache,  // [max_rotary_position, head_dim]
                     at::Tensor& cos,        // [num_tokens, head_dim]
                     at::Tensor& sin,        // [num_tokens, head_dim]
                     at::Tensor& sequence_lengths,  // [batch_size]
                     int max_seq_len_in_batch, bool is_prompts);

void flash_decoding_attention(
    torch::Tensor& out,    // [num_tokens, num_heads, head_size]
    torch::Tensor& query,  // [num_tokens, num_heads, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    torch::Tensor&
        value_cache,  // [num_blocks, num_kv_heads, block_size, head_size]
    torch::Tensor& context_lens,  // [num_tokens]
    torch::Tensor& block_tables,  // [num_tokens, max_num_blocks_per_seq]
    int block_size, int max_context_len,
    torch::Tensor&
        tmp_out,  // [num_tokens, num_heads, max_num_partitions, head_size]
    torch::Tensor& exp_sums,    // [num_tokens, num_heads, max_num_partitions]
    torch::Tensor& max_logits,  // [num_tokens, num_heads, max_num_partitions]
    const c10::optional<torch::Tensor>& alibi_slopes, float scale);

void convert_fp8(torch::Tensor& input, torch::Tensor& output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode_kv_cache_memcpy", &decode_kv_cache_memcpy,
        "Copy the GPU memory of kvcache during the decode stage.");

  m.def("context_kv_cache_memcpy", &context_kv_cache_memcpy,
        "Copy the GPU memory of kvcache during the context stage.");

  m.def(
      "rotary_embedding_and_cache_copy", &rotary_embedding_and_cache_copy,
      "Performing Rotary Embedding-related calculations and KVCache Memcopy.");

  m.def("rotary_embedding", &rotary_embedding,
        "Performing Rotary Embedding-related calculations.");

  m.def("silu_and_mul", &silu_and_mul, "Silu with a following multiply");

  m.def("rms_layernorm", &rms_layernorm,
        "Apply Root Mean Square (RMS) Normalization to the input tensor.");

  m.def("fused_add_rms_layernorm", &fused_add_rms_layernorm,
        "In-place fused Add and RMS Normalization.");

  m.def("get_cos_and_sin", &get_cos_and_sin, "Get cos and sin from the cache.");

  m.def("flash_decoding_attention", &flash_decoding_attention,
        "Compute the attention between an input query and the cached "
        "keys/values using PagedAttention.");

  m.def("convert_fp8", &convert_fp8,
        "Convert input to fp8 output or convert fp8 input to output.");
}
