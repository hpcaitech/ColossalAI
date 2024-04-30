#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "utils/vec_copy.h"
#include "common/micros.h"

using colossalAI::cuda::utils::get_vec_size;
using colossalAI::cuda::utils::copy;
using colossalAI::funcs::CastFunctor;


template<typename T, typename CacheT, bool Aligned, int VecSize>
__global__ void context_kv_cache_memcpy_kernel(
    const T* __restrict__ key,
    const T* __restrict__ value,
    CacheT* __restrict__ key_cache,
    CacheT* __restrict__ value_cache,
    const int* __restrict__ sequence_lengths,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ block_tables,
    const int head_num,
    const int head_dim,
    const int block_size,
    const int batch_size,
    const int block_table_stride,
    const int64_t key_stride,
    const int64_t value_stride,
    const int x
)
{
    const int seq_token_id = blockIdx.x;
    const int seq_id = blockIdx.y;
    const int block_id = block_tables[seq_id * block_table_stride + seq_token_id / block_size];

    if (block_id < 0 || seq_token_id > sequence_lengths[seq_id] - 1) {
        return ;
    }

    const int block_offset = seq_token_id % block_size;
    const int hidden_size = head_num * head_dim;
    const int total_token_id = cu_seqlens[seq_id] + seq_token_id;
    int head_id;
    int head_offset;
    int x_id;
    int x_offset;
    int64_t key_src_id;
    int64_t value_src_id;
    int64_t target_key_id;
    int64_t target_value_id;

    int i = threadIdx.x * VecSize;

    for (; i <= (hidden_size - VecSize); i += blockDim.x * VecSize) {
        head_id = i / head_dim;
        head_offset = i % head_dim;
        x_id = head_offset / x;
        x_offset = head_offset % x;
        key_src_id = total_token_id * key_stride + i;
        value_src_id = total_token_id * value_stride + i;
        target_key_id = block_id * hidden_size * block_size
                                      + head_id * block_size * head_dim
                                      + x_id * block_size * x
                                      + block_offset * x
                                      + x_offset;
        target_value_id = block_id * hidden_size * block_size
                                      + head_id * block_size * head_dim
                                      + block_offset * head_dim + head_offset;

        copy<T, CacheT, VecSize>(key + key_src_id, key_cache + target_key_id);
        copy<T, CacheT, VecSize>(value + value_src_id, value_cache + target_value_id);
    }

    // tail process
    if (!Aligned) {
        for (; i < hidden_size; ++i ) {
            head_id = i / head_dim;
            head_offset = i % head_dim;
            x_id = head_offset / x;
            x_offset = head_offset % x;
            key_src_id = total_token_id * key_stride + i;
            value_src_id = total_token_id * value_stride + i;
            target_key_id = block_id * hidden_size * block_size
                                        + head_id * block_size * head_dim
                                        + x_id * block_size * x
                                        + block_offset * x
                                        + x_offset;
            target_value_id = block_id * hidden_size * block_size
                                        + head_id * block_size * head_dim
                                        + block_offset * head_dim + head_offset;

            key_cache[target_key_id] =  CastFunctor<T, CacheT>()(key[key_src_id]);
            value_cache[target_value_id] = CastFunctor<T, CacheT>()(value[value_src_id]);
        }
    }

}

template<typename T, typename CacheT>
void apply_context_kv_cache_memcpy(
    torch::Tensor& key,                 // [num_tokens, head_num, head_dim]
    torch::Tensor& value,               // [num_tokens, head_num, head_dim]
    torch::Tensor& key_cache,           // [num_blocks, head_num, head_dim/x, block_size, x]
    torch::Tensor& value_cache,         // [num_blocks, head_num, block_size, head_dim]
    torch::Tensor& sequence_lengths,    // [batch_size]
    torch::Tensor& cu_seqlens,          // [batch_size + 1]
    torch::Tensor& block_tables,        // [batch_size, max_seq_len]
    int max_seq_len_in_batch)
{
    int num_tokens = key.size(0);
    int head_num = key.size(1);
    int head_dim = key.size(2);
    int block_size = key_cache.size(3);
    int x = key_cache.size(4);
    int batch_size = block_tables.size(0);

    int64_t key_stride = key.stride(0);
    int64_t value_stride = value.stride(0);
    int block_table_stride = block_tables.stride(0);

    int vec_size = get_vec_size<T>(key);

    bool aligned = true;
    if (head_dim % vec_size != 0) {
        aligned = false;
    }

    int thread_nums = head_num * head_dim / vec_size;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(max_seq_len_in_batch, batch_size);
    dim3 block(std::min(thread_nums, 512));

#define CONTEXT_KV_CACHE_MEMCOPY_KERNEL_LAUNCH(__aligned, __vec_size)                                   \
    do {                                                                                                \
        context_kv_cache_memcpy_kernel<T, CacheT, __aligned, __vec_size><<<grid, block, 0, stream>>>(    \
                reinterpret_cast<T*>(key.data_ptr()),                                                   \
                reinterpret_cast<T*>(value.data_ptr()),                                                 \
                reinterpret_cast<CacheT*>(key_cache.data_ptr()),                                        \
                reinterpret_cast<CacheT*>(value_cache.data_ptr()),                                      \
                sequence_lengths.data_ptr<int>(),                                                       \
                cu_seqlens.data_ptr<int>(),                                                             \
                block_tables.data_ptr<int>(),                                                           \
                head_num,                                                                               \
                head_dim,                                                                               \
                block_size,                                                                             \
                batch_size,                                                                             \
                block_table_stride,                                                                     \
                key_stride,                                                                             \
                value_stride,                                                                           \
                x                                                                                       \
            );                                                                                          \
    } while(0)

#define CONTEXT_KV_CACHE_MEMCOPY_KERNEL_LAUNCH_VEC_SIZE_CASE(__aligned)                                 \
    do {                                                                                                \
        switch (vec_size) {                                                                             \
            case 1:                                                                                     \
                CONTEXT_KV_CACHE_MEMCOPY_KERNEL_LAUNCH(__aligned, 1);                                   \
                break;                                                                                  \
            case 2:                                                                                     \
                CONTEXT_KV_CACHE_MEMCOPY_KERNEL_LAUNCH(__aligned, 2);                                   \
                break;                                                                                  \
            case 4:                                                                                     \
                CONTEXT_KV_CACHE_MEMCOPY_KERNEL_LAUNCH(__aligned, 4);                                   \
                break;                                                                                  \
            default:                                                                                    \
                AT_ERROR("Unsupported vectorized size ", vec_size);                                     \
                break;                                                                                  \
        }                                                                                               \
    } while(0)


    if (aligned) {
        CONTEXT_KV_CACHE_MEMCOPY_KERNEL_LAUNCH_VEC_SIZE_CASE(true);
    }
    else {
        CONTEXT_KV_CACHE_MEMCOPY_KERNEL_LAUNCH_VEC_SIZE_CASE(false);
    }

    AT_CUDA_CHECK(cudaGetLastError());

}

void context_kv_cache_memcpy(
    torch::Tensor& key,                 // [num_tokens, head_num, head_dim]
    torch::Tensor& value,               // [num_tokens, head_num, head_dim]
    torch::Tensor& key_cache,           // [num_blocks, head_num, head_dim/x, block_size, x]
    torch::Tensor& value_cache,         // [num_blocks, head_num, block_size, head_dim]
    torch::Tensor& sequence_lengths,    // [batch_size]
    torch::Tensor& cu_seqlens,          // [batch_size + 1]
    torch::Tensor& block_tables,        // [batch_size, max_seq_len]
    int max_seq_len_in_batch)
{

#define _(T, CacheT)                            \
    apply_context_kv_cache_memcpy<T, CacheT>(   \
        key,                                    \
        value,                                  \
        key_cache,                              \
        value_cache,                            \
        sequence_lengths,                       \
        cu_seqlens,                             \
        block_tables,                           \
        max_seq_len_in_batch                    \
    )

    if(key_cache.scalar_type() == at::ScalarType::Byte)
    {
        switch (key.scalar_type())
        {
            case at::ScalarType::Float:
                _(float, uint8_t);
                break;
            case at::ScalarType::Half:
                _(half, uint8_t);
                break;
            case at::ScalarType::BFloat16:
                _(__nv_bfloat16, uint8_t);
                break;
        }
    }
    else
    {
        switch (key.scalar_type())
        {
            case at::ScalarType::Float:
                _(float, float);
                break;
            case at::ScalarType::Half:
                _(half, half);
                break;
            case at::ScalarType::BFloat16:
                _(__nv_bfloat16, __nv_bfloat16);
                break;
        }
    }
#undef _
}
