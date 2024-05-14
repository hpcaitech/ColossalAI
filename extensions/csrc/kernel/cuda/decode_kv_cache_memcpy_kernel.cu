#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "utils/vec_copy.h"
#include "funcs/cast_functor.h"
#include "common/micros.h"

using colossalAI::cuda::utils::get_vec_size;
using colossalAI::cuda::utils::copy;
using colossalAI::funcs::CastFunctor;


template<typename T, typename CacheT, bool Aligned, int VecSize>
__global__ void decode_kv_cache_memcpy_kernel(
    const T* __restrict__ key,
    const T* __restrict__ value,
    CacheT* __restrict__ key_cache,
    CacheT* __restrict__ value_cache,
    const int* __restrict__ sequence_lengths,
    const int* __restrict__ block_tables,
    const int head_num,
    const int head_dim,
    const int block_size,
    const int64_t key_stride,
    const int64_t value_stride,
    const int block_table_stride,
    const int x
)
{
    const int seq_id = blockIdx.x;
    const int seq_len = sequence_lengths[seq_id] - 1;
    const int block_offset = seq_len % block_size;
    const int block_id = block_tables[seq_id * block_table_stride + seq_len / block_size];
    const int hidden_size = head_num * head_dim;

    if ( block_id < 0 ) {
        return ;
    }

    int i = threadIdx.x * VecSize;

    for (; i <= (hidden_size - VecSize); i += blockDim.x * VecSize) {
        const int head_id = i / head_dim;
        const int head_offset = i % head_dim;
        const int x_id = head_offset / x;
        const int x_offset = head_offset % x;
        const int64_t key_src_id = seq_id * key_stride + i;
        const int64_t value_src_id = seq_id * value_stride + i;
        const int64_t target_key_id = block_id * hidden_size * block_size
                                      + head_id * block_size * head_dim
                                      + x_id * block_size * x
                                      + block_offset * x
                                      + x_offset;
        const int64_t target_value_id = block_id * hidden_size * block_size
                                      + head_id * block_size * head_dim
                                      + block_offset * head_dim + head_offset;

        copy<T, CacheT, VecSize>(key + key_src_id, key_cache + target_key_id);
        copy<T, CacheT, VecSize>(value + value_src_id, value_cache + target_value_id);
    }

    if (!Aligned) {
        for (; i < hidden_size; ++i ) {
            const int head_id = i / head_dim;
            const int head_offset = i % head_dim;
            const int x_id = head_offset / x;
            const int x_offset = head_offset % x;
            const int64_t key_src_id = seq_id * key_stride + i;
            const int64_t value_src_id = seq_id * value_stride + i;
            const int64_t target_key_id = block_id * hidden_size * block_size
                                        + head_id * block_size * head_dim
                                        + x_id * block_size * x
                                        + block_offset * x
                                        + x_offset;
            const int64_t target_value_id = block_id * hidden_size * block_size
                                        + head_id * block_size * head_dim
                                        + block_offset * head_dim + head_offset;

            key_cache[target_key_id] = CastFunctor<T, CacheT>()(key[key_src_id]);
            value_cache[target_value_id] = CastFunctor<T, CacheT>()(value[value_src_id]);
        }
    }

}

template<typename T, typename CacheT>
void apply_decode_kv_cache_memcpy(
    at::Tensor& key,                 // [num_tokens, head_num, head_dim]
    at::Tensor& value,               // [num_tokens, head_num, head_dim]
    at::Tensor& key_cache,           // [num_blocks, head_num, head_dim/x, block_size, x]
    at::Tensor& value_cache,         // [num_blocks, head_num, block_size, head_dim]
    at::Tensor& sequence_lengths,    // [batch_size]
    at::Tensor& block_tables)        // [batch_size, max_seq_len]
{
    int num_tokens = key.size(0);
    int head_num = key.size(1);
    int head_dim = key.size(2);
    int block_size = key_cache.size(3);
    int x = key_cache.size(4);

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

    dim3 grid(num_tokens);
    dim3 block(std::min(thread_nums, 512));

#define DECODE_KV_CACHE_MEMCOPY_KERNEL_LAUNCH(__aligned, __vec_size)                                    \
    do {                                                                                                \
        decode_kv_cache_memcpy_kernel<T, CacheT, __aligned, __vec_size><<<grid, block, 0, stream>>>(    \
                reinterpret_cast<T*>(key.data_ptr()),                                                   \
                reinterpret_cast<T*>(value.data_ptr()),                                                 \
                reinterpret_cast<CacheT*>(key_cache.data_ptr()),                                        \
                reinterpret_cast<CacheT*>(value_cache.data_ptr()),                                      \
                sequence_lengths.data_ptr<int>(),                                                       \
                block_tables.data_ptr<int>(),                                                           \
                head_num,                                                                               \
                head_dim,                                                                               \
                block_size,                                                                             \
                key_stride,                                                                             \
                value_stride,                                                                           \
                block_table_stride,                                                                     \
                x                                                                                       \
            );                                                                                          \
    } while(0)

#define DECODE_KV_CACHE_MEMCOPY_KERNEL_LAUNCH_VEC_SIZE_CASE(__aligned, __vec_size)                      \
    do {                                                                                                \
        switch (__vec_size) {                                                                           \
            case 1:                                                                                     \
                DECODE_KV_CACHE_MEMCOPY_KERNEL_LAUNCH(__aligned, 1);                                    \
                break;                                                                                  \
            case 2:                                                                                     \
                DECODE_KV_CACHE_MEMCOPY_KERNEL_LAUNCH(__aligned, 2);                                    \
                break;                                                                                  \
            case 4:                                                                                     \
                DECODE_KV_CACHE_MEMCOPY_KERNEL_LAUNCH(__aligned, 4);                                    \
                break;                                                                                  \
            default:                                                                                    \
                AT_ERROR("Unsupported vectorized size ", __vec_size);                                   \
                break;                                                                                  \
        }                                                                                               \
    } while(0)

    if (aligned) {
        DECODE_KV_CACHE_MEMCOPY_KERNEL_LAUNCH_VEC_SIZE_CASE(true, vec_size);
    }
    else {
        DECODE_KV_CACHE_MEMCOPY_KERNEL_LAUNCH_VEC_SIZE_CASE(false, vec_size);
    }

    AT_CUDA_CHECK(cudaGetLastError());

}

void decode_kv_cache_memcpy(
    at::Tensor& key,                 // [num_tokens, head_num, head_dim]
    at::Tensor& value,               // [num_tokens, head_num, head_dim]
    at::Tensor& key_cache,           // [num_blocks, head_num, head_dim/x, block_size, x]
    at::Tensor& value_cache,         // [num_blocks, head_num, block_size, head_dim]
    at::Tensor& sequence_lengths,    // [batch_size]
    at::Tensor& block_tables)        // [batch_size, max_seq_len]
{

#define _(T, CacheT)                            \
    apply_decode_kv_cache_memcpy<T, CacheT>(    \
        key,                                    \
        value,                                  \
        key_cache,                              \
        value_cache,                            \
        sequence_lengths,                       \
        block_tables                            \
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
