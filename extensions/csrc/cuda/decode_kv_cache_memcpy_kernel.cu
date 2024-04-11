#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "utils/vec_copy.h"
#include "../common/micros.h"

template<typename scalar_t, bool Aligned, int VecSize>
__global__ void decode_kv_cache_memcpy_kernel(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ key_cache,
    scalar_t* __restrict__ value_cache,
    const int* __restrict__ sequence_lengths,
    const int* __restrict__ block_tables,
    const int head_num,
    const int head_dim,
    const int block_size,
    const int64_t key_stride,
    const int64_t value_stride,
    const int block_table_stride
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
        const int64_t key_src_id = seq_id * key_stride + i;
        const int64_t value_src_id = seq_id * value_stride + i;
        const int64_t target_id = block_id * hidden_size * block_size
                                      + head_id * block_size * head_dim
                                      + block_offset * head_dim + head_offset;

        copy_vector<scalar_t, VecSize>(key_cache + target_id, key + key_src_id);
        copy_vector<scalar_t, VecSize>(value_cache + target_id, value + value_src_id);
    }

    if (!Aligned) {
        for (; i < hidden_size; ++i ) {
            const int head_id = i / head_dim;
            const int head_offset = i % head_dim;
            const int64_t key_src_id = seq_id * key_stride + i;
            const int64_t value_src_id = seq_id * value_stride + i;
            const int64_t target_id = block_id * hidden_size * block_size
                                        + head_id * block_size * head_dim
                                        + block_offset * head_dim + head_offset;

            key_cache[target_id] = key[key_src_id];
            value_cache[target_id] = value[value_src_id];
        }
    }

}

template<typename scalar_t>
void apply_decode_kv_cache_memcpy(
    at::Tensor& key,                 // [num_tokens, head_num, head_dim]
    at::Tensor& value,               // [num_tokens, head_num, head_dim]
    at::Tensor& key_cache,           // [num_blocks, head_num, block_size, head_dim]
    at::Tensor& value_cache,         // [num_blocks, head_num, block_size, head_dim]
    at::Tensor& sequence_lengths,    // [batch_size]
    at::Tensor& block_tables)        // [batch_size, max_seq_len]
{
    int num_tokens = key.size(0);
    int head_num = key.size(1);
    int head_dim = key.size(2);
    int block_size = key_cache.size(2);

    int64_t key_stride = key.stride(0);
    int64_t value_stride = value.stride(0);
    int block_table_stride = block_tables.stride(0);

    int vec_size = get_vec_size<scalar_t>(key);

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
        decode_kv_cache_memcpy_kernel<scalar_t, __aligned, __vec_size><<<grid, block, 0, stream>>>(     \
                key.data_ptr<scalar_t>(),                                                               \
                value.data_ptr<scalar_t>(),                                                             \
                key_cache.data_ptr<scalar_t>(),                                                         \
                value_cache.data_ptr<scalar_t>(),                                                       \
                sequence_lengths.data_ptr<int>(),                                                       \
                block_tables.data_ptr<int>(),                                                           \
                head_num,                                                                               \
                head_dim,                                                                               \
                block_size,                                                                             \
                key_stride,                                                                             \
                value_stride,                                                                           \
                block_table_stride                                                                      \
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
    at::Tensor& key_cache,           // [num_blocks, head_num, block_size, head_dim]
    at::Tensor& value_cache,         // [num_blocks, head_num, block_size, head_dim]
    at::Tensor& sequence_lengths,    // [batch_size]
    at::Tensor& block_tables)        // [batch_size, max_seq_len]
{
    DISPATCH_FLOAT_HALF_AND_BFLOAT(
        key.scalar_type(),
        "decode_kv_cache_memcpy",
        apply_decode_kv_cache_memcpy<scalar_t>(
            key,
            value,
            key_cache,
            value_cache,
            sequence_lengths,
            block_tables
        );)
}
