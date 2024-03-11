#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <stdio.h>

#include "../common/micros.h"

template<typename scalar_t>
__global__ void decode_kv_cache_memcpy_kernel(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ key_cache,
    scalar_t* __restrict__ value_cache,
    const int* __restrict__ sequence_lengths,
    const int* __restrict__ block_tables,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int key_stride,
    const int value_stride,
    const int block_table_stride
)
{
    const int seq_id = blockIdx.x;
    const int seq_len = sequence_lengths[seq_id] - 1;
    const int seq_id_in_block_table = seq_len / block_size;
    const int block_offset = seq_len % block_size;
    const int block_id = block_tables[seq_id * block_table_stride + seq_id_in_block_table];
    const int hidden_size = num_heads * head_size;

    if ( block_id < 0 ) {
        return ;
    }

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const int head_id = i / head_size;
        const int head_offset = i % head_size;
        const int key_src_id = seq_id * key_stride + i;
        const int value_src_id = seq_id * value_stride + i;
        const int target_src_id = block_id * hidden_size * block_size
                                      + head_id * block_size * head_size
                                      + block_offset * head_size + head_offset;

        key_cache[target_src_id] = key[key_src_id];
        value_cache[target_src_id] = value[value_src_id];
    }

}

void decode_kv_cache_memcpy(
    torch::Tensor& key,                 // [num_tokens, num_heads, head_size]
    torch::Tensor& value,               // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,           // [num_blocks, num_heads, block_size, head_size]
    torch::Tensor& value_cache,         // [num_blocks, num_heads, block_size, head_size]
    torch::Tensor& sequence_lengths,    // [batch_size]
    torch::Tensor& block_tables)       // [batch_size, max_seq_len]
{
    int num_tokens = key.size(0);
    int num_heads = key.size(1);
    int head_size = key.size(2);
    int block_size = key_cache.size(2);

    int key_stride = key.stride(0);
    int value_stride = value.stride(0);
    int block_table_stride = block_tables.stride(0);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(num_tokens);
    dim3 block(std::min(num_heads * head_size, 512));
    DISPATCH_FLOAT_HALF_AND_BFLOAT(
        key.scalar_type(),
        "decode_kv_cache_memcpy",
        decode_kv_cache_memcpy_kernel<scalar_t><<<grid, block, 0, stream>>>(
            key.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(),
            key_cache.data_ptr<scalar_t>(),
            value_cache.data_ptr<scalar_t>(),
            sequence_lengths.data_ptr<int>(),
            block_tables.data_ptr<int>(),
            num_heads,
            head_size,
            block_size,
            key_stride,
            value_stride,
            block_table_stride
        );)

    AT_CUDA_CHECK(cudaGetLastError());

}
