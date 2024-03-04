#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <stdio.h>

#include "type_shim.h"

#include "scaled_upper_triang_masked_softmax.h"

template<typename scalar_t>
__global__ void rotary_embedding_and_cache_copy_kernel(
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    scalar_t* __restrict__ value,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    scalar_t* __restrict__ key_cache,
    scalar_t* __restrict__ value_cache,
    const int* __restrict__ sequence_lengths,
    const int* __restrict__ block_tables,
    const int64_t query_stride,
    const int64_t key_stride,
    const int64_t value_stride,
    const int cos_stride,
    const int sin_stride,
    const int block_table_stride,
    const int head_num,
    const int head_dim,
    const int kv_head_num,
    const int block_size
) {

    const int token_id = blockIdx.x;
    const int half_head_dim = head_dim / 2;
    int half_hidden_size = head_num * half_head_dim;

    //compute query
    {
        for (int i = threadIdx.x; i < half_hidden_size; i += blockDim.x) {
            const int head_id = i / half_head_dim;
            const int head_offset = i % half_head_dim;

            scalar_t cos_value = cos[token_id * cos_stride + head_offset];
            scalar_t sin_value = sin[token_id * sin_stride + head_offset];

            const int64_t addr_head = token_id * query_stride + head_id * head_dim;
            int64_t x_offest = addr_head + head_offset;
            int64_t y_offest = addr_head + half_head_dim + head_offset;

            scalar_t x = query[x_offest];
            scalar_t y = query[y_offest];

            query[x_offest] = x * cos_value - y * sin_value;
            query[y_offest] = y * cos_value + x * sin_value;
        }
    }

    //compute key
    {
        const int seq_len = sequence_lengths[token_id] - 1;
        const int seq_id_in_block_table = seq_len / block_size;
        const int block_offset = seq_len % block_size;
        const int block_id = block_tables[token_id * block_table_stride + seq_id_in_block_table];

        if ( block_id < 0 ) {
            return ;
        }

        const int hidden_size = head_num * head_dim;
        half_hidden_size = kv_head_num * half_head_dim;
        for (int i = threadIdx.x; i < half_hidden_size; i += blockDim.x) {
            const int head_id = i / half_head_dim;
            const int head_offset = i % half_head_dim;
            const int64_t target_src_id = block_id * hidden_size * block_size
                                        + head_id * block_size * head_dim
                                        + block_offset * head_dim + head_offset;

            scalar_t cos_value = cos[token_id * cos_stride + head_offset];
            scalar_t sin_value = sin[token_id * sin_stride + head_offset];

            const int64_t addr_head = token_id * key_stride + head_id * head_dim;

            scalar_t x = key[addr_head + head_offset];
            scalar_t y = key[addr_head + half_head_dim + head_offset];

            key_cache[target_src_id] = x * cos_value - y * sin_value;
            key_cache[target_src_id + half_head_dim] = y * cos_value + x * sin_value;
        }
        for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
            const int head_id = i / head_dim;
            const int head_offset = i % head_dim;
            const int64_t value_src_id = token_id * value_stride + i;
            const int64_t target_src_id = block_id * hidden_size * block_size
                                        + head_id * block_size * head_dim
                                        + block_offset * head_dim + head_offset;
            value_cache[target_src_id] = value[value_src_id];
        }
    }
}

void rotary_embedding_and_cache_copy(
    torch::Tensor& query,               // [num_tokens, head_num, head_dim]
    torch::Tensor& key,                 // [num_tokens, kv_head_num, head_dim]
    torch::Tensor& value,               // [num_tokens, head_num, head_dim]
    torch::Tensor& cos,                 // [num_tokens, head_dim]
    torch::Tensor& sin,                 // [num_tokens, head_dim]
    torch::Tensor& key_cache,           // [num_blocks, head_num, block_size, head_dim]
    torch::Tensor& value_cache,         // [num_blocks, head_num, block_size, head_dim]
    torch::Tensor& sequence_lengths,    // [batch_size]
    torch::Tensor& block_tables)        // [batch_size, max_seq_len]
{
    int num_tokens = query.size(0);
    int head_num = query.size(1);
    int head_dim = query.size(2);
    int kv_head_num = key.size(1);
    int block_size = key_cache.size(2);

    int64_t query_stride = query.stride(0);
    int64_t key_stride = key.stride(0);
    int64_t value_stride = value.stride(0);
    int cos_stride = cos.stride(0);
    int sin_stride = sin.stride(0);
    int block_table_stride = block_tables.stride(0);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(num_tokens);
    dim3 block(std::min(head_num * head_dim, 512));
    DISPATCH_FLOAT_HALF_AND_BFLOAT(
        query.scalar_type(),
        "rotary_embedding_and_cache_copy",
        rotary_embedding_and_cache_copy_kernel<scalar_t><<<grid, block, 0, stream>>>(
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(),
            cos.data_ptr<scalar_t>(),
            sin.data_ptr<scalar_t>(),
            key_cache.data_ptr<scalar_t>(),
            value_cache.data_ptr<scalar_t>(),
            sequence_lengths.data_ptr<int>(),
            block_tables.data_ptr<int>(),
            query_stride,
            key_stride,
            value_stride,
            cos_stride,
            sin_stride,
            block_table_stride,
            head_num,
            head_dim,
            kv_head_num,
            block_size
        );)

    AT_CUDA_CHECK(cudaGetLastError());

}
