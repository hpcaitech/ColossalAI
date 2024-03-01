#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <stdio.h>

#include "type_shim.h"


template<typename scalar_t>
__global__ void rotary_embedding_kernel(
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    const int64_t query_stride,
    const int64_t key_stride,
    const int cos_stride,
    const int sin_stride,
    const int head_num,
    const int head_dim,
    const int kv_head_num
) {
    const int token_id = blockIdx.x;
    const int half_head_dim = head_dim / 2; 
    int half_hidden_size = head_num * half_head_dim;
    
    //compute query
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

    //compute key
    half_hidden_size = kv_head_num * half_head_dim;
    for (int i = threadIdx.x; i < half_hidden_size; i += blockDim.x) {
        const int head_id = i / half_head_dim;
        const int head_offset = i % half_head_dim;
        
        scalar_t cos_value = cos[token_id * cos_stride + head_offset];
        scalar_t sin_value = sin[token_id * sin_stride + head_offset];

        const int64_t addr_head = token_id * key_stride + head_id * head_dim;
        int64_t x_offest = addr_head + head_offset;
        int64_t y_offest = addr_head + half_head_dim + head_offset;

        scalar_t x = key[x_offest];
        scalar_t y = key[y_offest];

        key[x_offest] = x * cos_value - y * sin_value;
        key[y_offest] = y * cos_value + x * sin_value;
    }
}

void rotary_embedding(
    torch::Tensor& query,   // [total_tokens, head_num, head_dim]
    torch::Tensor& key,     // [total_tokens, kv_head_num, head_dim]      
    torch::Tensor& cos,     // [total_tokens, head_dim]
    torch::Tensor& sin      // [total_tokens, head_dim]
){
    int num_tokens = query.size(0);
    int num_heads = query.size(1);
    int head_dim = query.size(2);
    int kv_head_num = key.size(1);

    int query_stride = query.stride(0);
    int key_stride = key.stride(0);
    int cos_stride = cos.stride(0);
    int sin_stride = sin.stride(0);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(num_tokens);
    dim3 block(std::min(num_heads * head_dim, 512));
    DISPATCH_FLOAT_HALF_AND_BFLOAT(
        query.scalar_type(),
        "rotary_embedding",
        rotary_embedding_kernel<scalar_t><<<grid, block, 0, stream>>>(
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            cos.data_ptr<scalar_t>(),
            sin.data_ptr<scalar_t>(),
            query_stride,
            key_stride,
            cos_stride,
            sin_stride,
            num_heads,
            head_dim,
            kv_head_num
        );)

    AT_CUDA_CHECK(cudaGetLastError());

}