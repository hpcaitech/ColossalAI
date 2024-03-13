
#include "../common/micros.h"
#include "rotary_emb_and_cache_utils.h"
#include "stdio.h"

template<typename scalar_t, int VecSize>
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
    const int64_t half_shard_element_num,
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
    const int shard_block_size = VecSize * 32;

    extern __shared__ char shard_ptr[];

    scalar_t *cos_ptr = (scalar_t*)shard_ptr;
    scalar_t *sin_ptr = cos_ptr + half_shard_element_num;

    // apply cos_sin memcopy
    cos_sin_memory_access<scalar_t, VecSize>(cos, sin, cos_ptr, sin_ptr, token_id, shard_block_size, cos_stride, sin_stride, half_head_dim);
    __syncthreads();

    //compute query
    apply_emb_rotary_compute<scalar_t, VecSize>(query, cos_ptr, sin_ptr, query_stride, token_id, shard_block_size, half_head_dim, head_num, head_dim);

    //compute key and copy kv
    apply_k_rotary_emb_compute<scalar_t, VecSize>(key, value, key_cache, value_cache, cos_ptr, sin_ptr, sequence_lengths, block_tables, key_stride, value_stride, token_id, block_table_stride, head_num, head_dim, kv_head_num, block_size, half_head_dim, shard_block_size);
}

template<typename scalar_t, int VecSize>
__global__ void rotary_embedding_kernel(
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    const int64_t query_stride,
    const int64_t key_stride,
    const int64_t half_shard_element_num,
    const int cos_stride,
    const int sin_stride,
    const int head_num,
    const int head_dim,
    const int kv_head_num
) {
    const int token_id = blockIdx.x;
    const int half_head_dim = head_dim / 2;
    const int shard_block_size = VecSize * 32;

    extern __shared__ char shard_ptr[];

    scalar_t *cos_ptr = (scalar_t*)shard_ptr;
    scalar_t *sin_ptr = cos_ptr + half_shard_element_num;

    // apply cos_sin memcopy
    cos_sin_memory_access<scalar_t, VecSize>(cos, sin, cos_ptr, sin_ptr, token_id, shard_block_size, cos_stride, sin_stride, half_head_dim);
    __syncthreads();

    //compute query
    apply_emb_rotary_compute<scalar_t, VecSize>(query, cos_ptr, sin_ptr, query_stride, token_id, shard_block_size, half_head_dim, head_num, head_dim);

    //compute key
    apply_emb_rotary_compute<scalar_t, VecSize>(key, cos_ptr, sin_ptr, key_stride, token_id, shard_block_size, half_head_dim, kv_head_num, head_dim);
}

template<typename scalar_t>
void apply_rotary_embedding_and_cache_copy(
    at::Tensor& query,               // [num_tokens, head_num, head_dim]
    at::Tensor& key,                 // [num_tokens, kv_head_num, head_dim]
    at::Tensor& value,               // [num_tokens, kv_head_num, head_dim]
    at::Tensor& cos,                 // [num_tokens, head_dim]
    at::Tensor& sin,                 // [num_tokens, head_dim]
    at::Tensor& key_cache,           // [num_blocks, head_num, block_size, head_dim]
    at::Tensor& value_cache,         // [num_blocks, head_num, block_size, head_dim]
    at::Tensor& sequence_lengths,    // [batch_size]
    at::Tensor& block_tables)        // [batch_size, max_seq_len]
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

    int vec_size = get_vec_size<scalar_t>(query);

    if ((head_dim / 2) % vec_size != 0) {
        // Disable vectorized loading optimization when head_dim is not divisible by VecSize.
        vec_size = 1;
    }

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int thread_nums = head_num * head_dim / vec_size / 2;
    const int shard_block_size = vec_size * 32 * 2;

    dim3 grid(num_tokens);
    dim3 block(std::min(thread_nums, 512));
    int64_t shard_element_num = ((head_dim + shard_block_size - 1) / shard_block_size) * shard_block_size ;

    switch (vec_size) {
        case 1:
            rotary_embedding_and_cache_copy_kernel<scalar_t, 1><<<grid, block, shard_element_num * sizeof(scalar_t), stream>>>(
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
                shard_element_num / 2,
                cos_stride,
                sin_stride,
                block_table_stride,
                head_num,
                head_dim,
                kv_head_num,
                block_size
            );
            break;
        case 2:
            rotary_embedding_and_cache_copy_kernel<scalar_t, 2><<<grid, block, shard_element_num * sizeof(scalar_t), stream>>>(
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
                shard_element_num / 2,
                cos_stride,
                sin_stride,
                block_table_stride,
                head_num,
                head_dim,
                kv_head_num,
                block_size
            );
            break;
        case 4:
            rotary_embedding_and_cache_copy_kernel<scalar_t, 4><<<grid, block, shard_element_num * sizeof(scalar_t), stream>>>(
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
                shard_element_num / 2,
                cos_stride,
                sin_stride,
                block_table_stride,
                head_num,
                head_dim,
                kv_head_num,
                block_size
            );
            break;
        default:
            AT_ERROR("Unsupported vectorized size ", vec_size);
            break;
    }

    AT_CUDA_CHECK(cudaGetLastError());
}

template<typename scalar_t>
void apply_rotary_embedding(
    at::Tensor& query,   // [total_tokens, head_num, head_dim]
    at::Tensor& key,     // [total_tokens, kv_head_num, head_dim]
    at::Tensor& cos,     // [total_tokens, head_dim]
    at::Tensor& sin      // [total_tokens, head_dim]
){
    int num_tokens = query.size(0);
    int head_num = query.size(1);
    int head_dim = query.size(2);
    int kv_head_num = key.size(1);

    int query_stride = query.stride(0);
    int key_stride = key.stride(0);
    int cos_stride = cos.stride(0);
    int sin_stride = sin.stride(0);

    int vec_size = get_vec_size<scalar_t>(query);

    if ((head_dim / 2) % vec_size != 0) {
        // Disable vectorized loading optimization when head_dim is not divisible by VecSize.
        vec_size = 1;
    }

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int thread_nums = head_num * head_dim / vec_size / 2;
    const int shard_block_size = vec_size * 32 * 2;

    dim3 grid(num_tokens);
    dim3 block(std::min(thread_nums, 512));
    int64_t shard_element_num = ((head_dim + shard_block_size - 1) / shard_block_size) * shard_block_size ;

    switch (vec_size) {
        case 1:
            rotary_embedding_kernel<scalar_t, 1><<<grid, block, shard_element_num * sizeof(scalar_t), stream>>>(
                    query.data_ptr<scalar_t>(),
                    key.data_ptr<scalar_t>(),
                    cos.data_ptr<scalar_t>(),
                    sin.data_ptr<scalar_t>(),
                    query_stride,
                    key_stride,
                    shard_element_num / 2,
                    cos_stride,
                    sin_stride,
                    head_num,
                    head_dim,
                    kv_head_num
                );
            break;
        case 2:
            rotary_embedding_kernel<scalar_t, 2><<<grid, block, shard_element_num * sizeof(scalar_t), stream>>>(
                    query.data_ptr<scalar_t>(),
                    key.data_ptr<scalar_t>(),
                    cos.data_ptr<scalar_t>(),
                    sin.data_ptr<scalar_t>(),
                    query_stride,
                    key_stride,
                    shard_element_num / 2,
                    cos_stride,
                    sin_stride,
                    head_num,
                    head_dim,
                    kv_head_num
                );
            break;
        case 4:
            rotary_embedding_kernel<scalar_t, 4><<<grid, block, shard_element_num * sizeof(scalar_t), stream>>>(
                    query.data_ptr<scalar_t>(),
                    key.data_ptr<scalar_t>(),
                    cos.data_ptr<scalar_t>(),
                    sin.data_ptr<scalar_t>(),
                    query_stride,
                    key_stride,
                    shard_element_num / 2,
                    cos_stride,
                    sin_stride,
                    head_num,
                    head_dim,
                    kv_head_num
                );
            break;
        default:
            AT_ERROR("Unsupported vectorized size ", vec_size);
            break;
    }
    AT_CUDA_CHECK(cudaGetLastError());
}

void rotary_embedding_and_cache_copy(
    at::Tensor& query,               // [num_tokens, head_num, head_dim]
    at::Tensor& key,                 // [num_tokens, kv_head_num, head_dim]
    at::Tensor& value,               // [num_tokens, kv_head_num, head_dim]
    at::Tensor& cos,                 // [num_tokens, head_dim]
    at::Tensor& sin,                 // [num_tokens, head_dim]
    at::Tensor& key_cache,           // [num_blocks, head_num, block_size, head_dim]
    at::Tensor& value_cache,         // [num_blocks, head_num, block_size, head_dim]
    at::Tensor& sequence_lengths,    // [batch_size]
    at::Tensor& block_tables)        // [batch_size, max_seq_len]
{
    DISPATCH_FLOAT_HALF_AND_BFLOAT(
        query.scalar_type(),
        "rotary_embedding_and_cache_copy",
        apply_rotary_embedding_and_cache_copy<scalar_t>(
            query,
            key,
            value,
            cos,
            sin,
            key_cache,
            value_cache,
            sequence_lengths,
            block_tables
        );)
}

void rotary_embedding(
    at::Tensor& query,   // [total_tokens, head_num, head_dim]
    at::Tensor& key,     // [total_tokens, kv_head_num, head_dim]
    at::Tensor& cos,     // [total_tokens, head_dim]
    at::Tensor& sin      // [total_tokens, head_dim]
){
    DISPATCH_FLOAT_HALF_AND_BFLOAT(
        query.scalar_type(),
        "rotary_embedding",
        apply_rotary_embedding<scalar_t>(
            query,
            key,
            cos,
            sin
        );)
}
