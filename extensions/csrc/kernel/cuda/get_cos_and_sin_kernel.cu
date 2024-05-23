#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "utils/vec_copy.h"
#include "common/micros.h"

using colossalAI::cuda::utils::copy;
using colossalAI::cuda::utils::get_vec_size;


template <typename scalar_t, bool Aligned, int VecSize>
__device__ void apply_cos_and_sin_memcopy(
    scalar_t* __restrict__ cos,
    scalar_t* __restrict__ sin,
    const scalar_t* __restrict__ cos_cache_ptr,
    const scalar_t* __restrict__ sin_cache_ptr,
    const int* __restrict__ sequence_lengths,
    const int head_dim,
    const int dest_offset_id,
    const int src_offset_id
 ) {

    int begin_id = threadIdx.x * VecSize;

    for (; begin_id <= head_dim - VecSize; begin_id += blockDim.x){
        copy<scalar_t, VecSize>(cos_cache_ptr + src_offset_id + begin_id, cos + dest_offset_id + begin_id);
        copy<scalar_t, VecSize>(sin_cache_ptr + src_offset_id + begin_id, sin + dest_offset_id + begin_id);
    }

    if (!Aligned) {
        for (; begin_id < head_dim; ++begin_id ) {
            cos[dest_offset_id + begin_id] = cos_cache_ptr[src_offset_id + begin_id];
            sin[dest_offset_id + begin_id] = sin_cache_ptr[src_offset_id + begin_id];
        }
    }
}

template <typename scalar_t, bool Aligned, int VecSize>
__global__ void apply_get_context_cos_and_sin_kernel(
    scalar_t* __restrict__ cos,
    scalar_t* __restrict__ sin,
    const scalar_t* __restrict__ cos_cache_ptr,
    const scalar_t* __restrict__ sin_cache_ptr,
    const int* __restrict__ sequence_lengths,
    const int* __restrict__ cumsum_lengths,
    const int batch_size,
    const int head_dim
) {
    int token_id = blockIdx.x;
    if ( token_id >= sequence_lengths[blockIdx.y] ) {
        return ;
    }

    int src_offset_id = token_id * head_dim;
    int dest_offset_id = src_offset_id;

    if (blockIdx.y > 0) {
        dest_offset_id += cumsum_lengths[blockIdx.y - 1] * head_dim;
    }

    apply_cos_and_sin_memcopy<scalar_t, Aligned, VecSize>(
        cos,
        sin,
        cos_cache_ptr,
        sin_cache_ptr,
        sequence_lengths,
        head_dim,
        dest_offset_id,
        src_offset_id
    );

}

template <typename scalar_t, bool Aligned, int VecSize>
__global__ void apply_get_decode_cos_and_sin_kernel(
    scalar_t* __restrict__ cos,
    scalar_t* __restrict__ sin,
    const scalar_t* __restrict__ cos_cache_ptr,
    const scalar_t* __restrict__ sin_cache_ptr,
    const int* __restrict__ sequence_lengths,
    const int batch_size,
    const int head_dim
) {
    int src_offset_id = ( sequence_lengths[blockIdx.y] - 1 ) * head_dim;
    int dest_offset_id = blockIdx.y * head_dim;

    apply_cos_and_sin_memcopy<scalar_t, Aligned, VecSize>(
        cos,
        sin,
        cos_cache_ptr,
        sin_cache_ptr,
        sequence_lengths,
        head_dim,
        dest_offset_id,
        src_offset_id
    );
}

template<typename scalar_t>
void apply_get_cos_and_sin(
    at::Tensor& cos_cache,           // [max_rotary_position, head_dim]
    at::Tensor& sin_cache,           // [max_rotary_position, head_dim]
    at::Tensor& cos,                 // [num_tokens, head_dim]
    at::Tensor& sin,                 // [num_tokens, head_dim]
    at::Tensor& sequence_lengths,    // [batch_size]
    int max_seq_len_in_batch,
    bool is_prompts
) {
    int token_num = cos.size(0);
    int head_dim = cos.size(1);
    int batch_size = sequence_lengths.size(0);

    at::Tensor cumsum_lengths;

    int vec_size = get_vec_size<scalar_t>(cos);

    bool aligned = true;
    if (head_dim % vec_size != 0) {
        aligned = false;
    }

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int block_size_y;
    int block_size_x;

    if (is_prompts) {
        block_size_y = batch_size;
        block_size_x = max_seq_len_in_batch;
        // TODO: The cumsum operation can be fused into get_cos_and_sin kernel later on.
        cumsum_lengths = torch::cumsum(sequence_lengths, 0, torch::kInt32);
    }
    else{
        block_size_y = batch_size;
        block_size_x = 1;
    }

    int thread_nums = (head_dim + vec_size - 1) / vec_size;

    dim3 grid(block_size_x, block_size_y);
    dim3 block(std::min(thread_nums, 512));

#define GET_COS_AND_SIN_KERNEL_LAUNCH(__aligned, __vec_size)                                                        \
    do {                                                                                                            \
        if (is_prompts){                                                                                            \
            apply_get_context_cos_and_sin_kernel<scalar_t, __aligned, __vec_size><<<grid, block, 0, stream>>>(      \
                cos.data_ptr<scalar_t>(),                                                                           \
                sin.data_ptr<scalar_t>(),                                                                           \
                cos_cache.data_ptr<scalar_t>(),                                                                     \
                sin_cache.data_ptr<scalar_t>(),                                                                     \
                sequence_lengths.data_ptr<int>(),                                                                   \
                cumsum_lengths.data_ptr<int>(),                                                                     \
                batch_size,                                                                                         \
                head_dim                                                                                            \
            );                                                                                                      \
        }                                                                                                           \
        else {                                                                                                      \
            apply_get_decode_cos_and_sin_kernel<scalar_t, __aligned, __vec_size><<<grid, block, 0, stream>>>(       \
                cos.data_ptr<scalar_t>(),                                                                           \
                sin.data_ptr<scalar_t>(),                                                                           \
                cos_cache.data_ptr<scalar_t>(),                                                                     \
                sin_cache.data_ptr<scalar_t>(),                                                                     \
                sequence_lengths.data_ptr<int>(),                                                                   \
                batch_size,                                                                                         \
                head_dim                                                                                            \
            );                                                                                                      \
        }                                                                                                           \
    } while(0)

#define GET_COS_AND_SIN_KERNEL_LAUNCH_VEC_SIZE_CASE(__aligned)                                          \
    do {                                                                                                \
        switch (vec_size) {                                                                             \
            case 1:                                                                                     \
                GET_COS_AND_SIN_KERNEL_LAUNCH(__aligned, 1);                                            \
                break;                                                                                  \
            case 2:                                                                                     \
                GET_COS_AND_SIN_KERNEL_LAUNCH(__aligned, 2);                                            \
                break;                                                                                  \
            case 4:                                                                                     \
                GET_COS_AND_SIN_KERNEL_LAUNCH(__aligned, 4);                                            \
                break;                                                                                  \
            default:                                                                                    \
                AT_ERROR("Unsupported vectorized size ", vec_size);                                     \
                break;                                                                                  \
        }                                                                                               \
    } while(0)

    if (aligned) {
        GET_COS_AND_SIN_KERNEL_LAUNCH_VEC_SIZE_CASE(true);
    }
    else {
        GET_COS_AND_SIN_KERNEL_LAUNCH_VEC_SIZE_CASE(false);
    }

    AT_CUDA_CHECK(cudaGetLastError());
}

void get_cos_and_sin(
    at::Tensor& cos_cache,           // [max_rotary_position, head_dim]
    at::Tensor& sin_cache,           // [max_rotary_position, head_dim]
    at::Tensor& cos,                 // [num_tokens, head_dim]
    at::Tensor& sin,                 // [num_tokens, head_dim]
    at::Tensor& sequence_lengths,    // [batch_size]
    int max_seq_len_in_batch,
    bool is_prompts
) {
    DISPATCH_FLOAT_HALF_AND_BFLOAT(
        cos.scalar_type(),
        "get_cos_and_sin",
        apply_get_cos_and_sin<scalar_t>(
            cos_cache,
            sin_cache,
            cos,
            sin,
            sequence_lengths,
            max_seq_len_in_batch,
            is_prompts
        );)
}
