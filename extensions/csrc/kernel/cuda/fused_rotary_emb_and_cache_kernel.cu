// in transformers source code, huggingface uses fp16 to compute rope so we follow the same precision
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "utils/vec_copy.h"
#include "common/micros.h"
#include "common/mp_type_traits.h"
#include "funcs/cast_functor.h"
#include "funcs/binary_functor.h"

using colossalAI::cuda::utils::get_vec_size;
using colossalAI::cuda::utils::copy;
using colossalAI::funcs::CastFunctor;
using colossalAI::funcs::BinaryOpFunctor;
using colossalAI::funcs::BinaryOpType;

template <typename T, typename MT, int VecSize>
__device__ void apply_emb_rotary_compute(
    T* __restrict__ src, const MT* __restrict__ cos_ptr,
    const MT* __restrict__ sin_ptr, const int64_t stride,
    const int token_id, const int shard_block_size, const int half_head_dim,
    const int head_num, const int head_dim) {
  BinaryOpFunctor<MT, MT, MT, BinaryOpType::kMul> mul;
  BinaryOpFunctor<MT, MT, MT, BinaryOpType::kMinus> sub;
  BinaryOpFunctor<MT, MT, MT, BinaryOpType::kAdd> add;
  CastFunctor<T, MT> t2mt;
  CastFunctor<MT, T> mt2t;

  T x[VecSize];
  T y[VecSize];
  T out_x[VecSize];
  T out_y[VecSize];

  for (int i = threadIdx.x * VecSize; i < head_num * half_head_dim;
       i += blockDim.x * VecSize) {
    const int head_offset = i % half_head_dim;
    const int shard_offset =
        (head_offset / shard_block_size) * shard_block_size +
        (head_offset % shard_block_size) / VecSize;
    const int64_t addr_offset =
        token_id * stride + (i / half_head_dim) * head_dim + head_offset;

    copy<T, VecSize>(src + addr_offset, x);
    copy<T, VecSize>(src + addr_offset + half_head_dim, y);

#pragma unroll
    for (int j = 0; j < VecSize; j++) {
      out_x[j] = mt2t(sub(mul(t2mt(x[j]), cos_ptr[j * 32 + shard_offset]),
                 mul(t2mt(y[j]), sin_ptr[j * 32 + shard_offset])));
      out_y[j] = mt2t(add(mul(t2mt(y[j]), cos_ptr[j * 32 + shard_offset]),
                 mul(t2mt(x[j]), sin_ptr[j * 32 + shard_offset])));
    }

    copy<T, VecSize>(out_x, src + addr_offset);
    copy<T, VecSize>(out_y, src + addr_offset + half_head_dim);
  }
}

template <typename T, typename CacheT, int VecSize>
__device__ void apply_kv_memcopy(
    T* __restrict__ src, CacheT* __restrict__ cache,
    const int64_t stride, const int token_id, const int block_id,
    const int hidden_size, const int block_size, const int block_offset,
    const int head_dim, const int half_head_dim) {
  for (int i = threadIdx.x * VecSize; i < hidden_size / 2;
       i += blockDim.x * VecSize) {
    const int head_id = i / half_head_dim;
    const int head_offset = i % half_head_dim;
    const int64_t src_id = token_id * stride + head_id * head_dim + head_offset;
    const int64_t target_id = block_id * hidden_size * block_size +
                              head_id * block_size * head_dim +
                              block_offset * head_dim + head_offset;

    copy<T, CacheT, VecSize>(src + src_id, cache + target_id);
    copy<T, CacheT, VecSize>(src + src_id + half_head_dim, cache + target_id + half_head_dim);
  }
}

template <typename T, typename MT, int VecSize>
__device__ void cos_sin_memory_access(
    const T* __restrict__ cos, const T* __restrict__ sin,
    MT* cos_ptr, MT* sin_ptr, const int token_id,
    const int shard_block_size, const int cos_stride, const int sin_stride,
    const int half_head_dim) {
  for (int i = threadIdx.x; i < half_head_dim; i += blockDim.x) {
    // We assume that the value of head_dim is less than 128*128.
    const int shard_offset = (i % shard_block_size) / VecSize;
    const int shard_head =
        (i / shard_block_size) * shard_block_size + i % VecSize * 32;
    cos_ptr[shard_head + shard_offset] = CastFunctor<T, MT>()(cos[token_id * cos_stride + i]);
    sin_ptr[shard_head + shard_offset] = CastFunctor<T, MT>()(sin[token_id * sin_stride + i]);
  }
}

template <typename T, typename MT, typename CacheT, int VecSize>
__device__ void apply_k_rotary_emb_compute(
    T* __restrict__ key, T* __restrict__ value,
    CacheT* __restrict__ key_cache, CacheT* __restrict__ value_cache,
    const MT* __restrict__ cos_ptr, const MT* __restrict__ sin_ptr,
    const int* __restrict__ sequence_lengths,
    const int* __restrict__ block_tables, const int64_t key_stride,
    const int64_t value_stride, const int token_id,
    const int block_table_stride, const int head_num, const int head_dim,
    const int kv_head_num, const int block_size, const int x, const int half_head_dim,
    const int shard_block_size) {

  BinaryOpFunctor<MT, MT, MT, BinaryOpType::kMul> mul;
  BinaryOpFunctor<MT, MT, MT, BinaryOpType::kMinus> sub;
  BinaryOpFunctor<MT, MT, MT, BinaryOpType::kAdd> add;
  const int seq_len = sequence_lengths[token_id] - 1;
  const int block_offset = seq_len % block_size;
  const int block_id =
      block_tables[token_id * block_table_stride + seq_len / block_size];

  if (block_id < 0) {
    return;
  }

  T x0[VecSize];
  T x1[VecSize];
  T out_x[VecSize];
  T out_y[VecSize];

  for (int i = threadIdx.x * VecSize; i < kv_head_num * half_head_dim;
       i += blockDim.x * VecSize) {
    const int half_head_offset = i % half_head_dim;
    const int x_id = half_head_offset / x;
    const int x_offset = half_head_offset % x;
    const int shard_offset =
        (half_head_offset / shard_block_size) * shard_block_size +
        (half_head_offset % shard_block_size) / VecSize;
    const int64_t addr_offset =
        token_id * key_stride + (i / half_head_dim) * head_dim + half_head_offset;
    const int64_t target_id = block_id * kv_head_num * head_dim * block_size
                                + (i / half_head_dim) * block_size * head_dim
                                + x_id * block_size * x
                                + block_offset * x
                                + x_offset;

    copy<T, VecSize>(key + addr_offset, x0);
    copy<T, VecSize>(key + addr_offset + half_head_dim, x1);

#pragma unroll
    for (int j = 0; j < VecSize; j++) {
      out_x[j] = CastFunctor<MT, T>()(sub(mul(CastFunctor<T, MT>()(x0[j]), cos_ptr[j * 32 + shard_offset]),
                 mul(CastFunctor<T, MT>()(x1[j]), sin_ptr[j * 32 + shard_offset])));
      out_y[j] = CastFunctor<MT, T>()(add(mul(CastFunctor<T, MT>()(x1[j]), cos_ptr[j * 32 + shard_offset]),
                 mul(CastFunctor<T, MT>()(x0[j]), sin_ptr[j * 32 + shard_offset])));
    }

    copy<T, CacheT, VecSize>(out_x, key_cache + target_id);
    copy<T, CacheT, VecSize>(out_y, key_cache + target_id + half_head_dim * block_size);
  }

  // apply value memcopy
  apply_kv_memcopy<T, CacheT, VecSize>(
      value, value_cache, value_stride, token_id, block_id, kv_head_num * head_dim,
      block_size, block_offset, head_dim, half_head_dim);
}

template<typename T, typename MT, typename CacheT, int VecSize>
__global__ void rotary_embedding_and_cache_copy_kernel(
    T* __restrict__ query,
    T* __restrict__ key,
    T* __restrict__ value,
    const T* __restrict__ cos,
    const T* __restrict__ sin,
    CacheT* __restrict__ key_cache,
    CacheT* __restrict__ value_cache,
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
    const int block_size,
    const int x
) {

    const int token_id = blockIdx.x;
    const int half_head_dim = head_dim / 2;
    const int shard_block_size = VecSize * 32;

    extern __shared__ char shard_ptr[];

    MT *cos_ptr = reinterpret_cast<MT*>(shard_ptr);
    MT *sin_ptr = cos_ptr + half_shard_element_num;

    // apply cos_sin memcopy
    cos_sin_memory_access<T, MT, VecSize>(cos, sin, cos_ptr, sin_ptr, token_id, shard_block_size, cos_stride, sin_stride, half_head_dim);
    __syncthreads();

    //compute query
    apply_emb_rotary_compute<T, MT, VecSize>(query, cos_ptr, sin_ptr, query_stride, token_id, shard_block_size, half_head_dim, head_num, head_dim);

    //compute key and copy kv
    apply_k_rotary_emb_compute<T, MT, CacheT, VecSize>(key, value, key_cache, value_cache, cos_ptr, sin_ptr, sequence_lengths, block_tables, key_stride, value_stride, token_id, block_table_stride, head_num, head_dim, kv_head_num, block_size, x, half_head_dim, shard_block_size);
}

template<typename T, typename MT, int VecSize>
__global__ void rotary_embedding_kernel(
    T* __restrict__ query,
    T* __restrict__ key,
    const T* __restrict__ cos,
    const T* __restrict__ sin,
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

    MT *cos_ptr = (MT*)shard_ptr;
    MT *sin_ptr = cos_ptr + half_shard_element_num;

    // apply cos_sin memcopy
    cos_sin_memory_access<T, MT, VecSize>(cos, sin, cos_ptr, sin_ptr, token_id, shard_block_size, cos_stride, sin_stride, half_head_dim);
    __syncthreads();

    //compute query
    apply_emb_rotary_compute<T, MT, VecSize>(query, cos_ptr, sin_ptr, query_stride, token_id, shard_block_size, half_head_dim, head_num, head_dim);

    //compute key
    apply_emb_rotary_compute<T, MT, VecSize>(key, cos_ptr, sin_ptr, key_stride, token_id, shard_block_size, half_head_dim, kv_head_num, head_dim);
}

#define ROTARY_EMBEDDING_AND_CACHE_COPY_LAUNCHER(VEC_SIZE)                                                              \
  rotary_embedding_and_cache_copy_kernel<T, MT, CacheT, VEC_SIZE><<<grid, block, shared_memory_size, stream>>>(         \
    reinterpret_cast<T*>(query.data_ptr()),                                                                             \
    reinterpret_cast<T*>(key.data_ptr()),                                                                               \
    reinterpret_cast<T*>(value.data_ptr()),                                                                             \
    reinterpret_cast<T*>(cos.data_ptr()),                                                                               \
    reinterpret_cast<T*>(sin.data_ptr()),                                                                               \
    reinterpret_cast<CacheT*>(key_cache.data_ptr()),                                                                    \
    reinterpret_cast<CacheT*>(value_cache.data_ptr()),                                                                  \
    sequence_lengths.data_ptr<int>(),                                                                                   \
    block_tables.data_ptr<int>(),                                                                                       \
    query_stride,                                                                                                       \
    key_stride,                                                                                                         \
    value_stride,                                                                                                       \
    shard_element_num / 2,                                                                                              \
    cos_stride,                                                                                                         \
    sin_stride,                                                                                                         \
    block_table_stride,                                                                                                 \
    head_num,                                                                                                           \
    head_dim,                                                                                                           \
    kv_head_num,                                                                                                        \
    block_size,                                                                                                         \
    x);                                                                                                                 \


template<typename T, typename CacheT, bool high_precision>
void apply_rotary_embedding_and_cache_copy(
    at::Tensor& query,               // [num_tokens, head_num, head_dim]
    at::Tensor& key,                 // [num_tokens, kv_head_num, head_dim]
    at::Tensor& value,               // [num_tokens, kv_head_num, head_dim]
    at::Tensor& cos,                 // [num_tokens, head_dim]
    at::Tensor& sin,                 // [num_tokens, head_dim]
    at::Tensor& key_cache,           // [num_blocks, head_num, head_dim/x, block_size, x]
    at::Tensor& value_cache,         // [num_blocks, head_num, block_size, head_dim]
    at::Tensor& sequence_lengths,    // [batch_size]
    at::Tensor& block_tables)        // [batch_size, max_seq_len]
{
    int num_tokens = query.size(0);
    int head_num = query.size(1);
    int head_dim = query.size(2);
    int kv_head_num = key.size(1);
    int block_size = key_cache.size(3);
    int x = key_cache.size(4);

    int64_t query_stride = query.stride(0);
    int64_t key_stride = key.stride(0);
    int64_t value_stride = value.stride(0);
    int cos_stride = cos.stride(0);
    int sin_stride = sin.stride(0);
    int block_table_stride = block_tables.stride(0);

    using MT = typename colossalAI::common::ScalarTypeTrait<high_precision, T>::Type;

    int vec_size = get_vec_size<T>(query);

    if ((head_dim / 2) % vec_size != 0) {
        // Disable vectorized loading optimization when head_dim is not divisible by VecSize.
        vec_size = 1;
    }

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int thread_nums = head_num * head_dim / vec_size / 2;
    const int shard_block_size = vec_size * 32 * 2;

    dim3 grid(num_tokens);
    dim3 block(std::min(thread_nums, 512));
    int64_t shard_element_num = ((head_dim + shard_block_size - 1) / shard_block_size) * shard_block_size;
    const int shared_memory_size = shard_element_num * sizeof(MT);

    switch (vec_size) {
        case 1:
            ROTARY_EMBEDDING_AND_CACHE_COPY_LAUNCHER(1);
            break;
        case 2:
            ROTARY_EMBEDDING_AND_CACHE_COPY_LAUNCHER(2);
            break;
        case 4:
            ROTARY_EMBEDDING_AND_CACHE_COPY_LAUNCHER(4);
            break;
        default:
            AT_ERROR("Unsupported vectorized size ", vec_size);
            break;
    }

    AT_CUDA_CHECK(cudaGetLastError());
}

template<typename T, bool high_precision>
void apply_rotary_embedding(
    at::Tensor& query,   // [total_tokens, head_num, head_dim]
    at::Tensor& key,     // [total_tokens, kv_head_num, head_dim]
    at::Tensor& cos,     // [total_tokens, head_dim]
    at::Tensor& sin     // [total_tokens, head_dim]
){
    int num_tokens = query.size(0);
    int head_num = query.size(1);
    int head_dim = query.size(2);
    int kv_head_num = key.size(1);

    int query_stride = query.stride(0);
    int key_stride = key.stride(0);
    int cos_stride = cos.stride(0);
    int sin_stride = sin.stride(0);

    using MT = typename colossalAI::common::ScalarTypeTrait<high_precision, T>::Type;

    int vec_size = get_vec_size<T>(query);

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
            rotary_embedding_kernel<T, MT, 1><<<grid, block, shard_element_num * sizeof(MT), stream>>>(
                    query.data_ptr<T>(),
                    key.data_ptr<T>(),
                    cos.data_ptr<T>(),
                    sin.data_ptr<T>(),
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
            rotary_embedding_kernel<T, MT, 2><<<grid, block, shard_element_num * sizeof(MT), stream>>>(
                    query.data_ptr<T>(),
                    key.data_ptr<T>(),
                    cos.data_ptr<T>(),
                    sin.data_ptr<T>(),
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
            rotary_embedding_kernel<T, MT, 4><<<grid, block, shard_element_num * sizeof(MT), stream>>>(
                    query.data_ptr<T>(),
                    key.data_ptr<T>(),
                    cos.data_ptr<T>(),
                    sin.data_ptr<T>(),
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
    at::Tensor& key_cache,           // [num_blocks, head_num, head_dim/x, block_size, x]
    at::Tensor& value_cache,         // [num_blocks, head_num, block_size, head_dim]
    at::Tensor& sequence_lengths,    // [batch_size]
    at::Tensor& block_tables,        // [batch_size, max_seq_len]
    bool high_precision)
{
#define _(T, CacheT, HIGH_PRECISION)                                    \
    apply_rotary_embedding_and_cache_copy<T, CacheT, HIGH_PRECISION>(   \
        query,                                                          \
        key,                                                            \
        value,                                                          \
        cos,                                                            \
        sin,                                                            \
        key_cache,                                                      \
        value_cache,                                                    \
        sequence_lengths,                                               \
        block_tables);

    if(key_cache.scalar_type() == at::ScalarType::Byte)
    {
        if(high_precision) {
            switch (key.scalar_type())
            {
            case at::ScalarType::Float:
                _(float, uint8_t, true)
                break;
            case at::ScalarType::Half:
                _(half, uint8_t, true)
                break;
            case at::ScalarType::BFloat16:
                _(__nv_bfloat16, uint8_t, true)
                break;
            }
        }
        else {
            switch (key.scalar_type())
            {
            case at::ScalarType::Float:
                _(float, uint8_t, false)
                break;
            case at::ScalarType::Half:
                _(half, uint8_t, false)
                break;
            case at::ScalarType::BFloat16:
                _(__nv_bfloat16, uint8_t, false)
                break;
            }
        }
    }
    else
    {
        if(high_precision) {
            switch (key.scalar_type())
            {
            case at::ScalarType::Float:
                _(float, float, true)
                break;
            case at::ScalarType::Half:
                _(half, half, true)
                break;
            case at::ScalarType::BFloat16:
                _(__nv_bfloat16, __nv_bfloat16, true)
                break;
            }
        }
        else {
            switch (key.scalar_type())
            {
            case at::ScalarType::Float:
                _(float, float, false)
                break;
            case at::ScalarType::Half:
                _(half, half, false)
                break;
            case at::ScalarType::BFloat16:
                _(__nv_bfloat16, __nv_bfloat16, false)
                break;
            }
        }
    }
#undef _
}

void rotary_embedding(
    at::Tensor& query,   // [total_tokens, head_num, head_dim]
    at::Tensor& key,     // [total_tokens, kv_head_num, head_dim]
    at::Tensor& cos,     // [total_tokens, head_dim]
    at::Tensor& sin,      // [total_tokens, head_dim]
    bool high_precision
){
    DISPATCH_FLOAT_HALF_AND_BFLOAT_WITH_HIGH_PRECISION(
        high_precision,
        query.scalar_type(),
        "rotary_embedding",
        apply_rotary_embedding<scalar_t, high_precision>(
            query,
            key,
            cos,
            sin
        );)
}
