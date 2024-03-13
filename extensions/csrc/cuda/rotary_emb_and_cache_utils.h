#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "../common/vector_copy_utils.h"
#include "stdio.h"

template <typename scalar_t, int VecSize>
inline __device__ void apply_emb_rotary_compute(
    scalar_t* __restrict__ src, const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr, const int64_t stride,
    const int token_id, const int shard_block_size, const int half_head_dim,
    const int head_num, const int head_dim) {
  scalar_t x[VecSize];
  scalar_t y[VecSize];
  scalar_t out_x[VecSize];
  scalar_t out_y[VecSize];

  for (int i = threadIdx.x * VecSize; i < head_num * half_head_dim;
       i += blockDim.x * VecSize) {
    const int head_offset = i % half_head_dim;
    const int shard_offset =
        (head_offset / shard_block_size) * shard_block_size +
        (head_offset % shard_block_size) / VecSize;
    const int64_t addr_offset =
        token_id * stride + (i / half_head_dim) * head_dim + head_offset;

    copy_vector<scalar_t, VecSize>(x, src + addr_offset);
    copy_vector<scalar_t, VecSize>(y, src + addr_offset + half_head_dim);

#pragma unroll
    for (int j = 0; j < VecSize; j++) {
      out_x[j] = x[j] * cos_ptr[j * 32 + shard_offset] -
                 y[j] * sin_ptr[j * 32 + shard_offset];
      out_y[j] = y[j] * cos_ptr[j * 32 + shard_offset] +
                 x[j] * sin_ptr[j * 32 + shard_offset];
    }

    copy_vector<scalar_t, VecSize>(src + addr_offset, out_x);
    copy_vector<scalar_t, VecSize>(src + addr_offset + half_head_dim, out_y);
  }
}

template <typename scalar_t, int VecSize>
inline __device__ void apply_kv_memcopy(
    scalar_t* __restrict__ src, scalar_t* __restrict__ cache,
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

    copy_vector<scalar_t, VecSize>(cache + target_id, src + src_id);
    copy_vector<scalar_t, VecSize>(cache + target_id + half_head_dim,
                                   src + src_id + half_head_dim);
  }
}

template <typename scalar_t, int VecSize>
inline __device__ void cos_sin_memory_access(
    const scalar_t* __restrict__ cos, const scalar_t* __restrict__ sin,
    scalar_t* cos_ptr, scalar_t* sin_ptr, const int token_id,
    const int shard_block_size, const int cos_stride, const int sin_stride,
    const int half_head_dim) {
  for (int i = threadIdx.x; i < half_head_dim; i += blockDim.x) {
    // We assume that the value of head_dim is less than 128*128.
    const int shard_offset = (i % shard_block_size) / VecSize;
    const int shard_head =
        (i / shard_block_size) * shard_block_size + i % VecSize * 32;
    cos_ptr[shard_head + shard_offset] = cos[token_id * cos_stride + i];
    sin_ptr[shard_head + shard_offset] = sin[token_id * sin_stride + i];
  }
}

template <typename scalar_t, int VecSize>
inline __device__ void apply_k_rotary_emb_compute(
    scalar_t* __restrict__ key, scalar_t* __restrict__ value,
    scalar_t* __restrict__ key_cache, scalar_t* __restrict__ value_cache,
    const scalar_t* __restrict__ cos_ptr, const scalar_t* __restrict__ sin_ptr,
    const int* __restrict__ sequence_lengths,
    const int* __restrict__ block_tables, const int64_t key_stride,
    const int64_t value_stride, const int token_id,
    const int block_table_stride, const int head_num, const int head_dim,
    const int kv_head_num, const int block_size, const int half_head_dim,
    const int shard_block_size) {
  const int seq_len = sequence_lengths[token_id] - 1;
  const int block_offset = seq_len % block_size;
  const int block_id =
      block_tables[token_id * block_table_stride + seq_len / block_size];

  if (block_id < 0) {
    return;
  }

  scalar_t x[VecSize];
  scalar_t y[VecSize];
  scalar_t out_x[VecSize];
  scalar_t out_y[VecSize];

  for (int i = threadIdx.x * VecSize; i < kv_head_num * half_head_dim;
       i += blockDim.x * VecSize) {
    const int head_offset = i % half_head_dim;
    const int shard_offset =
        (head_offset / shard_block_size) * shard_block_size +
        (head_offset % shard_block_size) / VecSize;
    const int64_t addr_offset =
        token_id * key_stride + (i / half_head_dim) * head_dim + head_offset;
    const int64_t target_id = block_id * head_num * head_dim * block_size +
                              (i / half_head_dim) * block_size * head_dim +
                              block_offset * head_dim + head_offset;

    copy_vector<scalar_t, VecSize>(x, key + addr_offset);
    copy_vector<scalar_t, VecSize>(y, key + addr_offset + half_head_dim);

#pragma unroll
    for (int j = 0; j < VecSize; j++) {
      out_x[j] = x[j] * cos_ptr[j * 32 + shard_offset] -
                 y[j] * sin_ptr[j * 32 + shard_offset];
      out_y[j] = y[j] * cos_ptr[j * 32 + shard_offset] +
                 x[j] * sin_ptr[j * 32 + shard_offset];
    }

    copy_vector<scalar_t, VecSize>(key_cache + target_id, out_x);
    copy_vector<scalar_t, VecSize>(key_cache + target_id + half_head_dim,
                                   out_y);
  }

  // apply value memcopy
  apply_kv_memcopy<scalar_t, VecSize>(
      value, value_cache, value_stride, token_id, block_id, head_num * head_dim,
      block_size, block_offset, head_dim, half_head_dim);
}
