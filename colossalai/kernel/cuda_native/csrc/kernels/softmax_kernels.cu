#include <cooperative_groups.h>
#include <math.h>

#include <cub/block/block_load.cuh>
#include <cub/cub.cuh>

#include "block_reduce.h"
#include "kernels.h"

namespace cg = cooperative_groups;
const float EPSILON = 1e-8f;

/**
@brief: softmax_kernel
Softmax forward kernel for
  enc-self-attn, dec-self-attn, encdec-attn

@thread
gridDim.x = dynamic
gridDim.y = batch_size
gridDim.z = nhead
blockDim.x = from_len

@param
inp: [batch_size, nhead, from_len, to_len], softmax input.
attn_mask: [batch_size, to_len], padding tokens are -inf,
  non padding tokens are 0.
  attn_mask!=nullptr for enc-self-attn and enc-dec-attn
  attn_mask=nullptr and mask_future=ture for dec-self-attn training
  attn_mask=nullptr and mask_future=false for dec-self-attn infer
*/
template <typename T, int block_dim, int ele_per_thread>
__global__ void ker_attn_softmax(T *inp, const T *attn_mask, int from_len,
                                 int to_len, bool mask_future) {
  int batch_id = blockIdx.y;
  int head_id = blockIdx.z;
  const int nhead = gridDim.z;
  const int token_per_reduce = 1;
  typedef cub::BlockLoad<T, block_dim, ele_per_thread,
                         cub::BLOCK_LOAD_VECTORIZE>
      BlockLoad;
  __shared__ typename BlockLoad::TempStorage ts_load;
  typedef cub::BlockStore<T, block_dim, ele_per_thread,
                          cub::BLOCK_STORE_VECTORIZE>
      BlockStore;
  __shared__ typename BlockStore::TempStorage ts_store;

  T mval[ele_per_thread];
  if (attn_mask) {
    attn_mask += batch_id * to_len;
    BlockLoad(ts_load).Load(attn_mask, mval, to_len, REDUCE_FLOAT_INF_NEG);
  }

  inp += flat_3dim(batch_id, head_id, 0, nhead, from_len * to_len);
  for (int token_id = blockIdx.x * token_per_reduce; token_id < from_len;
       token_id += gridDim.x * token_per_reduce) {
    T inp_val[token_per_reduce][ele_per_thread];
    for (int i = 0; i < token_per_reduce && (token_id + i) < from_len; i++) {
      BlockLoad(ts_load).Load(inp + (token_id + i) * to_len, inp_val[i], to_len,
                              REDUCE_FLOAT_INF_NEG);
    }

    /* step 1. compute max */
    // thread local max
    float val[token_per_reduce][ele_per_thread];
    float l_max[token_per_reduce];
    for (int i = 0; i < token_per_reduce; i++) {
      l_max[i] = REDUCE_FLOAT_INF_NEG;
      for (int j = 0; j < ele_per_thread; j++) {
        if (attn_mask) {
          val[i][j] = (float)inp_val[i][j] + (float)mval[j];
        } else {
          if (mask_future && ele_per_thread * threadIdx.x + j > token_id + i) {
            val[i][j] = REDUCE_FLOAT_INF_NEG;
          } else {
            val[i][j] = (float)inp_val[i][j];
          }
        }
        l_max[i] = fmaxf(l_max[i], val[i][j]);
      }
    }
    // block reduce max
    blockReduce<ReduceType::kMax, token_per_reduce>(l_max);
    // write shared
    __shared__ float s_max[token_per_reduce];
    if (threadIdx.x == 0) {
      for (int i = 0; i < token_per_reduce; i++) {
        s_max[i] = l_max[i];
      }
    }
    __syncthreads();

    /* step 2. compute sum */
    // thread local sum
    float l_sum[token_per_reduce];
    for (int i = 0; i < token_per_reduce; i++) {
      l_sum[i] = 0.f;
      for (int j = 0; j < ele_per_thread; j++) {
        val[i][j] = __expf(val[i][j] - s_max[i]);
        l_sum[i] += val[i][j];
      }
    }
    // block reduce sum
    blockReduce<ReduceType::kSum, token_per_reduce>(l_sum);
    // write shared
    __shared__ float s_sum[token_per_reduce];
    if (threadIdx.x == 0) {
      for (int i = 0; i < token_per_reduce; i++) {
        s_sum[i] = __fdividef(1.0f, l_sum[i] + EPSILON);
      }
    }
    __syncthreads();

    /* step 3. compute final result */
    for (int i = 0; i < token_per_reduce && (token_id + i) < from_len; i++) {
      for (int j = 0; j < ele_per_thread; j++) {
        inp_val[i][j] = (T)(val[i][j] * s_sum[i]);
      }
      BlockStore(ts_store).Store(inp + (token_id + i) * to_len, inp_val[i],
                                 to_len);
    }
  }  // blockIdx.x
}

template <typename T, int block_dim, int ele_per_thread>
__global__ void ker_attn_softmax_lt32(T *inp, const T *attn_mask, int from_len,
                                      int to_len, bool mask_future) {
  int batch_id = blockIdx.y;
  int head_id = blockIdx.z;
  const int nhead = gridDim.z;
  const int token_per_reduce = 1;
  typedef cub::BlockLoad<T, block_dim, ele_per_thread,
                         cub::BLOCK_LOAD_VECTORIZE>
      BlockLoad;
  __shared__ typename BlockLoad::TempStorage ts_load;
  typedef cub::BlockStore<T, block_dim, ele_per_thread,
                          cub::BLOCK_STORE_VECTORIZE>
      BlockStore;
  __shared__ typename BlockStore::TempStorage ts_store;

  T mval[ele_per_thread];
  if (attn_mask) {
    attn_mask += batch_id * to_len;
    BlockLoad(ts_load).Load(attn_mask, mval, to_len, REDUCE_FLOAT_INF_NEG);
  }

  inp += flat_3dim(batch_id, head_id, 0, nhead, from_len * to_len);
  for (int token_id = blockIdx.x * token_per_reduce; token_id < from_len;
       token_id += gridDim.x * token_per_reduce) {
    T inp_val[token_per_reduce][ele_per_thread];
    for (int i = 0; i < token_per_reduce && (token_id + i) < from_len; i++) {
      BlockLoad(ts_load).Load(inp + (token_id + i) * to_len, inp_val[i], to_len,
                              REDUCE_FLOAT_INF_NEG);
    }

    /* step 1. compute max */
    // thread local max
    float val[token_per_reduce][ele_per_thread];
    float l_max[token_per_reduce];
    for (int i = 0; i < token_per_reduce; i++) {
      l_max[i] = REDUCE_FLOAT_INF_NEG;
      for (int j = 0; j < ele_per_thread; j++) {
        if (attn_mask) {
          val[i][j] = (float)inp_val[i][j] + (float)mval[j];
        } else {
          if (mask_future && ele_per_thread * threadIdx.x + j > token_id + i) {
            val[i][j] = REDUCE_FLOAT_INF_NEG;
          } else {
            val[i][j] = (float)inp_val[i][j];
          }
        }
        l_max[i] = fmaxf(l_max[i], val[i][j]);
      }
    }
    // warp reduce max
    warpReduce<ReduceType::kMax, token_per_reduce>(l_max);

    /* step 2. compute sum */
    // thread local sum
    float l_sum[token_per_reduce];
    for (int i = 0; i < token_per_reduce; i++) {
      l_sum[i] = 0.f;
      for (int j = 0; j < ele_per_thread; j++) {
        val[i][j] = __expf(val[i][j] - l_max[i]);
        l_sum[i] += val[i][j];
      }
    }
    // warp reduce sum
    warpReduce<ReduceType::kSum, token_per_reduce>(l_sum);

    /* step 3. compute final result */
    for (int i = 0; i < token_per_reduce && (token_id + i) < from_len; i++) {
      l_sum[i] = __fdividef(1.0f, l_sum[i] + EPSILON);
      for (int j = 0; j < ele_per_thread; j++) {
        inp_val[i][j] = (T)(val[i][j] * l_sum[i]);
      }
      BlockStore(ts_store).Store(inp + (token_id + i) * to_len, inp_val[i],
                                 to_len);
    }
  }  // blockIdx.x
}

/*
  attn_mask!=nullptr for enc-self-attn and enc-dec-attn
  attn_mask=nullptr and mask_future=ture for dec-self-attn training
  attn_mask=nullptr and mask_future=false for dec-self-attn infer
*/
template <>
void launch_attn_softmax<float>(float *inp, const float *attn_mask,
                                int batch_size, int nhead, int from_len,
                                int to_len, bool mask_future,
                                cudaStream_t stream) {
  dim3 grid_dim(1, batch_size, nhead);
  if (to_len <= 32) {
    ker_attn_softmax_lt32<float, 32, 1><<<grid_dim, 32, 0, stream>>>(
        inp, attn_mask, from_len, to_len, mask_future);
  } else if (to_len <= 64) {
    ker_attn_softmax_lt32<float, 32, 2><<<grid_dim, 32, 0, stream>>>(
        inp, attn_mask, from_len, to_len, mask_future);
  } else if (to_len <= 128) {
    grid_dim.x = 16;
    ker_attn_softmax<float, 64, 2><<<grid_dim, 64, 0, stream>>>(
        inp, attn_mask, from_len, to_len, mask_future);
  } else if (to_len <= 256) {
    grid_dim.x = 32;
    ker_attn_softmax<float, 128, 2><<<grid_dim, 128, 0, stream>>>(
        inp, attn_mask, from_len, to_len, mask_future);
  } else if (to_len <= 512) {
    grid_dim.x = 64;
    ker_attn_softmax<float, 256, 2><<<grid_dim, 256, 0, stream>>>(
        inp, attn_mask, from_len, to_len, mask_future);
  } else {
    throw std::runtime_error(
        "Sequence length greater than 512 is currently not supported");
  }
}

template <>
void launch_attn_softmax<__half>(__half *inp, const __half *attn_mask,
                                 int batch_size, int nhead, int from_len,
                                 int to_len, bool mask_future,
                                 cudaStream_t stream) {
  dim3 grid_dim(1, batch_size, nhead);
  if (to_len <= 32) {
    ker_attn_softmax_lt32<__half, 32, 1><<<grid_dim, 32, 0, stream>>>(
        inp, attn_mask, from_len, to_len, mask_future);
  } else if (to_len <= 64) {
    ker_attn_softmax_lt32<__half, 32, 2><<<grid_dim, 32, 0, stream>>>(
        inp, attn_mask, from_len, to_len, mask_future);
  } else if (to_len <= 128) {
    grid_dim.x = 8;
    ker_attn_softmax<__half, 64, 2><<<grid_dim, 64, 0, stream>>>(
        inp, attn_mask, from_len, to_len, mask_future);
  } else if (to_len <= 256) {
    grid_dim.x = 16;
    ker_attn_softmax<__half, 128, 2><<<grid_dim, 128, 0, stream>>>(
        inp, attn_mask, from_len, to_len, mask_future);
  } else if (to_len <= 512) {
    grid_dim.x = 32;
    ker_attn_softmax<__half, 256, 2><<<grid_dim, 256, 0, stream>>>(
        inp, attn_mask, from_len, to_len, mask_future);
  } else {
    throw std::runtime_error(
        "Sequence length greater than 512 is currently not supported");
  }
}

/**
@brief: ker_attn_softmax_bw
Softmax backward in self attention.

@thread
gridDim.x = batch_size * nhead * seq_len / warps_per_block
blockDim.x = WARP_SIZE
blockDim.y = warps_per_block

@param
grad: [batch_size, nhead, seq_len, seq_len], output grad.
output: [batch_size, nhead, seq_len, seq_len], output of softmax forward.
*/
template <typename T, int ITERATIONS>
__global__ void ker_attn_softmax_bw(T *grad, const T *inp, int softmax_length) {
  int batch_idx = blockIdx.x * blockDim.y + threadIdx.y;
  int offset = batch_idx * softmax_length + threadIdx.x;

  grad += offset;
  inp += offset;

  T grad_reg[ITERATIONS];
  T inp_reg[ITERATIONS];
  float sum = 0.0;

#pragma unroll
  for (int i = 0; i < ITERATIONS; ++i) {
    int curr_idx = threadIdx.x + i * WARP_SIZE;
    if (curr_idx < softmax_length) {
      grad_reg[i] = grad[i * WARP_SIZE];
      inp_reg[i] = inp[i * WARP_SIZE];
      sum += (float)grad_reg[i] * (float)inp_reg[i];
    }
  }

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

  for (int i = 1; i < WARP_SIZE; i <<= 1) sum += g.shfl_xor(sum, i);

#pragma unroll
  for (int i = 0; i < ITERATIONS; ++i) {
    int curr_idx = threadIdx.x + i * WARP_SIZE;
    if (curr_idx < softmax_length)
      grad[i * WARP_SIZE] = (T)((float)inp_reg[i] * ((float)grad_reg[i] - sum));
  }
}

template <typename T>
void launch_attn_softmax_bw(T *out_grad, const T *soft_inp, int rows,
                            int softmax_len, cudaStream_t stream) {
  const int warps_per_block = 4;
  // rows = batch_size * nhead * from_len
  dim3 grid_dim(rows / warps_per_block);
  dim3 block_dim(WARP_SIZE, warps_per_block);

  if (softmax_len <= 32)
    ker_attn_softmax_bw<T, 1>
        <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, softmax_len);
  else if (softmax_len <= 64)
    ker_attn_softmax_bw<T, 2>
        <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, softmax_len);
  else if (softmax_len <= 128)
    ker_attn_softmax_bw<T, 4>
        <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, softmax_len);
  else if (softmax_len <= 256)
    ker_attn_softmax_bw<T, 8>
        <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, softmax_len);
  else if (softmax_len <= 384)
    ker_attn_softmax_bw<T, 12>
        <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, softmax_len);
  else if (softmax_len <= 512)
    ker_attn_softmax_bw<T, 16>
        <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, softmax_len);
  else if (softmax_len <= 768)
    ker_attn_softmax_bw<T, 24>
        <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, softmax_len);
  else if (softmax_len <= 1024)
    ker_attn_softmax_bw<T, 32>
        <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, softmax_len);
  else if (softmax_len <= 2048)
    ker_attn_softmax_bw<T, 64>
        <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, softmax_len);
  else
    throw std::runtime_error(
        std::string(
            "Special sequence length found in softmax backward, seq_len: ") +
        std::to_string(softmax_len));
}

template void launch_attn_softmax_bw<__half>(__half *out_grad,
                                             const __half *soft_inp, int rows,
                                             int softmax_len,
                                             cudaStream_t stream);
template void launch_attn_softmax_bw<float>(float *out_grad,
                                            const float *soft_inp, int rows,
                                            int softmax_len,
                                            cudaStream_t stream);
