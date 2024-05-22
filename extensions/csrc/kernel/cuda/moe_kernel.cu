#include <cuda.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cub/cub.cuh>

#include "funcs/reduce_function.h"

using colossalAI::funcs::block_reduce;
using colossalAI::funcs::ReduceType;

template <typename T, int block_size, int pack_size>
__device__ void moe_dpch_one_fwd(T *src_row, T *dst_row, const int cols) {
  assert(cols % pack_size == 0);
  const int bpack_size = block_size * pack_size;

  typedef cub::BlockLoad<T, block_size, pack_size, cub::BLOCK_LOAD_VECTORIZE>
      BlockLoad;
  __shared__ typename BlockLoad::TempStorage ts_load;

  typedef cub::BlockStore<T, block_size, pack_size, cub::BLOCK_STORE_VECTORIZE>
      BlockStore;
  __shared__ typename BlockStore::TempStorage ts_store;

  int tps = threadIdx.x * pack_size;
  T pack[pack_size];
  for (int idx = 0; idx + tps < cols; idx += bpack_size) {
    BlockLoad(ts_load).Load(src_row + idx, pack);
    BlockStore(ts_store).Store(dst_row + idx, pack);
  }
}

template <typename T, int block_size, int pack_size>
__device__ void moe_dpch_one_bwd(T *src_row, T *dst_row, const int cols) {
  assert(cols % pack_size == 0);
  const int bpack_size = block_size * pack_size;

  typedef cub::BlockLoad<T, block_size, pack_size, cub::BLOCK_LOAD_VECTORIZE>
      BlockLoad;
  __shared__ typename BlockLoad::TempStorage ts_load;

  typedef cub::BlockStore<T, block_size, pack_size, cub::BLOCK_STORE_VECTORIZE>
      BlockStore;
  __shared__ typename BlockStore::TempStorage ts_store;

  int tps = threadIdx.x * pack_size;
  T pack[pack_size];
  for (int idx = 0; idx + tps < cols; idx += bpack_size) {
    BlockLoad(ts_load).Load(dst_row + idx, pack);
    BlockStore(ts_store).Store(src_row + idx, pack);
  }
}

template <typename T, int block_size, int pack_size>
__device__ void moe_dpch_two_fwd(T *src_row, T *dst_row1, T *dst_row2,
                                 const int cols) {
  assert(cols % pack_size == 0);
  const int bpack_size = block_size * pack_size;

  typedef cub::BlockLoad<T, block_size, pack_size, cub::BLOCK_LOAD_VECTORIZE>
      BlockLoad;
  __shared__ typename BlockLoad::TempStorage ts_load;

  typedef cub::BlockStore<T, block_size, pack_size, cub::BLOCK_STORE_VECTORIZE>
      BlockStore;
  __shared__ typename BlockStore::TempStorage ts_store;

  int tps = threadIdx.x * pack_size;
  T pack[pack_size];
  for (int idx = 0; idx + tps < cols; idx += bpack_size) {
    BlockLoad(ts_load).Load(src_row + idx, pack);
    BlockStore(ts_store).Store(dst_row1 + idx, pack);
    BlockStore(ts_store).Store(dst_row2 + idx, pack);
  }
}

template <typename T, int block_size, int pack_size>
__device__ void moe_dpch_two_bwd(T *src_row, T *dst_row1, T *dst_row2,
                                 const int cols) {
  assert(cols % pack_size == 0);
  const int bpack_size = block_size * pack_size;

  typedef cub::BlockLoad<T, block_size, pack_size, cub::BLOCK_LOAD_VECTORIZE>
      BlockLoad;
  __shared__ typename BlockLoad::TempStorage ts_load;

  typedef cub::BlockStore<T, block_size, pack_size, cub::BLOCK_STORE_VECTORIZE>
      BlockStore;
  __shared__ typename BlockStore::TempStorage ts_store;

  int tps = threadIdx.x * pack_size;
  T pack1[pack_size], pack2[pack_size];
  for (int idx = 0; idx + tps < cols; idx += bpack_size) {
    BlockLoad(ts_load).Load(dst_row1 + idx, pack1);
    BlockLoad(ts_load).Load(dst_row2 + idx, pack2);

#pragma unroll
    for (int i = 0; i < pack_size; ++i) {
      pack1[i] += pack2[i];
    }

    BlockStore(ts_store).Store(src_row + idx, pack1);
  }
}

template <typename T, int block_size, int pack_size>
__device__ void moe_cb_one_fwd(T *src_row, T *dst_row, const T weight,
                               const int cols) {
  assert(cols % pack_size == 0);
  const int bpack_size = block_size * pack_size;

  typedef cub::BlockLoad<T, block_size, pack_size, cub::BLOCK_LOAD_VECTORIZE>
      BlockLoad;
  __shared__ typename BlockLoad::TempStorage ts_load;

  typedef cub::BlockStore<T, block_size, pack_size, cub::BLOCK_STORE_VECTORIZE>
      BlockStore;
  __shared__ typename BlockStore::TempStorage ts_store;

  int tps = threadIdx.x * pack_size;
  T pack[pack_size];
  for (int idx = 0; idx + tps < cols; idx += bpack_size) {
    BlockLoad(ts_load).Load(src_row + idx, pack);

#pragma unroll
    for (int i = 0; i < pack_size; ++i) {
      pack[i] *= weight;
    }

    BlockStore(ts_store).Store(dst_row + idx, pack);
  }
}

template <typename T, int block_size, int pack_size>
__device__ void moe_cb_one_bwd(T *src_row, T *dst_row, T *tks_row,
                               T *weight_grad, const T weight, const int cols) {
  assert(cols % pack_size == 0);
  const int bpack_size = block_size * pack_size;

  typedef cub::BlockLoad<T, block_size, pack_size, cub::BLOCK_LOAD_VECTORIZE>
      BlockLoad;
  __shared__ typename BlockLoad::TempStorage ts_load;

  typedef cub::BlockStore<T, block_size, pack_size, cub::BLOCK_STORE_VECTORIZE>
      BlockStore;
  __shared__ typename BlockStore::TempStorage ts_store;

  int tps = threadIdx.x * pack_size;
  T grad[pack_size], tokens[pack_size];
  float thread_sum = 0;
  for (int idx = 0; idx + tps < cols; idx += bpack_size) {
    BlockLoad(ts_load).Load(dst_row + idx, grad);
    BlockLoad(ts_load).Load(tks_row + idx, tokens);

#pragma unroll
    for (int i = 0; i < pack_size; ++i) {
      thread_sum += grad[i] * tokens[i];
      grad[i] *= weight;
    }

    BlockStore(ts_store).Store(src_row + idx, grad);
  }
  block_reduce<float, ReduceType::kSum, 1>(&thread_sum);

  if (threadIdx.x == 0) *weight_grad = static_cast<T>(thread_sum);
}

template <typename T, int block_size, int pack_size>
__device__ void moe_cb_two_fwd(T *src_row1, T *src_row2, T *dst_row,
                               const T weight1, const T weight2,
                               const int cols) {
  assert(cols % pack_size == 0);
  const int bpack_size = block_size * pack_size;

  typedef cub::BlockLoad<T, block_size, pack_size, cub::BLOCK_LOAD_VECTORIZE>
      BlockLoad;
  __shared__ typename BlockLoad::TempStorage ts_load;

  typedef cub::BlockStore<T, block_size, pack_size, cub::BLOCK_STORE_VECTORIZE>
      BlockStore;
  __shared__ typename BlockStore::TempStorage ts_store;

  int tps = threadIdx.x * pack_size;
  T pack1[pack_size], pack2[pack_size];
  for (int idx = 0; idx + tps < cols; idx += bpack_size) {
    BlockLoad(ts_load).Load(src_row1 + idx, pack1);
    BlockLoad(ts_load).Load(src_row2 + idx, pack2);

#pragma unroll
    for (int i = 0; i < pack_size; ++i) {
      pack1[i] = pack1[i] * weight1 + pack2[i] * weight2;
    }

    BlockStore(ts_store).Store(dst_row + idx, pack1);
  }
}

template <typename T, int block_size, int pack_size>
__device__ void moe_cb_two_bwd(T *src_row1, T *src_row2, T *dst_row,
                               T *tks_row1, T *tks_row2, T *weight_grad1,
                               T *weight_grad2, const T weight1,
                               const T weight2, const int cols) {
  assert(cols % pack_size == 0);
  const int bpack_size = block_size * pack_size;

  typedef cub::BlockLoad<T, block_size, pack_size, cub::BLOCK_LOAD_VECTORIZE>
      BlockLoad;
  __shared__ typename BlockLoad::TempStorage ts_load;

  typedef cub::BlockStore<T, block_size, pack_size, cub::BLOCK_STORE_VECTORIZE>
      BlockStore;
  __shared__ typename BlockStore::TempStorage ts_store;

  int tps = threadIdx.x * pack_size;
  T grad[pack_size], tokens1[pack_size], tokens2[pack_size], sgrad1[pack_size],
      sgrad2[pack_size];
  float thread_sum[2] = {0, 0};
  for (int idx = 0; idx + tps < cols; idx += bpack_size) {
    BlockLoad(ts_load).Load(dst_row + idx, grad);
    BlockLoad(ts_load).Load(tks_row1 + idx, tokens1);
    BlockLoad(ts_load).Load(tks_row2 + idx, tokens2);

#pragma unroll
    for (int i = 0; i < pack_size; ++i) {
      thread_sum[0] += grad[i] * tokens1[i];
      thread_sum[1] += grad[i] * tokens2[i];
      sgrad1[i] = weight1 * grad[i];
      sgrad2[i] = weight2 * grad[i];
    }

    BlockStore(ts_store).Store(src_row1 + idx, sgrad1);
    BlockStore(ts_store).Store(src_row2 + idx, sgrad2);
  }

  block_reduce<float, ReduceType::kSum, 2>(thread_sum);

  if (threadIdx.x == 0)
    *weight_grad1 = static_cast<T>(thread_sum[0]);
  else if (threadIdx.x == 1)
    *weight_grad2 = static_cast<T>(thread_sum[1]);
}

// DISPATCH KERNELS --------------------------------

template <typename T, int block_size, int pack_size>
__device__ void moe_dpch_fwd_selector(T *src_row, T *dst_row1, T *dst_row2,
                                      const int cols, const int indicator1,
                                      const int indicator2) {
  if (indicator1 != 0 && indicator2 != 0)
    moe_dpch_two_fwd<T, block_size, pack_size>(src_row, dst_row1, dst_row2,
                                               cols);
  else if (indicator1 != 0)
    moe_dpch_one_fwd<T, block_size, pack_size>(src_row, dst_row1, cols);
  else if (indicator2 != 0)
    moe_dpch_one_fwd<T, block_size, pack_size>(src_row, dst_row2, cols);
  else
    return;
}

template <typename T, int block_size, int pack_size>
__device__ void moe_dpch_bwd_selector(T *src_row, T *dst_row1, T *dst_row2,
                                      const int cols, const int indicator1,
                                      const int indicator2) {
  if (indicator1 != 0 && indicator2 != 0)
    moe_dpch_two_bwd<T, block_size, pack_size>(src_row, dst_row1, dst_row2,
                                               cols);
  else if (indicator1 != 0)
    moe_dpch_one_bwd<T, block_size, pack_size>(src_row, dst_row1, cols);
  else if (indicator2 != 0)
    moe_dpch_one_bwd<T, block_size, pack_size>(src_row, dst_row2, cols);
  else
    return;
}

template <typename T, int block_size, int pack_size>
__global__ void moe_dpch_fwd_kernel(T *batch_tokens, T *expert_input,
                                    int *mask1, int *mask2, int *dest1,
                                    int *dest2, const int h) {
  int row = blockIdx.x;
  int indicator2 = mask2 == nullptr ? 0 : mask2[row];
  moe_dpch_fwd_selector<T, block_size, pack_size>(
      batch_tokens + (row * h), expert_input + (dest1[row] * h),
      expert_input + (dest2[row] * h), h, mask1[row], indicator2);
}

template <typename T, int block_size, int pack_size>
__global__ void moe_dpch_bwd_kernel(T *tokens_grad, T *expert_grad, int *mask1,
                                    int *mask2, int *dest1, int *dest2,
                                    const int h) {
  int row = blockIdx.x;
  int indicator2 = mask2 == nullptr ? 0 : mask2[row];
  moe_dpch_bwd_selector<T, block_size, pack_size>(
      tokens_grad + (row * h), expert_grad + (dest1[row] * h),
      expert_grad + (dest2[row] * h), h, mask1[row], indicator2);
}

// COMBINE KERNELS --------------------------------

template <typename T, int block_size, int pack_size>
__device__ void moe_cb_fwd_selector(T *src_row1, T *src_row2, T *dst_row,
                                    const int cols, const T weight1,
                                    const T weight2, const int indicator1,
                                    const int indicator2) {
  if (indicator1 != 0 && indicator2 != 0)
    moe_cb_two_fwd<T, block_size, pack_size>(src_row1, src_row2, dst_row,
                                             weight1, weight2, cols);
  else if (indicator1 != 0)
    moe_cb_one_fwd<T, block_size, pack_size>(src_row1, dst_row, weight1, cols);
  else if (indicator2 != 0)
    moe_cb_one_fwd<T, block_size, pack_size>(src_row2, dst_row, weight2, cols);
  else
    return;
}

template <typename T, int block_size, int pack_size>
__device__ void moe_cb_bwd_selector(T *src_row1, T *src_row2, T *dst_row,
                                    const int cols, T *tks_row1, T *tks_row2,
                                    T *wt_grad1, T *wt_grad2, const T weight1,
                                    const T weight2, const int indicator1,
                                    const int indicator2) {
  if (indicator1 != 0 && indicator2 != 0)
    moe_cb_two_bwd<T, block_size, pack_size>(src_row1, src_row2, dst_row,
                                             tks_row1, tks_row2, wt_grad1,
                                             wt_grad2, weight1, weight2, cols);
  else if (indicator1 != 0)
    moe_cb_one_bwd<T, block_size, pack_size>(src_row1, dst_row, tks_row1,
                                             wt_grad1, weight1, cols);
  else if (indicator2 != 0)
    moe_cb_one_bwd<T, block_size, pack_size>(src_row2, dst_row, tks_row2,
                                             wt_grad2, weight2, cols);
  else
    return;
}

template <typename T, int block_size, int pack_size>
__global__ void moe_cb_fwd_kernel(T *expert_tokens, T *combine_tokens,
                                  T *logits, int *mask1, int *mask2, int *dest1,
                                  int *dest2, const int e, const int c,
                                  const int h) {
  int row = blockIdx.x, eid1 = dest1[row] / c, eid2 = dest2[row] / c;
  int indicator2 = mask2 == nullptr ? 0 : mask2[row];
  T *row_log = logits + (row * e);
  moe_cb_fwd_selector<T, block_size, pack_size>(
      expert_tokens + (dest1[row] * h), expert_tokens + (dest2[row] * h),
      combine_tokens + (row * h), h, row_log[eid1], row_log[eid2], mask1[row],
      indicator2);
}

template <typename T, int block_size, int pack_size>
__global__ void moe_cb_bwd_kernel(T *tokens_grad, T *expert_grad, T *tks,
                                  T *logits, T *logits_grad, int *mask1,
                                  int *mask2, int *dest1, int *dest2,
                                  const int e, const int c, const int h) {
  int row = blockIdx.x, eid1 = dest1[row] / c, eid2 = dest2[row] / c;
  int indicator2 = mask2 == nullptr ? 0 : mask2[row];
  T *row_log = logits + (row * e), *row_grad = logits_grad + (row * e);
  moe_cb_bwd_selector<T, block_size, pack_size>(
      expert_grad + (dest1[row] * h), expert_grad + (dest2[row] * h),
      tokens_grad + (row * h), h, tks + (dest1[row] * h),
      tks + (dest2[row] * h), row_grad + eid1, row_grad + eid2, row_log[eid1],
      row_log[eid2], mask1[row], indicator2);
}

// CUMSUM KERNEL --------------------------------

template <int block_size, int pack_size>
__global__ void cumsum_kernel(int *inputs, int *outputs, const int s,
                              const int e) {
  assert(s % pack_size == 0);
  constexpr int bpack_size = block_size * pack_size;
  int tid = threadIdx.x, bid = blockIdx.x, tps = tid * pack_size, last_sum = -1;
  __shared__ int temp[block_size + 1];
  int pack[pack_size];

  for (int idx = 0; idx < s; idx += bpack_size) {
    int offset = 1;

    if (idx + tps < s) {
      temp[tid] = inputs[tps * e + bid];
#pragma unroll
      for (int i = 1; i < pack_size; ++i) {
        pack[i] = inputs[(tps + i) * e + bid];
      }
#pragma unroll
      for (int i = 1; i < pack_size; ++i) {
        temp[tid] += pack[i];
      }
    }

    for (int i = block_size >> 1; i > 0; i >>= 1) {
      __syncthreads();
      if (tid < i) {
        int j = offset * (2 * tid + 1) - 1;
        temp[j + offset] += temp[j];
      }
      offset <<= 1;
    }

    if (tid == 0) {
      temp[block_size] = temp[block_size - 1];
      temp[block_size - 1] = 0;
    }

    for (int i = 1; i < block_size; i <<= 1) {
      offset >>= 1;
      __syncthreads();
      if (tid < i) {
        int j = offset * (2 * tid + 1) - 1, k = j + offset, ts = temp[j];
        temp[j] = temp[k];
        temp[k] += ts;
      }
    }
    __syncthreads();

    if (tid == 0) temp[0] = temp[block_size];
    __syncthreads();

    if (idx + tps < s) {
      temp[tid + 1] += last_sum;
#pragma unroll
      for (int i = pack_size - 1; i > 0; --i) {
        outputs[(tps + i) * e + bid] = temp[tid + 1];
        temp[tid + 1] -= pack[i];
      }
      outputs[tps * e + bid] = temp[tid + 1];
    }
    __syncthreads();

    last_sum += temp[0];
    inputs += bpack_size * e;
    outputs += bpack_size * e;
  }
}

// LAUNCH FUNCTIONS --------------------------------

template <typename T>
void moe_dpch_fwd_launch(T *batch_tokens, T *expert_input, int *mask1,
                         int *mask2, int *dest1, int *dest2, const int s,
                         const int h) {
  if (h < 256)
    moe_dpch_fwd_kernel<T, 32, 4>
        <<<s, 32>>>(batch_tokens, expert_input, mask1, mask2, dest1, dest2, h);
  else if (h < 512)
    moe_dpch_fwd_kernel<T, 32, 8>
        <<<s, 32>>>(batch_tokens, expert_input, mask1, mask2, dest1, dest2, h);
  else if (h < 1024)
    moe_dpch_fwd_kernel<T, 32, 16>
        <<<s, 32>>>(batch_tokens, expert_input, mask1, mask2, dest1, dest2, h);
  else if (h < 2048)
    moe_dpch_fwd_kernel<T, 64, 16>
        <<<s, 64>>>(batch_tokens, expert_input, mask1, mask2, dest1, dest2, h);
  else
    moe_dpch_fwd_kernel<T, 128, 16>
        <<<s, 128>>>(batch_tokens, expert_input, mask1, mask2, dest1, dest2, h);
}

template <typename T>
void moe_dpch_bwd_launch(T *tokens_grad, T *expert_grad, int *mask1, int *mask2,
                         int *dest1, int *dest2, const int s, const int h) {
  if (h < 256)
    moe_dpch_bwd_kernel<T, 32, 4>
        <<<s, 32>>>(tokens_grad, expert_grad, mask1, mask2, dest1, dest2, h);
  else if (h < 512)
    moe_dpch_bwd_kernel<T, 32, 8>
        <<<s, 32>>>(tokens_grad, expert_grad, mask1, mask2, dest1, dest2, h);
  else if (h < 1024)
    moe_dpch_bwd_kernel<T, 32, 16>
        <<<s, 32>>>(tokens_grad, expert_grad, mask1, mask2, dest1, dest2, h);
  else if (h < 2048)
    moe_dpch_bwd_kernel<T, 64, 16>
        <<<s, 64>>>(tokens_grad, expert_grad, mask1, mask2, dest1, dest2, h);
  else
    moe_dpch_bwd_kernel<T, 128, 16>
        <<<s, 128>>>(tokens_grad, expert_grad, mask1, mask2, dest1, dest2, h);
}

template <typename T>
void moe_cb_fwd_launch(T *expert_tokens, T *combine_tokens, T *logits,
                       int *mask1, int *mask2, int *dest1, int *dest2,
                       const int s, const int e, const int c, const int h) {
  if (h < 256)
    moe_cb_fwd_kernel<T, 32, 4><<<s, 32>>>(expert_tokens, combine_tokens,
                                           logits, mask1, mask2, dest1, dest2,
                                           e, c, h);
  else if (h < 512)
    moe_cb_fwd_kernel<T, 32, 8><<<s, 32>>>(expert_tokens, combine_tokens,
                                           logits, mask1, mask2, dest1, dest2,
                                           e, c, h);
  else if (h < 1024)
    moe_cb_fwd_kernel<T, 32, 16><<<s, 32>>>(expert_tokens, combine_tokens,
                                            logits, mask1, mask2, dest1, dest2,
                                            e, c, h);
  else if (h < 2048)
    moe_cb_fwd_kernel<T, 64, 16><<<s, 64>>>(expert_tokens, combine_tokens,
                                            logits, mask1, mask2, dest1, dest2,
                                            e, c, h);
  else
    moe_cb_fwd_kernel<T, 128, 16><<<s, 128>>>(expert_tokens, combine_tokens,
                                              logits, mask1, mask2, dest1,
                                              dest2, e, c, h);
}

template <typename T>
void moe_cb_bwd_launch(T *tokens_grad, T *expert_grad, T *tks, T *logits,
                       T *logits_grad, int *mask1, int *mask2, int *dest1,
                       int *dest2, const int s, const int e, const int c,
                       const int h) {
  if (h < 256)
    moe_cb_bwd_kernel<T, 32, 4><<<s, 32>>>(tokens_grad, expert_grad, tks,
                                           logits, logits_grad, mask1, mask2,
                                           dest1, dest2, e, c, h);
  else  // if (h < 512)
    moe_cb_bwd_kernel<T, 64, 4><<<s, 64>>>(tokens_grad, expert_grad, tks,
                                           logits, logits_grad, mask1, mask2,
                                           dest1, dest2, e, c, h);
  // else if (h < 1024)
  //     moe_cb_bwd_kernel<T, 128, 4><<<s, 128>>>
  //         (tokens_grad, expert_grad, tks, logits, logits_grad, mask1, mask2,
  //         dest1, dest2, e, c, h);
  // else
  //     moe_cb_bwd_kernel<T, 256, 4><<<s, 256>>>
  //         (tokens_grad, expert_grad, tks, logits, logits_grad, mask1, mask2,
  //         dest1, dest2, e, c, h);
}

void cumsum_launch(int *inputs, int *outputs, const int s, const int e) {
  if (s <= 256)
    cumsum_kernel<256, 1><<<e, 256>>>(inputs, outputs, s, e);
  else if (s <= 512)
    cumsum_kernel<512, 1><<<e, 512>>>(inputs, outputs, s, e);
  else if (s <= 1024)
    cumsum_kernel<1024, 1><<<e, 1024>>>(inputs, outputs, s, e);
  else if (s <= 2048)
    cumsum_kernel<1024, 2><<<e, 1024>>>(inputs, outputs, s, e);
  else
    cumsum_kernel<1024, 4><<<e, 1024>>>(inputs, outputs, s, e);
}

// API FUNCTIONS --------------------------------

#define DISPATCH_FLOAT_AND_HALF_MOE(TYPE, NAME, ...)                   \
  switch (TYPE) {                                                      \
    case at::ScalarType::Float: {                                      \
      using scalar_t = float;                                          \
      __VA_ARGS__;                                                     \
      break;                                                           \
    }                                                                  \
    case at::ScalarType::Half: {                                       \
      using scalar_t = at::Half;                                       \
      __VA_ARGS__;                                                     \
      break;                                                           \
    }                                                                  \
    default:                                                           \
      AT_ERROR(#NAME, " not implemented yet for specific data type."); \
  }

torch::Tensor moe_dispatch_cuda_forward(int s, int ec, int h,
                                        torch::Tensor batch_tokens,
                                        torch::Tensor mask,
                                        torch::Tensor dest_idx) {
  assert(h % 16 == 0);
  auto res = torch::zeros(
      {ec, h},
      torch::dtype(batch_tokens.dtype()).device(batch_tokens.device()));
  auto k = mask.size(0);

  DISPATCH_FLOAT_AND_HALF_MOE(
      batch_tokens.scalar_type(), "moe dispatch forward",
      moe_dpch_fwd_launch<scalar_t>(
          batch_tokens.data_ptr<scalar_t>(), res.data_ptr<scalar_t>(),
          mask[0].data_ptr<int>(), k == 1 ? nullptr : mask[1].data_ptr<int>(),
          dest_idx[0].data_ptr<int>(),
          k == 1 ? dest_idx[0].data_ptr<int>() : dest_idx[1].data_ptr<int>(), s, h));

  return res;
}

torch::Tensor moe_dispatch_cuda_backward(int s, int ec, int h,
                                         torch::Tensor expert_grad,
                                         torch::Tensor mask,
                                         torch::Tensor dest_idx) {
  assert(h % 16 == 0);
  auto res = torch::zeros(
      {s, h}, torch::dtype(expert_grad.dtype()).device(expert_grad.device()));
  auto k = mask.size(0);

  DISPATCH_FLOAT_AND_HALF_MOE(
      expert_grad.scalar_type(), "moe dispatch backward",
      moe_dpch_bwd_launch<scalar_t>(
          res.data_ptr<scalar_t>(), expert_grad.data_ptr<scalar_t>(),
          mask[0].data_ptr<int>(), k == 1 ? nullptr : mask[1].data_ptr<int>(),
          dest_idx[0].data_ptr<int>(),
          k == 1 ? dest_idx[0].data_ptr<int>() : dest_idx[1].data_ptr<int>(), s, h));

  return res;
}

torch::Tensor moe_combine_cuda_forward(int s, int e, int c, int h,
                                       torch::Tensor expert_tokens,
                                       torch::Tensor logits, torch::Tensor mask,
                                       torch::Tensor dest_idx) {
  assert(h % 16 == 0);
  assert(expert_tokens.dtype() == logits.dtype());

  auto res = torch::zeros(
      {s, h},
      torch::dtype(expert_tokens.dtype()).device(expert_tokens.device()));
  auto k = mask.size(0);

  DISPATCH_FLOAT_AND_HALF_MOE(
      expert_tokens.scalar_type(), "moe combine forward",
      moe_cb_fwd_launch<scalar_t>(
          expert_tokens.data_ptr<scalar_t>(), res.data_ptr<scalar_t>(),
          logits.data_ptr<scalar_t>(), mask[0].data_ptr<int>(),
          k == 1 ? nullptr : mask[1].data_ptr<int>(), dest_idx[0].data_ptr<int>(),
          k == 1 ? dest_idx[0].data_ptr<int>() : dest_idx[1].data_ptr<int>(), s, e, c,
          h));

  return res;
}

std::vector<torch::Tensor> moe_combine_cuda_backward(
    int s, int e, int c, int h, torch::Tensor tokens_grad,
    torch::Tensor expert_tokens, torch::Tensor logits, torch::Tensor mask,
    torch::Tensor dest_idx) {
  assert(h % 16 == 0);
  assert(tokens_grad.dtype() == expert_tokens.dtype());
  assert(expert_tokens.dtype() == logits.dtype());

  auto egrad = torch::zeros(
           {e * c, h},
           torch::dtype(tokens_grad.dtype()).device(tokens_grad.device())),
       wgrad = torch::zeros(
           {s, e}, torch::dtype(logits.dtype()).device(logits.device()));
  auto k = mask.size(0);

  DISPATCH_FLOAT_AND_HALF_MOE(
      tokens_grad.scalar_type(), "moe combine backward",
      moe_cb_bwd_launch<scalar_t>(
          tokens_grad.data_ptr<scalar_t>(), egrad.data_ptr<scalar_t>(),
          expert_tokens.data_ptr<scalar_t>(), logits.data_ptr<scalar_t>(),
          wgrad.data_ptr<scalar_t>(), mask[0].data_ptr<int>(),
          k == 1 ? nullptr : mask[1].data_ptr<int>(), dest_idx[0].data_ptr<int>(),
          k == 1 ? dest_idx[0].data_ptr<int>() : dest_idx[1].data_ptr<int>(), s, e, c,
          h));

  return {egrad, wgrad};
}

torch::Tensor cumsum_sub_one_in_dim0(torch::Tensor mask) {
  assert(mask.dim() == 2);
  assert(mask.dtype() == torch::kInt32);

  const int s = mask.size(0), e = mask.size(1);
  auto res =
      torch::empty({s, e}, torch::dtype(torch::kInt32).device(mask.device()));
  cumsum_launch(mask.data_ptr<int>(), res.data_ptr<int>(), s, e);

  return res;
}
