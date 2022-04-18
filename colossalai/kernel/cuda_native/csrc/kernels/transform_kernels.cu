#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>

#include "kernels.h"

using namespace cub;

/**
@brief: transform_0213
Split the attention heads and reshape input
during backward progress of encoder self-attention

@thread
gridDim.x = batch_size
gridDim.y = seq_len
blockDim.x = min(hidden_dim, MAX_THREADS)

@param
input: [batch_size, seq_len, hidden_dim]
output: [batch_size, nhead, seq_len, head_dim]
batch_size: the size of the current batch
seq_len: the sequence length of the current batch
hidden_dim: dim of the hidden tensor
nhead: number of attention heads
*/

template <typename T>
__global__ void transform_0213(T *output, const T *input, int hidden_dim,
                               int head_dim);

template <>
__global__ void transform_0213<float>(float *output, const float *input,
                                      int hidden_dim, int head_dim) {
  int batch_id = blockIdx.x;
  int token_id = blockIdx.y;
  int seq_len = gridDim.y;
  int nhead = hidden_dim / head_dim;

  // [b, s, h]
  int src_offset = flat_3dim(batch_id, token_id, 0, seq_len, hidden_dim);
  // [b, nh, s, ad]
  int trg_offset =
      flat_4dim(batch_id, 0, token_id, 0, nhead, seq_len, head_dim);

  const float4 *input4 = reinterpret_cast<const float4 *>(input);
  float4 *res4 = reinterpret_cast<float4 *>(output);
  float4 vinput4;

  for (std::size_t i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
    vinput4 = input4[src_offset + i];

    int head_id = i / head_dim;
    int dim_id = i % head_dim;
    int cur_trg_offset = flat_3dim(head_id, 0, dim_id, seq_len, head_dim);
    res4[trg_offset + cur_trg_offset] = vinput4;
  }
}

template <>
__global__ void transform_0213<__half>(__half *output, const __half *input,
                                       int hidden_dim, int head_dim) {
  int batch_id = blockIdx.x;
  int token_id = blockIdx.y;
  int seq_len = gridDim.y;
  int nhead = hidden_dim / head_dim;

  // [b, s, h]
  int src_offset = flat_3dim(batch_id, token_id, 0, seq_len, hidden_dim);
  // [b, nh, s, ad]
  int trg_offset =
      flat_4dim(batch_id, 0, token_id, 0, nhead, seq_len, head_dim);

  const float4 *input4 = reinterpret_cast<const float4 *>(input);
  float4 *res4 = reinterpret_cast<float4 *>(output);
  float4 vinput4;

  for (std::size_t i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
    vinput4 = input4[src_offset + i];

    int head_id = i / head_dim;
    int dim_id = i % head_dim;
    int cur_trg_offset = flat_3dim(head_id, 0, dim_id, seq_len, head_dim);
    res4[trg_offset + cur_trg_offset] = vinput4;
  }
}

// [b, s, h] -> [b, nh, s, ad]
template <>
void launch_transform_0213<float>(float *output, const float *input,
                                  int batch_size, int seq_len, int hidden_dim,
                                  int nhead, cudaStream_t stream) {
  hidden_dim >>= 2;
  int head_dim = hidden_dim / nhead;

  dim3 grid_dim(batch_size, seq_len);
  dim3 block_dim(min(hidden_dim, MAX_THREADS));

  transform_0213<float>
      <<<grid_dim, block_dim, 0, stream>>>(output, input, hidden_dim, head_dim);
}

template <>
void launch_transform_0213<__half>(__half *output, const __half *input,
                                   int batch_size, int seq_len, int hidden_dim,
                                   int nhead, cudaStream_t stream) {
  hidden_dim >>= 3;
  int head_dim = hidden_dim / nhead;

  dim3 grid_dim(batch_size, seq_len);
  dim3 block_dim(min(hidden_dim, MAX_THREADS));

  transform_0213<__half>
      <<<grid_dim, block_dim, 0, stream>>>(output, input, hidden_dim, head_dim);
}

/**
@brief: bias_add_transform_20314
Add bias to input, transform from
[0, 1, 2, 3, 4] to [2, 0, 3, 1, 4]

@thread
gridDim.x = dim_0
gridDim.y = dim_1
gridDim.z = dim_2
blockDim.x = min(dim_3 * dim_4, MAX_THREADS)

@param
input: [dim_0, dim_1, dim_2, dim_3, dim_4]
bias: [dim_2, dim_3, dim_4]
output: [dim_2, dim_0, dim_3, dim_1, dim_4]
*/
template <typename T>
__global__ void bias_add_transform_20314(T *output, const T *input,
                                         const T *bias, int dim_3, int dim_4);

template <>
__global__ void
bias_add_transform_20314<float>(float *output, const float *input,
                                const float *bias, int dim_3, int dim_4) {
  int id0 = blockIdx.x;
  int id1 = blockIdx.y;
  int id2 = blockIdx.z;
  int dim_0 = gridDim.x;
  int dim_1 = gridDim.y;
  int dim_2 = gridDim.z;
  int dim_34 = dim_3 * dim_4;

  int src_offset = flat_4dim(id0, id1, id2, 0, dim_1, dim_2, dim_34);
  int trg_offset = flat_5dim(id2, id0, 0, id1, 0, dim_0, dim_3, dim_1, dim_4);
  int bias_offset = flat_2dim(id2, 0, dim_34);

  const float4 *qkv4 = reinterpret_cast<const float4 *>(input);
  const float4 *bias4 = reinterpret_cast<const float4 *>(bias);
  float4 *res4 = reinterpret_cast<float4 *>(output);
  float4 vqkv4;
  float4 vbias4;
  float4 vres4;

  for (std::size_t i = threadIdx.x; i < dim_34; i += blockDim.x) {
    vqkv4 = qkv4[src_offset + i];
    vbias4 = bias4[bias_offset + i];
    vres4.x = vqkv4.x + vbias4.x;
    vres4.y = vqkv4.y + vbias4.y;
    vres4.z = vqkv4.z + vbias4.z;
    vres4.w = vqkv4.w + vbias4.w;

    int id3 = i / dim_4;
    int id4 = i % dim_4;
    int cur_trg_offset = flat_3dim(id3, 0, id4, dim_1, dim_4);
    res4[trg_offset + cur_trg_offset] = vres4;
  }
}

template <>
__global__ void
bias_add_transform_20314<__half>(__half *output, const __half *input,
                                 const __half *bias, int dim_3, int dim_4) {
  int id0 = blockIdx.x;
  int id1 = blockIdx.y;
  int id2 = blockIdx.z;
  int dim_0 = gridDim.x;
  int dim_1 = gridDim.y;
  int dim_2 = gridDim.z;
  int dim_34 = dim_3 * dim_4;

  int src_offset = flat_4dim(id0, id1, id2, 0, dim_1, dim_2, dim_34);
  int trg_offset = flat_5dim(id2, id0, 0, id1, 0, dim_0, dim_3, dim_1, dim_4);
  int bias_offset = flat_2dim(id2, 0, dim_34);

  const float4 *qkv4 = reinterpret_cast<const float4 *>(input);
  const float4 *bias4 = reinterpret_cast<const float4 *>(bias);
  float4 *res4 = reinterpret_cast<float4 *>(output);
  float4 vqkv4;
  float4 vbias4;
  float4 vres4;
  __half2 *h2_qkv = reinterpret_cast<__half2 *>(&vqkv4);
  __half2 *h2_bias = reinterpret_cast<__half2 *>(&vbias4);
  __half2 *h2_res = reinterpret_cast<__half2 *>(&vres4);

  for (std::size_t i = threadIdx.x; i < dim_34; i += blockDim.x) {
    vqkv4 = qkv4[src_offset + i];
    vbias4 = bias4[bias_offset + i];
    h2_res[0] = __hadd2(h2_qkv[0], h2_bias[0]);
    h2_res[1] = __hadd2(h2_qkv[1], h2_bias[1]);
    h2_res[2] = __hadd2(h2_qkv[2], h2_bias[2]);
    h2_res[3] = __hadd2(h2_qkv[3], h2_bias[3]);

    int id3 = i / dim_4;
    int id4 = i % dim_4;
    int cur_trg_offset = flat_3dim(id3, 0, id4, dim_1, dim_4);
    res4[trg_offset + cur_trg_offset] = vres4;
  }
}

// [b, s, 3, h] -> [3, b, nh, s, ad]
template <>
void launch_bias_add_transform_20314<float>(float *output, const float *input,
                                            const float *bias, int dim_0,
                                            int dim_1, int dim_2, int dim_3,
                                            int dim_4, cudaStream_t stream) {
  dim_4 >>= 2;

  dim3 grid_dim(dim_0, dim_1, dim_2);
  dim3 block_dim(min(dim_3 * dim_4, MAX_THREADS));

  bias_add_transform_20314<float>
      <<<grid_dim, block_dim, 0, stream>>>(output, input, bias, dim_3, dim_4);
}

template <>
void launch_bias_add_transform_20314<__half>(__half *output,
                                             const __half *input,
                                             const __half *bias, int dim_0,
                                             int dim_1, int dim_2, int dim_3,
                                             int dim_4, cudaStream_t stream) {
  dim_4 >>= 3;

  dim3 grid_dim(dim_0, dim_1, dim_2);
  dim3 block_dim(min(dim_3 * dim_4, MAX_THREADS));

  bias_add_transform_20314<__half>
      <<<grid_dim, block_dim, 0, stream>>>(output, input, bias, dim_3, dim_4);
}

/**
@brief: transform4d_0213
Reshape the input matrix to merge the heads

@thread
gridDim.x = (num_all + max_block_thread - 1) / max_block_thread
blockDim.x = max_block_thread

@param
input: [trans_count, batch_size, nhead, seq_len, head_dim]
output: [batch_size, seq_len, trans_count, nhead, head_dim]
batch_size: the size of the current batch
seq_len: the sequence length of the current batch
hidden_dim: dim of the hidden tensor
nhead: number of attention heads
trans_count: 1 or 3, the count of matrice need to be transformed
*/
template <typename T>
__global__ void transform4d_0213(T *output, const T *input, int batch_size,
                                 int seq_len, int trans_count, int nhead,
                                 int head_dim, int num_all) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= num_all) {
    return;
  }
  int trans_id, batch_id, head_id, token_id, dim_id;
  decompose_5dim(offset, batch_size, nhead, seq_len, head_dim, &trans_id,
                 &batch_id, &head_id, &token_id, &dim_id);
  // [b, s, tc, nh, ad]
  int trg_offset = flat_5dim(batch_id, token_id, trans_id, head_id, dim_id,
                             seq_len, trans_count, nhead, head_dim);

  const float4 *input4 = reinterpret_cast<const float4 *>(input);
  float4 *res4 = reinterpret_cast<float4 *>(output);
  res4[trg_offset] = input4[offset];
}

// [tc, b, nh, s, ad] -> [b, s, tc, nh, ad]
template <>
void launch_transform4d_0213<float>(float *output, const float *input,
                                    int batch_size, int seq_len, int hidden_dim,
                                    int nhead, int trans_count,
                                    cudaStream_t stream) {
  hidden_dim >>= 2;
  int head_dim = hidden_dim / nhead;
  int num_all = batch_size * seq_len * trans_count * hidden_dim;
  int nblock = (num_all + MAX_THREADS - 1) / MAX_THREADS;

  transform4d_0213<float><<<nblock, MAX_THREADS, 0, stream>>>(
      output, input, batch_size, seq_len, trans_count, nhead, head_dim,
      num_all);
}

template <>
void launch_transform4d_0213<__half>(__half *output, const __half *input,
                                     int batch_size, int seq_len,
                                     int hidden_dim, int nhead, int trans_count,
                                     cudaStream_t stream) {
  hidden_dim >>= 3;
  int head_dim = hidden_dim / nhead;
  int num_all = batch_size * seq_len * trans_count * hidden_dim;
  int nblock = (num_all + MAX_THREADS - 1) / MAX_THREADS;

  transform4d_0213<__half><<<nblock, MAX_THREADS, 0, stream>>>(
      output, input, batch_size, seq_len, trans_count, nhead, head_dim,
      num_all);
}
