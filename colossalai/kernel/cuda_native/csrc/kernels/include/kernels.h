#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

#include <stdexcept>

#define MAX_THREADS 1024
#define WARP_SIZE 32

enum class ActivationType { kRelu, kGelu };

void launch_curand_init(int total_count, int dim, cudaStream_t stream);

template <typename T>
void launch_layer_norm(T *ln_res, T *vars, T *means, const T *inp,
                       const T *scale, const T *bias, int batch_size,
                       int hidden_dim, cudaStream_t stream);

template <typename T>
void launch_ln_bw(T *gamma_grad, T *betta_grad, T *inp_grad, const T *out_grad,
                  const T *residual_grad, const T *inp_or_out, const T *gamma,
                  const T *betta, const T *vars, const T *means, int batch,
                  int hidden_dim, cudaStream_t stream[2]);

template <typename T>
void launch_attn_softmax(T *vals, const T *attn_mask, int batch_size, int heads,
                         int from_len, int to_len, bool mask_future,
                         cudaStream_t stream);

template <typename T>
void launch_attn_softmax_bw(T *out_grad, const T *soft_inp, int rows,
                            int softmax_len, cudaStream_t stream);

// [b, s, h] -> [b, nh, s, ad]
template <typename T>
void launch_transform_0213(T *output, const T *vals, int batch_size,
                           int seq_length, int hidden_dim, int nhead,
                           cudaStream_t stream);

// [b, s, 3, h] -> [3, b, nh, s, ad]
template <typename T>
void launch_bias_add_transform_20314(T *output, const T *input, const T *bias,
                                     int dim_0, int dim_1, int dim_2, int dim_3,
                                     int dim_4, cudaStream_t stream);

// [tc, b, nh, s, ad] -> [b, s, tc, nh, ad]
template <typename T>
void launch_transform4d_0213(T *output, const T *vals, int batch_size,
                             int seq_len, int hidden_dim, int nhead,
                             int trans_count, cudaStream_t stream);

template <typename T>
void launch_ls_dropout(T *out, const T *vals, uint8_t *mask, int total_count,
                       float ratio, cudaStream_t stream, bool backward = false);

template <typename T>
void launch_ls_dropout_res_bias(T *out, const T *vals, uint8_t *mask,
                                const T *bias, const T *residual,
                                int total_count, int dim, float ratio,
                                cudaStream_t stream);

template <ActivationType, typename T>
void launch_ls_dropout_act_bias(T *out, const T *vals, uint8_t *mask,
                                const T *bias, int total_count, int dim,
                                float ratio, cudaStream_t stream);

template <typename T>
void launch_ls_dropout_bias_bwd(T *in_grad, T *bias_grad, const T *out_grad,
                                const uint8_t *mask, int row_size, int dim,
                                float ratio, cudaStream_t stream);

template <ActivationType act_type, typename T>
void launch_ls_dropout_act_bias_bwd(T *in_grad, T *bias_grad, const T *input,
                                    const T *bias, const T *out_grad,
                                    const uint8_t *mask, int row_size, int dim,
                                    float ratio, cudaStream_t stream);

template <typename T>
void launch_fuse_transpose_bias_kernel(const T *inp, T *out, int rows, int cols,
                                       cudaStream_t stream);

void launch_param_update(const float *input, __half *output, int size,
                         cudaStream_t stream);

template <typename T>
void launch_concat3_dim1(const T *inp1, const T *inp2, T *output, int sz0,
                         int sz2, int sz1_1, int sz1_2, cudaStream_t stream);

template <typename T>
void launch_fused_add2(T *out, const T *inp1, const T *inp2, int batch_size,
                       int seq_len, int hidden_size, cudaStream_t &stream);

template <typename T>
void launch_cross_entropy_fw(const T *inputs_ptr, const int *targets_ptr,
                             float *outputs_ptr, float *nll_loss_ptr,
                             float *loss_buffer, const int padding_idx,
                             const float epsilon, const int batch_size,
                             const int seq_len, const int vocab_size,
                             cudaStream_t stream);

template <typename T>
void launch_cross_entropy_bw(const float *grad_outputs_ptr, const T *inputs_ptr,
                             const int *targets_ptr, T *grad_inputs_ptr,
                             const int padding_idx, const float epsilon,
                             const int batch_size, const int seq_len,
                             const int vocab_size, cudaStream_t stream);

template <typename T>
void launch_lookup_scale_pos_dropout(
    T *output, const int *input, const T *embeddings, const T *pos_embeddings,
    uint8_t *dropout_mask, int batch_size, int seq_len, int embedding_dim,
    int padding_idx, float dropout_ratio, int step, cudaStream_t &stream);

template <typename T>
void launch_d_lookup_scale_pos_dropout(
    T *grad_embeddings, const T *grad_output, const int *input,
    const uint8_t *dropout_mask, int batch_size, int seq_len, int embedding_dim,
    int vocab_size, int padding_idx, float dropout_ratio, cudaStream_t &stream);

/* Convert 2-dim tensor index into vector index */
__forceinline__ __host__ __device__ int flat_2dim(int id1, int id2, int dim2) {
  return id1 * dim2 + id2;
}

/* Convert 3-dim tensor index into vector index */
__forceinline__ __host__ __device__ int flat_3dim(int id1, int id2, int id3,
                                                  int dim2, int dim3) {
  return id1 * dim2 * dim3 + id2 * dim3 + id3;
}

/* Convert 4-dim tensor index into vector index */
__forceinline__ __host__ __device__ int flat_4dim(int id1, int id2, int id3,
                                                  int id4, int dim2, int dim3,
                                                  int dim4) {
  // return id1*(dim2*dim3*dim4) + id2*(dim3*dim4) + id3*dim4 + id4;
  int res = id4;

  int ld = dim4;
  res += id3 * ld;

  ld *= dim3;
  res += id2 * ld;

  ld *= dim2;
  res += id1 * ld;

  return res;
}

/* Convert 5-dim tensor index into vector index */
__forceinline__ __host__ __device__ int flat_5dim(int id1, int id2, int id3,
                                                  int id4, int id5, int dim2,
                                                  int dim3, int dim4,
                                                  int dim5) {
  // return id1*(dim2*dim3*dim4*dim5) + id2*(dim3*dim4*dim5) + id3*(dim4*dim5) +
  // id4*dim5 + dim5;
  int res = id5;

  int ld = dim5;
  res += id4 * ld;

  ld *= dim4;
  res += id3 * ld;

  ld *= dim3;
  res += id2 * ld;

  ld *= dim2;
  res += id1 * ld;

  return res;
}

/* Convert 6-dim tensor index into vector index */
__forceinline__ __host__ __device__ int flat_6dim(int id1, int id2, int id3,
                                                  int id4, int id5, int id6,
                                                  int dim2, int dim3, int dim4,
                                                  int dim5, int dim6) {
  // return id1*(dim2*dim3*dim4*dim5*dim6) + id2*(dim3*dim4*dim5*dim6) +
  // id3*(dim4*dim5*dim6) + id4*(dim5*dim6) + id5*dim6 + id6;
  int res = id6;

  int ld = dim6;
  res += id5 * ld;

  ld *= dim5;
  res += id4 * ld;

  ld *= dim4;
  res += id3 * ld;

  ld *= dim3;
  res += id2 * ld;

  ld *= dim2;
  res += id1 * ld;

  return res;
}

/* Convert vector index to 6-dim tensor index */
__forceinline__ __host__ __device__ void decompose_6dim(
    int src, int dim1, int dim2, int dim3, int dim4, int dim5, int *id0,
    int *id1, int *id2, int *id3, int *id4, int *id5) {
  *id5 = src % dim5;
  src /= dim5;

  *id4 = src % dim4;
  src /= dim4;

  *id3 = src % dim3;
  src /= dim3;

  *id2 = src % dim2;
  src /= dim2;

  *id1 = src % dim1;
  *id0 = src / dim1;
}

/* Convert vector index to 5-dim tensor index */
__forceinline__ __host__ __device__ void decompose_5dim(int src, int dim1,
                                                        int dim2, int dim3,
                                                        int dim4, int *id0,
                                                        int *id1, int *id2,
                                                        int *id3, int *id4) {
  *id4 = src % dim4;
  src /= dim4;

  *id3 = src % dim3;
  src /= dim3;

  *id2 = src % dim2;
  src /= dim2;

  *id1 = src % dim1;
  *id0 = src / dim1;
}

/* Convert vector index to 4-dim tensor index */
__forceinline__ __host__ __device__ void decompose_4dim(int src, int dim1,
                                                        int dim2, int dim3,
                                                        int *id0, int *id1,
                                                        int *id2, int *id3) {
  *id3 = src % dim3;
  src /= dim3;

  *id2 = src % dim2;
  src /= dim2;

  *id1 = src % dim1;
  *id0 = src / dim1;
}

/* Convert vector index to 3-dim tensor index */
__forceinline__ __host__ __device__ void decompose_3dim(int src, int dim1,
                                                        int dim2, int *id0,
                                                        int *id1, int *id2) {
  *id2 = src % dim2;
  src /= dim2;

  *id1 = src % dim1;
  *id0 = src / dim1;
}

/* Convert vector index to 2-dim tensor index */
__forceinline__ __host__ __device__ void decompose_2dim(int src, int dim1,
                                                        int *id0, int *id1) {
  *id1 = src % dim1;
  *id0 = src / dim1;
}
