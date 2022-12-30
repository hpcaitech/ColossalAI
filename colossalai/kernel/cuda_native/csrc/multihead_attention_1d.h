#pragma once

#include <c10/util/intrusive_ptr.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <torch/torch.h>

#if TORCH_VERSION_MAJOR > 1 || \
    (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 13)
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#else
#include <c10d/ProcessGroup.hpp>
#endif

#include <string>
#include <type_traits>

#include "cuda_util.h"
#include "dropout.h"
#include "feed_forward.h"
#include "normalize_layer.h"
#include "softmax.h"
#include "strided_batch_gemm.h"

template <typename T>
class MultiHeadAttention {
 public:
  MultiHeadAttention(int layer_id, int max_batch_tokens, int _max_seq_len,
                     int hidden_size, int num_heads, float attn_dropout_ratio,
                     float hidden_output_dropout_ratio,
                     bool pre_or_postLayerNorm);

  virtual ~MultiHeadAttention();

  void Forward(const T *input_ptr, const T *input_mask_ptr, T *out_ptr);

  void Backward(const T *grad_output_ptr, const T *input_ptr,
                const T *output_ptr, const T *input_mask_ptr,
                T *grad_input_ptr);

  void attn_layer_fw(const T *input_ptr, const T *input_mask_ptr, T *output_ptr,
                     T *buffer);

  void attn_layer_bw(const T *input_ptr, const T *input_mask_ptr,
                     const T *output_ptr, const T *grad_output_ptr,
                     T *grad_input_attn_layer_bwptr, T *buffer);

  void set_cur_batch_shape(int batch_size, int seq_len) {
    _batch_size = batch_size;
    _seq_len = seq_len;
    _batch_tokens = batch_size * seq_len;
    _batch_heads = batch_size * _heads / pg_size;
    _batch_dim = _batch_tokens * _hidden_size;
    _attn_scores.SetConfig(_seq_len, _seq_len, _hidden_size / _heads);
    _attn_context.SetConfig(_hidden_size / _heads, _seq_len, _seq_len);
  }

  void SetTrainingMode(bool training);
  inline bool IsTrainingMode() const { return _training; }

  void SetPG(c10::intrusive_ptr<c10d::ProcessGroup> pg_) {
    pg = pg_;
    pg_size = 1;
    if (pg != c10::detail::UniqueVoidPtr()) {
      pg_size = pg->getSize();
    }
    allocate_mem_buffer();
  }

  // weights ptr
  const T *_attn_qkvw_ptr;
  const T *_attn_qkvb_ptr;
  const T *_attn_ow_ptr;
  const T *_attn_ob_ptr;
  const T *_attn_nw_ptr;
  const T *_attn_nb_ptr;

  // grads ptr
  T *_grad_attn_qkvw_ptr;
  T *_grad_attn_qkvb_ptr;
  T *_grad_attn_ow_ptr;
  T *_grad_attn_ob_ptr;
  T *_grad_attn_nw_ptr;
  T *_grad_attn_nb_ptr;

 private:
  void allocate_mem_buffer() {
    // allocate local gpu memory
    if (_pre_or_postLayerNorm) {
      _gemmQKV_inp_ptr = cuda_malloc<T>(_max_batch_tokens * _hidden_size);
    } else {
      _gemmQKV_inp_ptr = nullptr;
    }

    _qkv_ptr = cuda_malloc<T>(_max_batch_tokens * _hidden_size * 3);
    _soft_out_ptr =
        cuda_malloc<T>(_max_batch_tokens * _heads / pg_size * _max_seq_len);
    _ctx_bufB_ptr =
        cuda_malloc<T>(_max_batch_tokens * _heads / pg_size * _max_seq_len);
    _attn_o_inp_ptr = cuda_malloc<T>(_max_batch_tokens * _hidden_size);

    // buffer size needed by attn bw
    size_t smem_size =
        4 * _max_batch_tokens * _hidden_size / pg_size +
        std::max(3 * _max_batch_tokens * _hidden_size / pg_size,
                 _max_batch_tokens * _heads / pg_size * _max_seq_len);

    if (!_shared_mem_ptr) {
      cuda_free(_shared_mem_ptr);
      _shared_mem_ptr = cuda_malloc<T>(smem_size);
    }
  }

  void free_mem_buffer() {
    // free local gpu memory
    cuda_free(_gemmQKV_inp_ptr);
    cuda_free(_qkv_ptr);
    cuda_free(_soft_out_ptr);
    cuda_free(_ctx_bufB_ptr);
    cuda_free(_attn_o_inp_ptr);

    // free shared gpu memory between layers
    cuda_free(_shared_mem_ptr);
    _shared_mem_ptr = nullptr;
  }

  // const parameter between batch
  const size_t _layer_id;
  const size_t _hidden_size;
  const size_t _heads;
  const size_t _max_batch_tokens;
  const size_t _max_seq_len;
  const bool _pre_or_postLayerNorm;
  // dynamic parameter between batch
  size_t _batch_size;
  size_t _seq_len;
  size_t _batch_tokens;
  size_t _batch_heads;
  size_t _batch_dim;
  bool _training;

  cublasHandle_t _cublasHandle;
  cudaStream_t _stream;

  // layers
  FeedForward<T> _qkv_linear;
  FeedForward<T> _attn_out_linear;
  Normalize_Layer<T> _attn_ln;
  Softmax<T> _softmax;
  Dropout<T> _attn_prob_dropout;
  Dropout<T> _attn_dropout;
  StridedBatchGemm<T> _attn_scores;
  StridedBatchGemm<T> _attn_context;

  // local GPU memory
  T *_gemmQKV_inp_ptr;
  T *_qkv_ptr;
  T *_soft_out_ptr;
  T *_ctx_bufB_ptr;
  T *_attn_o_inp_ptr;
  // shared GPU memory between layer
  static T *_shared_mem_ptr;

  c10::intrusive_ptr<c10d::ProcessGroup> pg;
  int pg_size;
};
