#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <type_traits>

#include "cuda_util.h"

template <typename T>
class CrossEntropyLayer {
 public:
  CrossEntropyLayer(float epsilon, int padding_idx, int max_batch_tokens);

  virtual ~CrossEntropyLayer();

  void Forward(const T *inputs_ptr, const int *targets_ptr, float *outputs_ptr,
               float *nll_loss_ptr);

  void Backward(const float *grad_outputs_ptr, const T *inputs_ptr,
                const int *targets_ptr, T *grad_inputs_ptr);

  void set_cur_batch_shape(int batch_size, int seq_len, int vocab_size);

 private:
  void allocate_mem_buffer() {
    // allocate local gpu memory
    _loss_buffer = cuda_malloc<float>(_max_batch_tokens * 2);
  }

  void free_mem_buffer() {
    // free local gpu memory
    cuda_free(_loss_buffer);
  }

  const int _padding_idx;
  const float _epsilon;
  const int _max_batch_tokens;

  size_t _batch_size;
  size_t _seq_len;
  size_t _vocab_size;

  float *_loss_buffer;
};
