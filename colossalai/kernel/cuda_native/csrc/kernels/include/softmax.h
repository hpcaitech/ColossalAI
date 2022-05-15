#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>

#include <fstream>

#include "kernels.h"

using namespace std;

template <typename T>
class Softmax {
 public:
  struct Config {
    size_t nhead;
    Config(size_t nhead) : nhead(nhead) {}
  };

  Softmax(Config config) : config_(config) {}

  ~Softmax() {}

  void Forward(T *vals, const T *attn_mask, int batch_size, int from_len,
               int to_len, cudaStream_t &stream, bool mask_future = true) {
    launch_attn_softmax<T>(vals, attn_mask, batch_size, config_.nhead, from_len,
                           to_len, mask_future, stream);
  }

  void Backward(T *out_grad, const T *soft_out, int batch_size, int from_len,
                int to_len, cudaStream_t stream) {
    launch_attn_softmax_bw<T>(out_grad, soft_out,
                              batch_size * config_.nhead * from_len, to_len,
                              stream);
  }

  void reset_size(size_t nhead) { config_.nhead = nhead; }

 private:
  Config config_;
};
