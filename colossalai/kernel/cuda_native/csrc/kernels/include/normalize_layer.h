#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>

#include <fstream>

#include "kernels.h"

using namespace std;

template <typename T> class Normalize_Layer {
public:
  struct Config {
    uint32_t hidden_dim;
    bool use_mean;
    Config(uint32_t hidden_dim, bool use_mean = false)
        : hidden_dim(hidden_dim), use_mean(use_mean) {}
  };

  Normalize_Layer(Config config, size_t max_rows)
      : config_(config), vars_(nullptr), means_(nullptr) {
    vars_ = cuda_malloc<T>(max_rows);
    if (config_.use_mean) {
      means_ = cuda_malloc<T>(max_rows);
    }
  }

  ~Normalize_Layer() {
    cuda_free(vars_);
    cuda_free(means_);
  }

  void Forward(T *ln_res, const T *inp, const T *gamma, const T *betta,
               int batch_size, cudaStream_t stream) {
    launch_layer_norm(ln_res, vars_, means_, inp, gamma, betta, batch_size,
                      config_.hidden_dim, stream);
  }

  /*
  residual_grad, inp_or_out, betta should be treated carefully.
  inp_or_out = input if use_mean else output
  residual_grad, betta can be nullptr.
  residual_grad will be added to dinp if it is not nullptr
    which is useful in transformer layer when pre-ln
  betta are only used to compute xhat,
    (use_mean == false) ^ (betta == nullptr) should be true
  */
  void Backward(T *gamma_grad, T *betta_grad, T *inp_grad, const T *out_grad,
                const T *residual_grad, const T *inp_or_out, const T *gamma,
                const T *betta, int batch_size, cudaStream_t stream[2]) {
    launch_ln_bw(gamma_grad, betta_grad, inp_grad, out_grad, residual_grad,
                 inp_or_out, gamma, betta, vars_, means_, batch_size,
                 config_.hidden_dim, stream);
  }

  inline bool use_mean() const { return config_.use_mean; }

private:
  Config config_;
  T *vars_;
  T *means_;
};
