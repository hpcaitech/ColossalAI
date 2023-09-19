#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>

#include <string>

#include "kernels.h"

template <typename T>
class Dropout {
 public:
  struct Config {
    float ratio;
    bool training;

    Config(float r) : ratio(r), training(true) {}
    float RATIO() const { return training ? ratio : 0.0; }
  };

  Dropout(const Config &config, size_t max_ele_num)
      : _config(config), _mask(nullptr) {
    _mask = cuda_malloc<uint8_t>(max_ele_num);
  }

  virtual ~Dropout() { cuda_free(_mask); }

  // after attention softmax
  void dropout(T *output, const T *input, int count, cudaStream_t stream,
               bool bwd = false) {
    launch_ls_dropout<T>(output, input, _mask, count, _config.RATIO(), stream,
                         bwd);
  }

  void d_dropout(T *d_inp_out, int count, cudaStream_t stream) {
    launch_ls_dropout<T>(d_inp_out, d_inp_out, _mask, count, _config.RATIO(),
                         stream, true);
  }

  // transformer layer's postprocessing dropout, after attn or ffn module,
  // before residual add.
  void bias_dropout_residual(T *output, const T *input, const T *residual,
                             const T *bias, int rows, int cols,
                             cudaStream_t stream) {
    launch_ls_dropout_res_bias<T>(output, input, _mask, bias, residual,
                                  rows * cols, cols, _config.RATIO(), stream);
  }

  void d_bias_dropout_residual(T *d_input, T *d_bias, const T *d_output,
                               int rows, int cols, cudaStream_t stream) {
    launch_ls_dropout_bias_bwd<T>(d_input, d_bias, d_output, _mask, rows, cols,
                                  _config.RATIO(), stream);
  }

  // dropout inside ffn.
  void bias_act_dropout(T *output, const T *input, const T *bias, int rows,
                        int cols, std::string activation_fn,
                        cudaStream_t stream) {
    if (activation_fn == "relu") {
      launch_ls_dropout_act_bias<ActivationType::kRelu, T>(
          output, input, _mask, bias, rows * cols, cols, _config.RATIO(),
          stream);
    } else if (activation_fn == "gelu") {
      launch_ls_dropout_act_bias<ActivationType::kGelu, T>(
          output, input, _mask, bias, rows * cols, cols, _config.RATIO(),
          stream);
    } else {
      throw std::runtime_error("not supported activation: " + activation_fn);
    }
  }

  void d_bias_act_dropout(T *d_inp_out, T *d_bias_out, const T *input,
                          const T *bias, int rows, int cols,
                          std::string activation_fn, cudaStream_t stream) {
    if (activation_fn == "relu") {
      launch_ls_dropout_act_bias_bwd<ActivationType::kRelu, T>(
          d_inp_out, d_bias_out, input, bias, d_inp_out, _mask, rows, cols,
          _config.RATIO(), stream);
    } else if (activation_fn == "gelu") {
      launch_ls_dropout_act_bias_bwd<ActivationType::kGelu, T>(
          d_inp_out, d_bias_out, input, bias, d_inp_out, _mask, rows, cols,
          _config.RATIO(), stream);
    } else {
      throw std::runtime_error("not supported activation: " + activation_fn);
    }
  }

  bool HasDropout() const { return _config.RATIO() > 0.0; }

  void SetTrainingMode(bool training) { _config.training = training; }

 private:
  uint8_t *_mask;
  Config _config;
};
