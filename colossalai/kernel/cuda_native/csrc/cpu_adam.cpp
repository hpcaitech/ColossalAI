/*
Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE
*/
#include "cpu_adam.h"

#include <math.h>
#include <omp.h>
#include <string.h>

#include <iostream>
#include <memory>
#include <type_traits>
#include <unordered_map>

// C++ interface

void Adam_Optimizer::Step_1(float *_params, float *grads, float *_exp_avg,
                            float *_exp_avg_sq, size_t _param_size,
                            bool param_half_precision, bool grad_half_precision,
                            float loss_scale) {
  size_t rounded_size = 0;

  float betta1_minus1 = 1 - _betta1;
  float betta2_minus1 = 1 - _betta2;
  float step_size = -1 * _alpha / _bias_correction1;
  float w_decay = -1 * _alpha * _weight_decay;

  __half *params_cast_h = NULL;
  __half *grads_cast_h = NULL;

  if (param_half_precision) {
    params_cast_h = reinterpret_cast<__half *>(_params);
  }
  if (grad_half_precision) {
    grads_cast_h = reinterpret_cast<__half *>(grads);
  }

#if defined(__AVX512__) or defined(__AVX256__) or defined(__AVX2__)
  AVX_Data betta1_4;
  betta1_4.data = SIMD_SET(_betta1);
  AVX_Data betta2_4;
  betta2_4.data = SIMD_SET(_betta2);

  AVX_Data betta1_minus1_4;
  betta1_minus1_4.data = SIMD_SET(betta1_minus1);
  AVX_Data betta2_minus1_4;
  betta2_minus1_4.data = SIMD_SET(betta2_minus1);

  AVX_Data bias2_sqrt;
  bias2_sqrt.data = SIMD_SET(_bias_correction2);

  AVX_Data eps_4;
  eps_4.data = SIMD_SET(_eps);

  AVX_Data step_size_4;
  step_size_4.data = SIMD_SET(step_size);

  AVX_Data weight_decay_4;
  if (_weight_decay > 0)
    weight_decay_4.data =
        (_adamw_mode ? SIMD_SET(w_decay) : SIMD_SET(_weight_decay));
  rounded_size = ROUND_DOWN(_param_size, SIMD_WIDTH);

  for (size_t t = 0; t < rounded_size; t += TILE) {
    size_t copy_size = TILE;
    if ((t + TILE) > rounded_size) copy_size = rounded_size - t;
    size_t offset = copy_size + t;

#pragma omp parallel for
    for (size_t i = t; i < offset; i += SIMD_WIDTH) {
      AVX_Data grad_4;
      if (grad_half_precision) {
        grad_4.data = SIMD_LOAD_HALF(grads_cast_h + i);
      } else {
        grad_4.data = SIMD_LOAD(grads + i);
      }
      if (loss_scale > 0) {
        AVX_Data loss_scale_vec;
        loss_scale_vec.data = SIMD_SET(loss_scale);
        grad_4.data = SIMD_DIV(grad_4.data, loss_scale_vec.data);
      }
      AVX_Data momentum_4;
      momentum_4.data = SIMD_LOAD(_exp_avg + i);

      AVX_Data variance_4;
      variance_4.data = SIMD_LOAD(_exp_avg_sq + i);

      AVX_Data param_4;
      if (param_half_precision) {
        param_4.data = SIMD_LOAD_HALF(params_cast_h + i);
      } else {
        param_4.data = SIMD_LOAD(_params + i);
      }

      if (_weight_decay > 0 && !_adamw_mode) {
        grad_4.data = SIMD_FMA(param_4.data, weight_decay_4.data, grad_4.data);
      }
      momentum_4.data = SIMD_MUL(momentum_4.data, betta1_4.data);
      momentum_4.data =
          SIMD_FMA(grad_4.data, betta1_minus1_4.data, momentum_4.data);
      variance_4.data = SIMD_MUL(variance_4.data, betta2_4.data);
      grad_4.data = SIMD_MUL(grad_4.data, grad_4.data);
      variance_4.data =
          SIMD_FMA(grad_4.data, betta2_minus1_4.data, variance_4.data);
      grad_4.data = SIMD_SQRT(variance_4.data);
      grad_4.data = SIMD_FMA(grad_4.data, bias2_sqrt.data, eps_4.data);
      grad_4.data = SIMD_DIV(momentum_4.data, grad_4.data);

      if (_weight_decay > 0 && _adamw_mode) {
        param_4.data =
            SIMD_FMA(param_4.data, weight_decay_4.data, param_4.data);
      }
      param_4.data = SIMD_FMA(grad_4.data, step_size_4.data, param_4.data);

      if (param_half_precision) {
        SIMD_STORE_HALF((float *)(params_cast_h + i), param_4.data);
      } else {
        SIMD_STORE(_params + i, param_4.data);
      }
      SIMD_STORE(_exp_avg + i, momentum_4.data);
      SIMD_STORE(_exp_avg_sq + i, variance_4.data);
    }
  }
#endif
  if (_param_size > rounded_size) {
    for (size_t t = rounded_size; t < _param_size; t += TILE) {
      size_t copy_size = TILE;
      if ((t + TILE) > _param_size) copy_size = _param_size - t;
      size_t offset = copy_size + t;

#pragma omp parallel for
      for (size_t k = t; k < offset; k++) {
        float grad = grad_half_precision ? (float)grads_cast_h[k] : grads[k];
        if (loss_scale > 0) {
          grad /= loss_scale;
        }
        float param =
            param_half_precision ? (float)params_cast_h[k] : _params[k];
        float momentum = _exp_avg[k];
        float variance = _exp_avg_sq[k];
        if (_weight_decay > 0 && !_adamw_mode) {
          grad = param * _weight_decay + grad;
        }
        momentum = momentum * _betta1;
        momentum = grad * betta1_minus1 + momentum;

        variance = variance * _betta2;
        grad = grad * grad;
        variance = grad * betta2_minus1 + variance;

        grad = sqrt(variance);
        grad = grad * _bias_correction2 + _eps;
        grad = momentum / grad;
        if (_weight_decay > 0 && _adamw_mode) {
          param += w_decay * param;
        }
        param = grad * step_size + param;

        if (param_half_precision)
          params_cast_h[k] = (__half)param;
        else
          _params[k] = param;
        _exp_avg[k] = momentum;
        _exp_avg_sq[k] = variance;
      }
    }
  }
}

void Adam_Optimizer::Step_4(float *_params, float *grads, float *_exp_avg,
                            float *_exp_avg_sq, size_t _param_size,
                            bool param_half_precision, bool grad_half_precision,
                            float loss_scale) {
  size_t rounded_size = 0;

  __half *params_cast_h = NULL;
  __half *grads_cast_h = NULL;
  if (param_half_precision) {
    params_cast_h = reinterpret_cast<__half *>(_params);
  }
  if (grad_half_precision) {
    grads_cast_h = reinterpret_cast<__half *>(grads);
  }

#if defined(__AVX512__) or defined(__AVX256__) or defined(__AVX2__)
  AVX_Data betta1_4;
  betta1_4.data = SIMD_SET(_betta1);
  AVX_Data betta2_4;
  betta2_4.data = SIMD_SET(_betta2);

  float betta1_minus1 = 1 - _betta1;
  AVX_Data betta1_minus1_4;
  betta1_minus1_4.data = SIMD_SET(betta1_minus1);
  float betta2_minus1 = 1 - _betta2;
  AVX_Data betta2_minus1_4;
  betta2_minus1_4.data = SIMD_SET(betta2_minus1);

  AVX_Data bias2_sqrt;
  bias2_sqrt.data = SIMD_SET(_bias_correction2);

  AVX_Data eps_4;
  eps_4.data = SIMD_SET(_eps);

  float step_size = -1 * _alpha / _bias_correction1;
  AVX_Data step_size_4;
  step_size_4.data = SIMD_SET(step_size);

  float w_decay = -1 * _alpha * _weight_decay;
  AVX_Data weight_decay_4;
  if (_weight_decay > 0)
    weight_decay_4.data =
        (_adamw_mode ? SIMD_SET(w_decay) : SIMD_SET(_weight_decay));
  rounded_size = ROUND_DOWN(_param_size, SIMD_WIDTH * 4);

  for (size_t t = 0; t < rounded_size; t += TILE) {
    size_t copy_size = TILE;
    if ((t + TILE) > rounded_size) copy_size = rounded_size - t;
    size_t offset = copy_size + t;

#pragma omp parallel for
    for (size_t i = t; i < offset; i += SIMD_WIDTH * 4) {
      AVX_Data grad_4[4];
      AVX_Data momentum_4[4];
      AVX_Data variance_4[4];
      AVX_Data param_4[4];
#pragma unroll 4
      for (int j = 0; j < 4; j++) {
        if (grad_half_precision) {
          grad_4[j].data = SIMD_LOAD_HALF(grads_cast_h + i + SIMD_WIDTH * j);
        } else {
          grad_4[j].data = SIMD_LOAD(grads + i + SIMD_WIDTH * j);
        }

        if (loss_scale > 0) {
          AVX_Data loss_scale_vec;
          loss_scale_vec.data = SIMD_SET(loss_scale);
          grad_4[j].data = SIMD_DIV(grad_4[j].data, loss_scale_vec.data);
        }

        momentum_4[j].data = SIMD_LOAD(_exp_avg + i + SIMD_WIDTH * j);
        variance_4[j].data = SIMD_LOAD(_exp_avg_sq + i + SIMD_WIDTH * j);

        if (param_half_precision) {
          param_4[j].data = SIMD_LOAD_HALF(params_cast_h + i + SIMD_WIDTH * j);
        } else {
          param_4[j].data = SIMD_LOAD(_params + i + SIMD_WIDTH * j);
        }

        if (_weight_decay > 0 && !_adamw_mode) {
          grad_4[j].data =
              SIMD_FMA(param_4[j].data, weight_decay_4.data, grad_4[j].data);
        }
        momentum_4[j].data = SIMD_MUL(momentum_4[j].data, betta1_4.data);
        momentum_4[j].data =
            SIMD_FMA(grad_4[j].data, betta1_minus1_4.data, momentum_4[j].data);
        variance_4[j].data = SIMD_MUL(variance_4[j].data, betta2_4.data);
        grad_4[j].data = SIMD_MUL(grad_4[j].data, grad_4[j].data);
        variance_4[j].data =
            SIMD_FMA(grad_4[j].data, betta2_minus1_4.data, variance_4[j].data);
        grad_4[j].data = SIMD_SQRT(variance_4[j].data);
        grad_4[j].data = SIMD_FMA(grad_4[j].data, bias2_sqrt.data, eps_4.data);
        grad_4[j].data = SIMD_DIV(momentum_4[j].data, grad_4[j].data);

        if (_weight_decay > 0 && _adamw_mode) {
          param_4[j].data =
              SIMD_FMA(param_4[j].data, weight_decay_4.data, param_4[j].data);
        }
        param_4[j].data =
            SIMD_FMA(grad_4[j].data, step_size_4.data, param_4[j].data);
        if (param_half_precision) {
          SIMD_STORE_HALF((float *)(params_cast_h + i + SIMD_WIDTH * j),
                          param_4[j].data);
        } else {
          SIMD_STORE(_params + i + SIMD_WIDTH * j, param_4[j].data);
        }
        SIMD_STORE(_exp_avg + i + SIMD_WIDTH * j, momentum_4[j].data);
        SIMD_STORE(_exp_avg_sq + i + SIMD_WIDTH * j, variance_4[j].data);
      }
    }
  }
#endif
  if (_param_size > rounded_size)
    Step_1((param_half_precision ? (float *)(params_cast_h + rounded_size)
                                 : _params + rounded_size),
           (grad_half_precision ? (float *)(grads_cast_h + rounded_size)
                                : grads + rounded_size),
           (_exp_avg + rounded_size), (_exp_avg_sq + rounded_size),
           (_param_size - rounded_size), param_half_precision,
           grad_half_precision, loss_scale);
}

void Adam_Optimizer::Step_8(float *_params, float *grads, float *_exp_avg,
                            float *_exp_avg_sq, size_t _param_size,
                            bool param_half_precision, bool grad_half_precision,
                            float loss_scale) {
  size_t rounded_size = 0;
  __half *params_cast_h = NULL;
  __half *grads_cast_h = NULL;
  if (param_half_precision) {
    params_cast_h = reinterpret_cast<__half *>(_params);
  }
  if (grad_half_precision) {
    grads_cast_h = reinterpret_cast<__half *>(grads);
  }
#if defined(__AVX512__) or defined(__AVX256__) or defined(__AVX2__)
  AVX_Data betta1_4;
  betta1_4.data = SIMD_SET(_betta1);
  AVX_Data betta2_4;
  betta2_4.data = SIMD_SET(_betta2);

  float betta1_minus1 = 1 - _betta1;
  AVX_Data betta1_minus1_4;
  betta1_minus1_4.data = SIMD_SET(betta1_minus1);
  float betta2_minus1 = 1 - _betta2;
  AVX_Data betta2_minus1_4;
  betta2_minus1_4.data = SIMD_SET(betta2_minus1);

  AVX_Data bias2_sqrt;
  bias2_sqrt.data = SIMD_SET(_bias_correction2);

  AVX_Data eps_4;
  eps_4.data = SIMD_SET(_eps);

  float step_size = -1 * _alpha / _bias_correction1;
  AVX_Data step_size_4;
  step_size_4.data = SIMD_SET(step_size);

  float w_decay = -1 * _alpha * _weight_decay;
  AVX_Data weight_decay_4;
  if (_weight_decay > 0)
    weight_decay_4.data =
        (_adamw_mode ? SIMD_SET(w_decay) : SIMD_SET(_weight_decay));
  rounded_size = ROUND_DOWN(_param_size, SIMD_WIDTH * 8);

  for (size_t t = 0; t < rounded_size; t += TILE) {
    size_t copy_size = TILE;
    if ((t + TILE) > rounded_size) copy_size = rounded_size - t;
    size_t offset = copy_size + t;

#pragma omp parallel for
    for (size_t i = t; i < offset; i += SIMD_WIDTH * 8) {
      AVX_Data grad_4[8];
      AVX_Data momentum_4[8];
      AVX_Data variance_4[8];
      AVX_Data param_4[8];
#pragma unroll 8
      for (int j = 0; j < 8; j++) {
        if (grad_half_precision) {
          grad_4[j].data = SIMD_LOAD_HALF(grads_cast_h + i + SIMD_WIDTH * j);
        } else {
          grad_4[j].data = SIMD_LOAD(grads + i + SIMD_WIDTH * j);
        }

        if (loss_scale > 0) {
          AVX_Data loss_scale_vec;
          loss_scale_vec.data = SIMD_SET(loss_scale);
          grad_4[j].data = SIMD_DIV(grad_4[j].data, loss_scale_vec.data);
        }

        momentum_4[j].data = SIMD_LOAD(_exp_avg + i + SIMD_WIDTH * j);
        variance_4[j].data = SIMD_LOAD(_exp_avg_sq + i + SIMD_WIDTH * j);

        if (param_half_precision) {
          param_4[j].data = SIMD_LOAD_HALF(params_cast_h + i + SIMD_WIDTH * j);
        } else {
          param_4[j].data = SIMD_LOAD(_params + i + SIMD_WIDTH * j);
        }

        if (_weight_decay > 0 && !_adamw_mode) {
          grad_4[j].data =
              SIMD_FMA(param_4[j].data, weight_decay_4.data, grad_4[j].data);
        }
        momentum_4[j].data = SIMD_MUL(momentum_4[j].data, betta1_4.data);
        momentum_4[j].data =
            SIMD_FMA(grad_4[j].data, betta1_minus1_4.data, momentum_4[j].data);
        variance_4[j].data = SIMD_MUL(variance_4[j].data, betta2_4.data);
        grad_4[j].data = SIMD_MUL(grad_4[j].data, grad_4[j].data);
        variance_4[j].data =
            SIMD_FMA(grad_4[j].data, betta2_minus1_4.data, variance_4[j].data);
        grad_4[j].data = SIMD_SQRT(variance_4[j].data);
        grad_4[j].data = SIMD_FMA(grad_4[j].data, bias2_sqrt.data, eps_4.data);
        grad_4[j].data = SIMD_DIV(momentum_4[j].data, grad_4[j].data);
        if (_weight_decay > 0 && _adamw_mode) {
          param_4[j].data =
              SIMD_FMA(param_4[j].data, weight_decay_4.data, param_4[j].data);
        }
        param_4[j].data =
            SIMD_FMA(grad_4[j].data, step_size_4.data, param_4[j].data);

        if (param_half_precision) {
          SIMD_STORE_HALF((float *)(params_cast_h + i + SIMD_WIDTH * j),
                          param_4[j].data);
        } else {
          SIMD_STORE(_params + i + SIMD_WIDTH * j, param_4[j].data);
        }

        SIMD_STORE(_exp_avg + i + (SIMD_WIDTH * j), momentum_4[j].data);
        SIMD_STORE(_exp_avg_sq + i + (SIMD_WIDTH * j), variance_4[j].data);
      }
    }
  }
#endif
  if (_param_size > rounded_size)
    Step_4((param_half_precision ? (float *)(params_cast_h + rounded_size)
                                 : _params + rounded_size),
           (grad_half_precision ? (float *)(grads_cast_h + rounded_size)
                                : grads + rounded_size),
           (_exp_avg + rounded_size), (_exp_avg_sq + rounded_size),
           (_param_size - rounded_size), param_half_precision,
           grad_half_precision, loss_scale);
}

void Adam_Optimizer::step(size_t step, float lr, float beta1, float beta2,
                          float epsilon, float weight_decay,
                          bool bias_correction, torch::Tensor &params,
                          torch::Tensor &grads, torch::Tensor &exp_avg,
                          torch::Tensor &exp_avg_sq, float loss_scale) {
  auto params_c = params.contiguous();
  auto grads_c = grads.contiguous();
  auto exp_avg_c = exp_avg.contiguous();
  auto exp_avg_sq_c = exp_avg_sq.contiguous();

  float *params_ptr = (float *)params_c.data_ptr();
  float *grads_ptr = (float *)grads_c.data_ptr();
  float *exp_avg_ptr = (float *)exp_avg_c.data_ptr();
  float *exp_avg_sq_ptr = (float *)exp_avg_sq_c.data_ptr();

  this->IncrementStep(step, beta1, beta2);
  this->update_state(lr, epsilon, weight_decay, bias_correction);
  this->Step_8(params_ptr, grads_ptr, exp_avg_ptr, exp_avg_sq_ptr,
               params_c.numel(), (params.options().dtype() == at::kHalf),
               (grads.options().dtype() == at::kHalf), loss_scale);
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<Adam_Optimizer>(m, "CPUAdamOptimizer")
      .def(py::init<float, float, float, float, float, bool>())
      .def("step", &Adam_Optimizer::step);
}
