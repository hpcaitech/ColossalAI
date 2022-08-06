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
#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <torch/extension.h>
#if (__x86_64__ || __i386__)
#include <cpuid.h>
#include <x86intrin.h>
#endif

#define ROUND_DOWN(size, step) ((size) & ~((step)-1))
#define TILE (128 * 1024 * 1024)

#if defined(__AVX512__) or defined(__AVX256__) or defined(__AVX2__)

#if defined(__AVX512__)
#define SIMD_WIDTH 16
#define INTV __m256i
#define SIMD_STORE(a, d) _mm512_storeu_ps(a, d)
#define SIMD_LOAD(x) _mm512_loadu_ps(x)
#define SIMD_SET(x) _mm512_set1_ps(x)
#define SIMD_ADD(x, y) _mm512_add_ps(x, y)
#define SIMD_MUL(x, y) _mm512_mul_ps(x, y)
#define SIMD_FMA(x, y, c) _mm512_fmadd_ps(x, y, c)
#define SIMD_SQRT(x) _mm512_sqrt_ps(x)
#define SIMD_DIV(x, y) _mm512_div_ps(x, y)
#define SIMD_LOAD_HALF(x) \
  _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(x)))
#define SIMD_STORE_HALF(x, d) \
  _mm256_store_ps(            \
      x, _mm256_castsi256_ps(_mm512_cvtps_ph(d, _MM_FROUND_TO_NEAREST_INT)))

#elif defined(__AVX256__) or defined(__AVX2__)
#define SIMD_WIDTH 8
#define INTV __m128i
#define SIMD_STORE(a, d) _mm256_storeu_ps(a, d)
#define SIMD_LOAD(x) _mm256_loadu_ps(x)
#define SIMD_SET(x) _mm256_set1_ps(x)
#define SIMD_ADD(x, y) _mm256_add_ps(x, y)
#define SIMD_MUL(x, y) _mm256_mul_ps(x, y)
#define SIMD_FMA(x, y, c) _mm256_fmadd_ps(x, y, c)
#define SIMD_SQRT(x) _mm256_sqrt_ps(x)
#define SIMD_DIV(x, y) _mm256_div_ps(x, y)
#define SIMD_LOAD_HALF(x) _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x)))
#define SIMD_STORE_HALF(x, d) \
  _mm_store_ps(               \
      x, _mm_castsi128_ps(_mm256_cvtps_ph(d, _MM_FROUND_TO_NEAREST_INT)))

#endif

union AVX_Data {
#if defined(__AVX512__)
  __m512 data;
#elif defined(__AVX256__) or defined(__AVX2__)
  __m256 data;
#endif
  // float data_f[16];
};

#endif

#define STEP(SPAN)                                                \
  void Step_##SPAN(float *_params, float *grads, float *_exp_avg, \
                   float *_exp_avg_sq, size_t _param_size,        \
                   bool param_half_precision = false,             \
                   bool grad_half_precision = false, float loss_scale = -1);

class Adam_Optimizer {
 public:
  Adam_Optimizer(float alpha = 1e-3, float betta1 = 0.9, float betta2 = 0.999,
                 float eps = 1e-8, float weight_decay = 0,
                 bool adamw_mode = true)
      : _alpha(alpha),
        _betta1(betta1),
        _betta2(betta2),
        _eps(eps),
        _weight_decay(weight_decay),
        _betta1_t(1.0),
        _betta2_t(1.0),
        _step(0),
        _adamw_mode(adamw_mode) {}
  ~Adam_Optimizer() {}

  STEP(1)
  STEP(4)
  STEP(8)
  inline void IncrementStep(size_t step, float beta1, float beta2) {
    if (beta1 != _betta1 || beta2 != _betta2) {
      _step = step;
      _betta1 = beta1;
      _betta2 = beta2;
      _betta1_t = std::pow(_betta1, step);
      _betta2_t = std::pow(_betta2, step);
    } else {
      _step++;
      if (_step != step) {
        _betta1_t = std::pow(_betta1, step);
        _betta2_t = std::pow(_betta2, step);
        _step = step;
      } else {
        _betta1_t *= _betta1;
        _betta2_t *= _betta2;
      }
    }
  }
  inline void update_state(float lr, float epsilon, float weight_decay,
                           bool bias_correction) {
    _alpha = lr;
    _eps = epsilon;
    _weight_decay = weight_decay;

    _bias_correction1 = 1.0f;
    _bias_correction2 = 1.0f;
    if (bias_correction == 1) {
      _bias_correction1 = 1 - _betta1_t;
      _bias_correction2 = 1 / sqrt(1 - _betta2_t);
    }
  }

  void step(size_t step, float lr, float beta1, float beta2, float epsilon,
            float weight_decay, bool bias_correction, torch::Tensor &params,
            torch::Tensor &grads, torch::Tensor &exp_avg,
            torch::Tensor &exp_avg_sq, float loss_scale);

 private:
  float _alpha;
  float _betta1;
  float _betta2;
  float _eps;
  float _weight_decay;

  float _betta1_t;
  float _betta2_t;
  size_t _step;

  float _bias_correction1;
  float _bias_correction2;

  bool _adamw_mode;
};
