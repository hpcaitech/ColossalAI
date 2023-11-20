#include "cpu_adam_arm.h"

void AdamOptimizer::Step_1(void *_params, void *grads, void *_exp_avg,
                           void *_exp_avg_sq, size_t _param_size,
                           at::ScalarType param_dtype,
                           at::ScalarType grad_dtype,
                           at::ScalarType exp_avg_dtype,
                           at::ScalarType exp_avg_sq_dtype, float loss_scale) {
  size_t rounded_size = 0;
#if defined(__aarch64__)
  rounded_size = ROUND_DOWN(_param_size, SIMD_WIDTH);
#endif

  float betta1_minus1 = 1 - _betta1;
  float betta2_minus1 = 1 - _betta2;
  float step_size = -1 * _alpha / _bias_correction1;
  float w_decay = -1 * _alpha * _weight_decay;

#if defined(__aarch64__)
  float32x4_t betta1_4 = simd_set(_betta1);
  float32x4_t betta2_4 = simd_set(_betta2);
  float32x4_t betta1_minus1_4 = simd_set(betta1_minus1);
  float32x4_t betta2_minus1_4 = simd_set(betta2_minus1);
  float32x4_t bias2_sqrt = simd_set(_bias_correction2);
  float32x4_t eps_4 = simd_set(_eps);
  float32x4_t step_size_4 = simd_set(step_size);
  float32x4_t weight_decay_4;
  if (_weight_decay > 0) {
    weight_decay_4 = _adamw_mode ? simd_set(w_decay) : simd_set(_weight_decay);
  }
  for (size_t t = 0; t < rounded_size; t += TILE) {
    size_t copy_size = TILE;
    if ((t + TILE) > rounded_size) copy_size = rounded_size - t;
    size_t offset = copy_size + t;

#pragma omp parallel for
    for (size_t i = t; i < offset; i += SIMD_WIDTH) {
      float32x4_t grad_4 = simd_load_offset(grads, grad_dtype, i);
      if (loss_scale > 0) {
        float32x4_t loss_scale_vec = simd_set(loss_scale);
        grad_4 = vdivq_f32(grad_4, loss_scale_vec);
      }
      float32x4_t momentum_4 = simd_load_offset(_exp_avg, exp_avg_dtype, i);
      float32x4_t variance_4 =
          simd_load_offset(_exp_avg_sq, exp_avg_sq_dtype, i);
      float32x4_t param_4 = simd_load_offset(_params, param_dtype, i);
      if (_weight_decay > 0 && !_adamw_mode) {
        grad_4 = vfmaq_f32(grad_4, param_4, weight_decay_4);
      }
      momentum_4 = vmulq_f32(momentum_4, betta1_4);
      momentum_4 = vfmaq_f32(momentum_4, grad_4, betta1_minus1_4);
      variance_4 = vmulq_f32(variance_4, betta2_4);
      grad_4 = vmulq_f32(grad_4, grad_4);
      variance_4 = vfmaq_f32(variance_4, grad_4, betta2_minus1_4);
      grad_4 = vsqrtq_f32(variance_4);
      grad_4 = vfmaq_f32(eps_4, grad_4, bias2_sqrt);
      grad_4 = vdivq_f32(momentum_4, grad_4);
      if (_weight_decay > 0 && _adamw_mode) {
        param_4 = vfmaq_f32(param_4, param_4, weight_decay_4);
      }
      param_4 = vfmaq_f32(param_4, grad_4, step_size_4);
      simd_store_offset(_params, param_dtype, param_4, i);
      simd_store_offset(_exp_avg, exp_avg_dtype, momentum_4, i);
      simd_store_offset(_exp_avg_sq, exp_avg_sq_dtype, variance_4, i);
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
        float grad = scalar_load_offset(grads, grad_dtype, k);
        if (loss_scale > 0) {
          grad /= loss_scale;
        }
        float param = scalar_load_offset(_params, param_dtype, k);
        float momentum = scalar_load_offset(_exp_avg, exp_avg_dtype, k);
        float variance = scalar_load_offset(_exp_avg_sq, exp_avg_sq_dtype, k);
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

        scalar_store_offset(_params, param_dtype, param, k);
        scalar_store_offset(_exp_avg, exp_avg_dtype, momentum, k);
        scalar_store_offset(_exp_avg_sq, exp_avg_sq_dtype, variance, k);
      }
    }
  }
}

void AdamOptimizer::Step_4(void *_params, void *grads, void *_exp_avg,
                           void *_exp_avg_sq, size_t _param_size,
                           at::ScalarType param_dtype,
                           at::ScalarType grad_dtype,
                           at::ScalarType exp_avg_dtype,
                           at::ScalarType exp_avg_sq_dtype, float loss_scale) {
  size_t rounded_size = 0;
#if defined(__aarch64__)
  rounded_size = ROUND_DOWN(_param_size, SIMD_WIDTH * 4);
#endif

  float betta1_minus1 = 1 - _betta1;
  float betta2_minus1 = 1 - _betta2;
  float step_size = -1 * _alpha / _bias_correction1;
  float w_decay = -1 * _alpha * _weight_decay;

#if defined(__aarch64__)
  float32x4_t betta1_4 = simd_set(_betta1);
  float32x4_t betta2_4 = simd_set(_betta2);
  float32x4_t betta1_minus1_4 = simd_set(betta1_minus1);
  float32x4_t betta2_minus1_4 = simd_set(betta2_minus1);
  float32x4_t bias2_sqrt = simd_set(_bias_correction2);
  float32x4_t eps_4 = simd_set(_eps);
  float32x4_t step_size_4 = simd_set(step_size);
  float32x4_t weight_decay_4;
  if (_weight_decay > 0) {
    weight_decay_4 = _adamw_mode ? simd_set(w_decay) : simd_set(_weight_decay);
  }

  for (size_t t = 0; t < rounded_size; t += TILE) {
    size_t copy_size = TILE;
    if ((t + TILE) > rounded_size) copy_size = rounded_size - t;
    size_t offset = copy_size + t;

#pragma omp parallel for
    for (size_t i = t; i < offset; i += SIMD_WIDTH * 4) {
      float32x4_t grad_4[4];
      float32x4_t momentum_4[4];
      float32x4_t variance_4[4];
      float32x4_t param_4[4];
#pragma unroll 4
      for (int j = 0; j < 4; j++) {
        grad_4[j] = simd_load_offset(grads, grad_dtype, i + SIMD_WIDTH * j);
        if (loss_scale > 0) {
          float32x4_t loss_scale_vec = simd_set(loss_scale);
          grad_4[j] = vdivq_f32(grad_4[j], loss_scale_vec);
        }
        momentum_4[j] =
            simd_load_offset(_exp_avg, exp_avg_dtype, i + SIMD_WIDTH * j);
        variance_4[j] =
            simd_load_offset(_exp_avg_sq, exp_avg_sq_dtype, i + SIMD_WIDTH * j);
        param_4[j] = simd_load_offset(_params, param_dtype, i + SIMD_WIDTH * j);
        if (_weight_decay > 0 && !_adamw_mode) {
          grad_4[j] = vfmaq_f32(grad_4[j], param_4[j], weight_decay_4);
        }
        momentum_4[j] = vmulq_f32(momentum_4[j], betta1_4);
        momentum_4[j] = vfmaq_f32(momentum_4[j], grad_4[j], betta1_minus1_4);
        variance_4[j] = vmulq_f32(variance_4[j], betta2_4);
        grad_4[j] = vmulq_f32(grad_4[j], grad_4[j]);
        variance_4[j] = vfmaq_f32(variance_4[j], grad_4[j], betta2_minus1_4);
        grad_4[j] = vsqrtq_f32(variance_4[j]);
        grad_4[j] = vfmaq_f32(eps_4, grad_4[j], bias2_sqrt);
        grad_4[j] = vdivq_f32(momentum_4[j], grad_4[j]);
        if (_weight_decay > 0 && _adamw_mode) {
          param_4[j] = vfmaq_f32(param_4[j], param_4[j], weight_decay_4);
        }
        param_4[j] = vfmaq_f32(param_4[j], grad_4[j], step_size_4);
        simd_store_offset(_params, param_dtype, param_4[j], i + SIMD_WIDTH * j);
        simd_store_offset(_exp_avg, exp_avg_dtype, momentum_4[j],
                          i + SIMD_WIDTH * j);
        simd_store_offset(_exp_avg_sq, exp_avg_sq_dtype, variance_4[j],
                          i + SIMD_WIDTH * j);
      }
    }
  }
#endif
  if (_param_size > rounded_size) {
    Step_1(scalar_seek_offset(_params, param_dtype, rounded_size),
           scalar_seek_offset(grads, grad_dtype, rounded_size),
           scalar_seek_offset(_exp_avg, exp_avg_dtype, rounded_size),
           scalar_seek_offset(_exp_avg_sq, exp_avg_sq_dtype, rounded_size),
           (_param_size - rounded_size), param_dtype, grad_dtype, exp_avg_dtype,
           exp_avg_sq_dtype, loss_scale);
  }
}

void AdamOptimizer::Step_8(void *_params, void *grads, void *_exp_avg,
                           void *_exp_avg_sq, size_t _param_size,
                           at::ScalarType param_dtype,
                           at::ScalarType grad_dtype,
                           at::ScalarType exp_avg_dtype,
                           at::ScalarType exp_avg_sq_dtype, float loss_scale) {
  size_t rounded_size = 0;
#if defined(__aarch64__)
  rounded_size = ROUND_DOWN(_param_size, SIMD_WIDTH * 8);
#endif

  float betta1_minus1 = 1 - _betta1;
  float betta2_minus1 = 1 - _betta2;
  float step_size = -1 * _alpha / _bias_correction1;
  float w_decay = -1 * _alpha * _weight_decay;
#if defined(__aarch64__)
  float32x4_t betta1_4 = simd_set(_betta1);
  float32x4_t betta2_4 = simd_set(_betta2);
  float32x4_t betta1_minus1_4 = simd_set(betta1_minus1);
  float32x4_t betta2_minus1_4 = simd_set(betta2_minus1);
  float32x4_t bias2_sqrt = simd_set(_bias_correction2);
  float32x4_t eps_4 = simd_set(_eps);
  float32x4_t step_size_4 = simd_set(step_size);
  float32x4_t weight_decay_4;
  if (_weight_decay > 0) {
    weight_decay_4 = _adamw_mode ? simd_set(w_decay) : simd_set(_weight_decay);
  }

  for (size_t t = 0; t < rounded_size; t += TILE) {
    size_t copy_size = TILE;
    if ((t + TILE) > rounded_size) copy_size = rounded_size - t;
    size_t offset = copy_size + t;

#pragma omp parallel for
    for (size_t i = t; i < offset; i += SIMD_WIDTH * 8) {
      float32x4_t grad_4[8];
      float32x4_t momentum_4[8];
      float32x4_t variance_4[8];
      float32x4_t param_4[8];
#pragma unroll 4
      for (int j = 0; j < 8; j++) {
        grad_4[j] = simd_load_offset(grads, grad_dtype, i + SIMD_WIDTH * j);
        if (loss_scale > 0) {
          float32x4_t loss_scale_vec = simd_set(loss_scale);
          grad_4[j] = vdivq_f32(grad_4[j], loss_scale_vec);
        }
        momentum_4[j] =
            simd_load_offset(_exp_avg, exp_avg_dtype, i + SIMD_WIDTH * j);
        variance_4[j] =
            simd_load_offset(_exp_avg_sq, exp_avg_sq_dtype, i + SIMD_WIDTH * j);
        param_4[j] = simd_load_offset(_params, param_dtype, i + SIMD_WIDTH * j);
        if (_weight_decay > 0 && !_adamw_mode) {
          grad_4[j] = vfmaq_f32(grad_4[j], param_4[j], weight_decay_4);
        }
        momentum_4[j] = vmulq_f32(momentum_4[j], betta1_4);
        momentum_4[j] = vfmaq_f32(momentum_4[j], grad_4[j], betta1_minus1_4);
        variance_4[j] = vmulq_f32(variance_4[j], betta2_4);
        grad_4[j] = vmulq_f32(grad_4[j], grad_4[j]);
        variance_4[j] = vfmaq_f32(variance_4[j], grad_4[j], betta2_minus1_4);
        grad_4[j] = vsqrtq_f32(variance_4[j]);
        grad_4[j] = vfmaq_f32(eps_4, grad_4[j], bias2_sqrt);
        grad_4[j] = vdivq_f32(momentum_4[j], grad_4[j]);
        if (_weight_decay > 0 && _adamw_mode) {
          param_4[j] = vfmaq_f32(param_4[j], param_4[j], weight_decay_4);
        }
        param_4[j] = vfmaq_f32(param_4[j], grad_4[j], step_size_4);
        simd_store_offset(_params, param_dtype, param_4[j], i + SIMD_WIDTH * j);
        simd_store_offset(_exp_avg, exp_avg_dtype, momentum_4[j],
                          i + SIMD_WIDTH * j);
        simd_store_offset(_exp_avg_sq, exp_avg_sq_dtype, variance_4[j],
                          i + SIMD_WIDTH * j);
      }
    }
  }
#endif
  if (_param_size > rounded_size) {
    Step_4(scalar_seek_offset(_params, param_dtype, rounded_size),
           scalar_seek_offset(grads, grad_dtype, rounded_size),
           scalar_seek_offset(_exp_avg, exp_avg_dtype, rounded_size),
           scalar_seek_offset(_exp_avg_sq, exp_avg_sq_dtype, rounded_size),
           (_param_size - rounded_size), param_dtype, grad_dtype, exp_avg_dtype,
           exp_avg_sq_dtype, loss_scale);
  }
}

void AdamOptimizer::step(size_t step, float lr, float beta1, float beta2,
                         float epsilon, float weight_decay,
                         bool bias_correction, torch::Tensor &params,
                         torch::Tensor &grads, torch::Tensor &exp_avg,
                         torch::Tensor &exp_avg_sq, float loss_scale) {
  auto params_c = params.contiguous();
  auto grads_c = grads.contiguous();
  auto exp_avg_c = exp_avg.contiguous();
  auto exp_avg_sq_c = exp_avg_sq.contiguous();

  this->IncrementStep(step, beta1, beta2);
  this->update_state(lr, epsilon, weight_decay, bias_correction);
  this->Step_8(params_c.data_ptr(), grads_c.data_ptr(), exp_avg_c.data_ptr(),
               exp_avg_sq_c.data_ptr(), params_c.numel(),
               params_c.scalar_type(), grads_c.scalar_type(),
               exp_avg_c.scalar_type(), exp_avg_sq_c.scalar_type(), loss_scale);
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<AdamOptimizer>(m, "CPUAdamOptimizer")
      .def(py::init<float, float, float, float, float, bool>())
      .def("step", &AdamOptimizer::step);
}
