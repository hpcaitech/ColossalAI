#pragma once
#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cmath>

#define ROUND_DOWN(size, step) ((size) & ~((step)-1))
#define TILE (128 * 1024 * 1024)

#if defined(__aarch64__)
#include <arm_neon.h>
#define SIMD_WIDTH 4

inline float32x4_t simd_load_offset(const void *ptr, at::ScalarType dtype,
                                    size_t offset) {
  switch (dtype) {
    case at::ScalarType::Float: {
      auto ptr_f = reinterpret_cast<const float32_t *>(ptr);
      return vld1q_f32(ptr_f + offset);
    }
    case at::ScalarType::Half: {
      auto ptr_h = reinterpret_cast<const float16_t *>(ptr);
      return vcvt_f32_f16(vld1_f16(ptr_h + offset));
    }
    // case at::ScalarType::BFloat16: {
    //   auto ptr_b = reinterpret_cast<const bfloat16_t *>(ptr);
    //   return vcvt_f32_bf16(vld1_bf16(ptr_b + offset));
    // }
    default:
      AT_ERROR("Unsupported dtype");
      break;
  }
}
inline float32x4_t simd_load(void const *ptr, at::ScalarType dtype) {
  return simd_load_offset(ptr, dtype, 0);
}

inline void simd_store_offset(void *ptr, at::ScalarType dtype, float32x4_t data,
                              size_t offset) {
  switch (dtype) {
    case at::ScalarType::Float: {
      auto ptr_f = reinterpret_cast<float32_t *>(ptr);
      vst1q_f32(ptr_f + offset, data);
      break;
    }
    case at::ScalarType::Half: {
      auto ptr_h = reinterpret_cast<float16_t *>(ptr);
      vst1_f16(ptr_h + offset, vcvt_f16_f32(data));
      break;
    }
    // case at::ScalarType::BFloat16: {
    //   auto ptr_b = reinterpret_cast<bfloat16_t *>(ptr);
    //   vst1_bf16(ptr_b + offset, vcvt_bf16_f32(data));
    //   break;
    // }
    default:
      AT_ERROR("Unsupported dtype");
      break;
  }
}

inline void simd_store(void *ptr, at::ScalarType dtype, float32x4_t data) {
  return simd_store_offset(ptr, dtype, data, 0);
}

inline float32x4_t simd_set(float value) {
  auto val = static_cast<float32_t>(value);
  return vdupq_n_f32(val);
}

#endif

inline float scalar_load_offset(const void *ptr, at::ScalarType dtype,
                                size_t offset) {
  switch (dtype) {
    case at::ScalarType::Float:
      return *(reinterpret_cast<const float *>(ptr) + offset);
    case at::ScalarType::Half:
      return static_cast<float>(
          *(reinterpret_cast<const at::Half *>(ptr) + offset));
    // case at::ScalarType::BFloat16:
    //   return static_cast<float>(
    //       *(reinterpret_cast<const at::BFloat16 *>(ptr) + offset));
    default:
      AT_ERROR("Unsupported dtype");
      break;
  }
}

inline void scalar_store_offset(void *ptr, at::ScalarType dtype, float data,
                                size_t offset) {
  switch (dtype) {
    case at::ScalarType::Float:
      *(reinterpret_cast<float *>(ptr) + offset) = data;
      break;
    case at::ScalarType::Half:
      *(reinterpret_cast<at::Half *>(ptr) + offset) = data;
      break;
      // case at::ScalarType::BFloat16:
      //   *(reinterpret_cast<at::BFloat16 *>(ptr) + offset) = data;
      break;
    default:
      AT_ERROR("Unsupported dtype");
      break;
  }
}

inline void *scalar_seek_offset(void *ptr, at::ScalarType dtype,
                                size_t offset) {
  switch (dtype) {
    case at::ScalarType::Float:
      return reinterpret_cast<float *>(ptr) + offset;
    case at::ScalarType::Half:
      return reinterpret_cast<at::Half *>(ptr) + offset;
    // case at::ScalarType::BFloat16:
    //   return reinterpret_cast<at::BFloat16 *>(ptr) + offset;
    default:
      AT_ERROR("Unsupported dtype");
      break;
  }
}
#define STEP(SPAN)                                                        \
  void Step_##SPAN(void *_params, void *grads, void *_exp_avg,            \
                   void *_exp_avg_sq, size_t _param_size,                 \
                   at::ScalarType param_dtype, at::ScalarType grad_dtype, \
                   at::ScalarType exp_avg_dtype,                          \
                   at::ScalarType exp_avg_sq_dtype, float loss_scale = -1);

class AdamOptimizer {
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

 public:
  AdamOptimizer(float alpha = 1e-3, float betta1 = 0.9, float betta2 = 0.999,
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
  ~AdamOptimizer() {}

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
};
