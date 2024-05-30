#pragma once

#include <ATen/ATen.h>

#include "micros.h"

#if defined(COLOSSAL_WITH_CUDA)
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif

namespace colossalAI {
namespace common {

template <typename T>
struct MPTypeTrait {
  using Type = float;
};

template <>
struct MPTypeTrait<float> {
  using Type = float;
};

template <>
struct MPTypeTrait<at::Half> {
  using Type = float;
};

template <>
struct MPTypeTrait<at::BFloat16> {
  using Type = float;
};

#if defined(COLOSSAL_WITH_CUDA)
template <>
struct MPTypeTrait<half> {
  using Type = float;
};

template <>
struct MPTypeTrait<__nv_bfloat16> {
  using Type = float;
};
#endif

template <bool high_precision, typename T>
struct ScalarTypeTrait {
  using Type =
      typename std::conditional<high_precision, typename MPTypeTrait<T>::Type,
                                T>::type;
};

}  // namespace common
}  // namespace colossalAI
