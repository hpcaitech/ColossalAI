#pragma once

#include <ATen/ATen.h>

#include "micros.h"

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

template <bool high_precision, typename scalar_t>
struct ScalarTypeTrait;

template <typename T>
struct ScalarTypeTrait<true, T> {
  using Type = typename MPTypeTrait<T>::Type;
};

template <typename T>
struct ScalarTypeTrait<false, T> {
  using Type = T;
};

}  // namespace common
}  // namespace colossalAI
