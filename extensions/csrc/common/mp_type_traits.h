#pragma once

#include <ATen/ATen.h>

#include "micros.h"

namespace colossalAI {
namespace common {

template <typename T>
class MPTypeTrait {
 public:
  using Type = float;
};

template <>
class MPTypeTrait<float> {
 public:
  using Type = float;
};

template <>
class MPTypeTrait<at::Half> {
 public:
  using Type = float;
};

template <>
class MPTypeTrait<at::BFloat16> {
 public:
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
