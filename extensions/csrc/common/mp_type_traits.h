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

}  // namespace common
}  // namespace colossalAI
