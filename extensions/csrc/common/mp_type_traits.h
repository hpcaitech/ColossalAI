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

}  // namespace common
}  // namespace colossalAI
