#pragma once

#include <ATen/ATen.h>

#include "../type_shim.h"

namespace infer {
namespace dtype {

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

}  // namespace dtype
}  // namespace infer
