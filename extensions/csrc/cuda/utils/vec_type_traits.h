#pragma once

#include <c10/macros/Macros.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include <cfloat>

namespace colossalAI {
namespace cuda {
namespace utils {

template <typename T, int VecSize>
struct VecTypeTrait {};

template <typename T>
struct VecTypeTrait<T, 1> {
  using Type = T;
};

template <>
struct VecTypeTrait<c10::BFloat16, 2> {
  using Type = float;
};

template <>
struct VecTypeTrait<c10::BFloat16, 4> {
  using Type = float2;
};

template <>
struct VecTypeTrait<c10::BFloat16, 8> {
  using Type = float4;
};

template <>
struct VecTypeTrait<c10::Half, 2> {
  using Type = float;
};

template <>
struct VecTypeTrait<c10::Half, 4> {
  using Type = float2;
};

template <>
struct VecTypeTrait<c10::Half, 8> {
  using Type = float4;
};

template <>
struct VecTypeTrait<float, 2> {
  using Type = float2;
};

template <>
struct VecTypeTrait<float, 4> {
  using Type = float4;
};

template <>
struct VecTypeTrait<float, 8> {
  using Type = float4;
};

template <>
struct VecTypeTrait<uint8_t, 2> {
  using Type = half;
};

template <>
struct VecTypeTrait<uint8_t, 4> {
  using Type = half2;
};

template <>
struct VecTypeTrait<uint8_t, 8> {
  using Type = float2;
};

}  // namespace utils
}  // namespace cuda
}  // namespace colossalAI
