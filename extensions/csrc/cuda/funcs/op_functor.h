#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <functional>

namespace colossalAI {
namespace cuda {
namespace funcs {

enum class BinaryOpType { kAdd = 0, kMinus, kMul, KDiv, kMax, KMin };

template <typename T, BinaryOpType Op>
struct BinaryOpFunctor;

template <typename T>
struct BinaryOpFunctor<T, BinaryOpType::kAdd>
    : public std::binary_function<T, T, T> {
  __host__ __device__ T operator()(T lhs, T rhs) { return lhs + rhs; }
};

template <typename T>
struct BinaryOpFunctor<T, BinaryOpType::kMax>
    : public std::binary_function<T, T, T> {
  __host__ __device__ T operator()(T lhs, T rhs) { return max(lhs, rhs); }
};

}  // namespace funcs
}  // namespace cuda
}  // namespace colossalAI
