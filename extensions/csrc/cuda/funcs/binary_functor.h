#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <functional>

#include "../utils/micros.h"

namespace colossalAI {
namespace cuda {
namespace funcs {

enum class BinaryOpType { kAdd = 0, kMinus, kMul, kDiv, kMax, kMin };

// Note(LiuYang): This file provides base math operation for data type
// include POD and cuda built-in type such as half and __nv_bfloat16.
// Implementation of common and simple binary operators should be placed here,
// otherwise, they should be placed in a new file under functors dir.
template <typename LT, typename RT, typename RET, BinaryOpType op_type>
struct BinaryOpFunctor;

#define COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(T, BINARY_OP_TYPE, STMT,    \
                                               FUNCTION_MODIFIER, ARGS...) \
  template <ARGS>                                                          \
  struct BinaryOpFunctor<T, T, T, BINARY_OP_TYPE>                          \
      : public std::binary_function<T, T, T> {                             \
    FUNCTION_MODIFIER T operator()(T lhs, T rhs) { return STMT; }          \
  };

COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(T, BinaryOpType::kAdd, lhs + rhs,
                                       HOSTDEVICE, typename T)
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(T, BinaryOpType::kMinus, lhs - rhs,
                                       HOSTDEVICE, typename T)
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(T, BinaryOpType::kMul, lhs* rhs,
                                       HOSTDEVICE, typename T)
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(T, BinaryOpType::kDiv, lhs / rhs,
                                       HOSTDEVICE, typename T)
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(T, BinaryOpType::kMax, max(lhs, rhs),
                                       HOSTDEVICE, typename T)
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(T, BinaryOpType::kMin, min(lhs, rhs),
                                       HOSTDEVICE, typename T)

COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(half, BinaryOpType::kAdd,
                                       __hadd(lhs, rhs), DEVICE)
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(half2, BinaryOpType::kAdd,
                                       __hadd2(lhs, rhs), DEVICE)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(__nv_bfloat16, BinaryOpType::kAdd,
                                       __hadd(lhs, rhs), DEVICE)
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(__nv_bfloat162, BinaryOpType::kAdd,
                                       __hadd2(lhs, rhs), DEVICE)
#else
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(__nv_bfloat16, BinaryOpType::kAdd,
                                       __float2bfloat16(__bfloat162float(lhs) +
                                                        __bfloat162float(rhs)),
                                       DEVICE)
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(
    __nv_bfloat162, BinaryOpType::kAdd,
    __floats2bfloat162_rn(__low2float(lhs) + __low2float(rhs),
                          __high2float(lhs) + __high2float(rhs)),
    DEVICE)
#endif

COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(half, BinaryOpType::kMul,
                                       __hmul(lhs, rhs), DEVICE)
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(half2, BinaryOpType::kMul,
                                       __hmul2(lhs, rhs), DEVICE)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(__nv_bfloat16, BinaryOpType::kMul,
                                       __hmul(lhs, rhs), DEVICE)
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(__nv_bfloat162, BinaryOpType::kMul,
                                       __hmul2(lhs, rhs), DEVICE)
#else
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(__nv_bfloat16, BinaryOpType::kMul,
                                       __float2bfloat16(__bfloat162float(lhs) *
                                                        __bfloat162float(rhs)),
                                       DEVICE)
COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION(
    __nv_bfloat162, BinaryOpType::kMul,
    __floats2bfloat162_rn(__low2float(lhs) * __low2float(rhs),
                          __high2float(lhs) * __high2float(rhs)),
    DEVICE)
#endif

#undef COLOSSAL_BINARY_FUNCTOR_SPECIALIZATION

}  // namespace funcs
}  // namespace cuda
}  // namespace colossalAI
