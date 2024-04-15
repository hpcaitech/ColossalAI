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

// Note(LiuYang): As a retrieved table to check which operation is supported
// already
enum class UnaryOpType { kLog2Ceil = 0, kAbs };

// Note(LiuYang): Implementation of common and simple unary operators should be
// placed here, otherwise, they should be placed in a new file under functors
// dir.
template <typename From, typename To, UnaryOpType op_type>
struct UnaryOpFunctor;

#define COLOSSAL_UNARY_FUNCTOR_SPECIALIZATION(                  \
    FROM, TO, UNARY_OP_TYPE, FUNCTION_MODIFIER, STMTS, ARGS...) \
  template <ARGS>                                               \
  struct UnaryOpFunctor<FROM, TO, UNARY_OP_TYPE>                \
      : public std::unary_function<FROM, TO> {                  \
    FUNCTION_MODIFIER TO operator()(FROM val) STMTS             \
  };

COLOSSAL_UNARY_FUNCTOR_SPECIALIZATION(
    T, T, UnaryOpType::kAbs, HOSTDEVICE, { return std::abs(val); }, typename T)

COLOSSAL_UNARY_FUNCTOR_SPECIALIZATION(int, int, UnaryOpType::kLog2Ceil,
                                      HOSTDEVICE, {
                                        int log2_value = 0;
                                        while ((1 << log2_value) < val)
                                          ++log2_value;
                                        return log2_value;
                                      })

#undef COLOSSAL_UARY_FUNCTOR_SPECIALIZATION

}  // namespace funcs
}  // namespace cuda
}  // namespace colossalAI
