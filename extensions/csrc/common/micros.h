/* Taken from NVIDIA/apex commit 855808f3fc268e9715d613f3c2e56469d8c986d8 */
/* Copyright 2020 The Microsoft DeepSpeed Team
   Copyright NVIDIA/apex
   This file is adapted from fused adam in NVIDIA/apex, commit a109f85
   Licensed under the MIT License.
*/

#pragma once

#include <ATen/ATen.h>

#define DISPATCH_HALF_AND_BFLOAT(TYPE, NAME, ...)                     \
  switch (TYPE) {                                                     \
    case at::ScalarType::Half: {                                      \
      using scalar_t = at::Half;                                      \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case at::ScalarType::BFloat16: {                                  \
      using scalar_t = at::BFloat16;                                  \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    default:                                                          \
      AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
  }

#define DISPATCH_FLOAT_HALF_AND_BFLOAT(TYPE, NAME, ...)               \
  switch (TYPE) {                                                     \
    case at::ScalarType::Float: {                                     \
      using scalar_t = float;                                         \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case at::ScalarType::Half: {                                      \
      using scalar_t = at::Half;                                      \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case at::ScalarType::BFloat16: {                                  \
      using scalar_t = at::BFloat16;                                  \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    default:                                                          \
      AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
  }

#define DISPATCH_FLOAT_HALF_AND_BFLOAT_WITH_HIGH_PRECISION(HIGH_PRECISION,  \
                                                           TYPE, NAME, ...) \
  if (HIGH_PRECISION) {                                                     \
    const bool high_precision = true;                                       \
    DISPATCH_FLOAT_HALF_AND_BFLOAT(TYPE, NAME, __VA_ARGS__);                \
  } else {                                                                  \
    const bool high_precision = false;                                      \
    DISPATCH_FLOAT_HALF_AND_BFLOAT(TYPE, NAME, __VA_ARGS__);                \
  }

#define DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(TYPEIN, TYPEOUT, NAME, ...) \
  switch (TYPEIN) {                                                            \
    case at::ScalarType::Float: {                                              \
      using scalar_t_in = float;                                               \
      switch (TYPEOUT) {                                                       \
        case at::ScalarType::Float: {                                          \
          using scalar_t_out = float;                                          \
          __VA_ARGS__;                                                         \
          break;                                                               \
        }                                                                      \
        case at::ScalarType::Half: {                                           \
          using scalar_t_out = at::Half;                                       \
          __VA_ARGS__;                                                         \
          break;                                                               \
        }                                                                      \
        case at::ScalarType::BFloat16: {                                       \
          using scalar_t_out = at::BFloat16;                                   \
          __VA_ARGS__;                                                         \
          break;                                                               \
        }                                                                      \
        default:                                                               \
          AT_ERROR(#NAME, " not implemented for '", toString(TYPEOUT), "'");   \
      }                                                                        \
      break;                                                                   \
    }                                                                          \
    case at::ScalarType::Half: {                                               \
      using scalar_t_in = at::Half;                                            \
      using scalar_t_out = at::Half;                                           \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case at::ScalarType::BFloat16: {                                           \
      using scalar_t_in = at::BFloat16;                                        \
      using scalar_t_out = at::BFloat16;                                       \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    default:                                                                   \
      AT_ERROR(#NAME, " not implemented for '", toString(TYPEIN), "'");        \
  }

// Forward/backward compatiblity hack around
// https://github.com/pytorch/pytorch/commit/3aeb78079bcd68282fe9117088e138b77318e288
// pending more future-proof guidance from upstream.
// struct TypeShim
// {
//   const at::Type& payload;
//   TypeShim(const at::Type& type) : payload(type) {}
//   // Enable trivial conversion to a const at::Type& for pre-3aeb78
//   operator const at::Type&(){ return payload; };
//   // Enable dispatch switch statements to take *this directly for post-3aeb78
//   //operator at::ScalarType(){ return payload.; };
// };

#define DISPATCH_FLOAT_AND_HALF(TYPE, LEVEL, NAME, ...)               \
  switch (TYPE) {                                                     \
    case at::ScalarType::Float: {                                     \
      using scalar_t_##LEVEL = float;                                 \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case at::ScalarType::Half: {                                      \
      using scalar_t_##LEVEL = at::Half;                              \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    default:                                                          \
      AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
  }

#define DISPATCH_FLOAT_HALF_AND_BYTE(TYPE, LEVEL, NAME, ...)          \
  switch (TYPE) {                                                     \
    case at::ScalarType::Float: {                                     \
      using scalar_t_##LEVEL = float;                                 \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case at::ScalarType::Half: {                                      \
      using scalar_t_##LEVEL = at::Half;                              \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case at::ScalarType::Byte: {                                      \
      using scalar_t_##LEVEL = uint8_t;                               \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    default:                                                          \
      AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
  }

#define DISPATCH_DOUBLE_FLOAT_AND_HALF(TYPE, LEVEL, NAME, ...)        \
  switch (TYPE) {                                                     \
    case at::ScalarType::Double: {                                    \
      using scalar_t_##LEVEL = double;                                \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case at::ScalarType::Float: {                                     \
      using scalar_t_##LEVEL = float;                                 \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case at::ScalarType::Half: {                                      \
      using scalar_t_##LEVEL = at::Half;                              \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    default:                                                          \
      AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
  }

#define DISPATCH_DOUBLE_AND_FLOAT(TYPE, LEVEL, NAME, ...)             \
  switch (TYPE) {                                                     \
    case at::ScalarType::Double: {                                    \
      using scalar_t_##LEVEL = double;                                \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case at::ScalarType::Float: {                                     \
      using scalar_t_##LEVEL = float;                                 \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    default:                                                          \
      AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
  }

#define DISPATCH_FLOAT_AND_HALF_FOR_G_P(GTYPE, PTYPE, LEVEL, NAME, ...)        \
  if (GTYPE == at::ScalarType::Float && PTYPE == at::ScalarType::Float) {      \
    using g_scalar_t_##LEVEL = float;                                          \
    using p_scalar_t_##LEVEL = float;                                          \
    __VA_ARGS__;                                                               \
  } else if (GTYPE == at::ScalarType::Float &&                                 \
             PTYPE == at::ScalarType::Half) {                                  \
    using g_scalar_t_##LEVEL = float;                                          \
    using p_scalar_t_##LEVEL = at::Half;                                       \
    __VA_ARGS__;                                                               \
  } else if (GTYPE == at::ScalarType::Half &&                                  \
             PTYPE == at::ScalarType::Float) {                                 \
    using g_scalar_t_##LEVEL = at::Half;                                       \
    using p_scalar_t_##LEVEL = float;                                          \
    __VA_ARGS__;                                                               \
  } else if (GTYPE == at::ScalarType::Half && PTYPE == at::ScalarType::Half) { \
    using g_scalar_t_##LEVEL = at::Half;                                       \
    using p_scalar_t_##LEVEL = at::Half;                                       \
    __VA_ARGS__;                                                               \
  } else if (GTYPE == at::ScalarType::Float &&                                 \
             PTYPE == at::ScalarType::BFloat16) {                              \
    using g_scalar_t_##LEVEL = float;                                          \
    using p_scalar_t_##LEVEL = at::BFloat16;                                   \
    __VA_ARGS__;                                                               \
  } else if (GTYPE == at::ScalarType::BFloat16 &&                              \
             PTYPE == at::ScalarType::Float) {                                 \
    using g_scalar_t_##LEVEL = at::BFloat16;                                   \
    using p_scalar_t_##LEVEL = float;                                          \
    __VA_ARGS__;                                                               \
  } else if (GTYPE == at::ScalarType::BFloat16 &&                              \
             PTYPE == at::ScalarType::BFloat16) {                              \
    using g_scalar_t_##LEVEL = at::BFloat16;                                   \
    using p_scalar_t_##LEVEL = at::BFloat16;                                   \
    __VA_ARGS__;                                                               \
  } else {                                                                     \
    AT_ERROR(#NAME, "not implemented for '", toString(GTYPE), toString(PTYPE), \
             "'");                                                             \
  }

#if defined(COLOSSAL_WITH_CUDA)
#define HOST __host__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define HOST
#define DEVICE
#define HOSTDEVICE
#endif
