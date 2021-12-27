#include <ATen/ATen.h>
#include "compat.h"


#define DISPATCH_HALF_AND_BFLOAT(TYPE, NAME, ...)			\
  switch(TYPE)								\
    {									\
    case at::ScalarType::Half:						\
      {									\
	using scalar_t = at::Half;					\
	__VA_ARGS__;							\
	break;								\
      }									\
    case at::ScalarType::BFloat16:					\
      {									\
	using scalar_t = at::BFloat16;					\
	__VA_ARGS__;							\
	break;								\
      }									\
    default:								\
      AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");	\
      }



#define DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(TYPEIN, TYPEOUT, NAME, ...) \
  switch(TYPEIN)							\
    {									\
    case at::ScalarType::Float:						\
      {									\
	using scalar_t_in = float;					\
	switch(TYPEOUT)							\
	  {								\
	  case at::ScalarType::Float:					\
	    {								\
	      using scalar_t_out = float;				\
	      __VA_ARGS__;						\
	      break;							\
	    }								\
	  case at::ScalarType::Half:					\
	    {								\
	      using scalar_t_out = at::Half;				\
	      __VA_ARGS__;						\
	      break;							\
	    }								\
	  case at::ScalarType::BFloat16:				\
	    {								\
	      using scalar_t_out = at::BFloat16;			\
	      __VA_ARGS__;						\
	      break;							\
	    }								\
	  default:							\
	    AT_ERROR(#NAME, " not implemented for '", toString(TYPEOUT), "'"); \
	  }								\
	break;								\
      }									\
    case at::ScalarType::Half:						\
      {									\
	using scalar_t_in = at::Half;					\
	using scalar_t_out = at::Half;					\
	__VA_ARGS__;							\
	break;								\
      }									\
    case at::ScalarType::BFloat16:					\
      {									\
	using scalar_t_in = at::BFloat16;				\
	using scalar_t_out = at::BFloat16;				\
	__VA_ARGS__;							\
	break;								\
      }									\
    default:								\
      AT_ERROR(#NAME, " not implemented for '", toString(TYPEIN), "'");	\
    }
