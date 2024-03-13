#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(func)                                           \
  {                                                                \
    auto status = func;                                            \
    if (status != cudaSuccess) {                                   \
      LOG(FATAL) << "CUDA Error : " << cudaGetErrorString(status); \
    }                                                              \
  }
