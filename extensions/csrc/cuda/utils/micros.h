#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <exception>

#define CUDA_CHECK(func)                                    \
  {                                                         \
    auto status = func;                                     \
    if (status != cudaSuccess) {                            \
      throw std::runtime_error(cudaGetErrorString(status)); \
    }                                                       \
  }
