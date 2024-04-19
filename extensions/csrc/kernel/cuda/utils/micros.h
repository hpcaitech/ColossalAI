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

#define HOST __host__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
