#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <ostream>
#include <string>
#include <vector>

#include "micros.h"

namespace colossalAI {
namespace cuda {
namespace utils {

class NVGPUDevInfo {
 public:
  explicit NVGPUDevInfo(int device_num) : device_num_(device_num) {
    CUDA_CHECK(cudaGetDeviceProperties(&prop_, device_num));
  }

  std::array<int, 3> GetMaxGridDims() const {
    std::array<int, 3> ret;
    ret[0] = prop_.maxGridSize[0];
    ret[1] = prop_.maxGridSize[1];
    ret[2] = prop_.maxGridSize[2];
    return ret;
  }

  std::array<int, 3> GetMaxBlockDims() const {
    std::array<int, 3> ret;
    ret[0] = prop_.maxThreadsDim[0];
    ret[1] = prop_.maxThreadsDim[1];
    ret[2] = prop_.maxThreadsDim[2];
    return ret;
  }

  std::array<int, 2> GetCapability() const {
    std::array<int, 2> ret;
    ret[0] = prop_.major;
    ret[1] = prop_.minor;
    return ret;
  }

  int GetMultiProcessorCount() const { return prop_.multiProcessorCount; }

  int GetMaxThreadsPerMultiProcessor() const {
    return prop_.maxThreadsPerMultiProcessor;
  }

  int GetMaxThreadsPerBlock() const { return prop_.maxThreadsPerBlock; }

 private:
  int device_num_;
  cudaDeviceProp prop_;
};

}  // namespace utils
}  // namespace cuda
}  // namespace colossalAI
