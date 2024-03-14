#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <ostream>
#include <string>
#include <vector>

#include "micros.h"
#include "target.h"

namespace colossalAI {
namespace cuda {
namespace utils {

class NVGPUDevInfo {
 public:
  explicit NVGPUDevInfo(int device_num) : device_num_(device_num) {
    CUDA_CALL(cudaGetDeviceProperties(prop_, device));
  }

  std::array<int, 3> GetMaxGridDims() const;
  std::array<int, 3> GetMaxBlockDims() const;
  std::array<int, 2> GetCapability() const;
  int GetMultiProcessorCount() const;
  int GetMaxThreadsPerMultiProcessor() const;
  int GetMaxThreadsPerBlock() const;

 private:
  int device_num_;
  cudaDeviceProp* prop_;
};

}  // namespace utils
}  // namespace cuda
}  // namespace colossalAI
