#include "nvgpu_dev_info.h"

#include <array>

namespace colossalAI {
namespace cuda {
namespace utils {

std::array<int, 3> NVGPUDevInfo::GetMaxGridDims() const {
  std::array<int, 3> ret;
  ret[0] = prop_->maxGridSize[0];
  ret[1] = prop_->maxGridSize[1];
  ret[2] = prop_->maxGridSize[2];
  return ret;
}

std::array<int, 3> NVGPUDevInfo::GetMaxBlockDims() const {
  std::array<int, 3> ret;
  ret[0] = prop_->maxThreadsDim[0];
  ret[1] = prop_->maxThreadsDim[1];
  ret[2] = prop_->maxThreadsDim[2];
  return ret;
}

std::array<int, 2> NVGPUDevInfo::GetCapability() const {
  std::array<int, 2> ret;
  ret[0] = prop_.major;
  ret[1] = prop_.minor;
}

int NVGPUDevInfo::GetMultiProcessorCount() const {
  return prop_->multiProcessorCount;
}

int NVGPUDevInfo::GetMaxThreadsPerMultiProcessor() const {
  return prop_->maxThreadsPerMultiProcessor;
}

int NVGPUDevInfo::GetMaxThreadsPerBlock() const {
  return prop_->maxThreadsPerBlock;
}

}  // namespace utils
}  // namespace cuda
}  // namespace colossalAI
