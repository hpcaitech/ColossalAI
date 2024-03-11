#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace colossalAI {
namespace cuda {
namespace utils {

GPULaunchConfig GPUGetGPULaunchConfig1D(int64_t numel, int vec_size);

// TODO(LiuYang): to be implemented
GPULaunchConfig GPUGetGPULaunchConfig2D(int64_t numel, int vec_size);

// TODO(LiuYang): to be implemented
GPULaunchConfig GPUGetGPULaunchConfig3D(int64_t numel, int vec_size);

class GPULaunchConfig {
 public:
  GPULaunchConfig(){};
  GPULaunchConfig(const dim3& block, const dim3& grid)
      : block_(block), grid_(grid) {}
  friend GPULaunchConfig GPUGetGPULaunchConfig1D(int64_t numel, int vec_size);

 protected:
  void set_block(const dim3& dim) { block_ = dim; }
  void set_grid(const dim3& dim) { grid_ = dim; }

 private:
  dim3 block_(1, 1, 1);
  dim3 grid_(1, 1, 1);
}

}  // namespace utils
}  // namespace cuda
}  // namespace colossalAI
