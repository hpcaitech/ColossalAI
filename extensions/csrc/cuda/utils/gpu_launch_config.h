#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "nvgpu_dev_info.h"

namespace colossalAI {
namespace cuda {
namespace utils {

struct GPULaunchConfig {
  dim3 block{1, 1, 1};
  dim3 grid{1, 1, 1};
};

static GPULaunchConfig GetGPULaunchConfig1D(const NVGPUDevInfo& dev_info,
                                            int64_t numel, int64_t vec_size) {
  const int64_t max_threads_per_block = dev_info.GetMaxThreadsPerBlock();
  const int64_t max_blocks_per_grid = dev_info.GetMaxGridDims()[0];
  const int64_t kMinimumSize = 64;
  const int64_t kMaximumSize = 512;
  int64_t active_threads = (numel + vec_size - 1) / vec_size;
  int64_t sm_num = dev_info.GetMultiProcessorCount();

  // Note(LiuYang): expected threads should be in [64, 128, 256, 512] generally
  int64_t expected_threads_per_block = kMaximumSize;

  auto RoundUpToPowerOfTwo = [](int64_t x) {
    bool is_power_of_two = false;
    int64_t ret = 1;
    int64_t y = x;
    while (y > 0) {
      is_power_of_two = ((ret ^ x) == 0);
      y = (x >> 1);
      ret = (ret << 1);
      if (y > 0) is_power_of_two = false;
    }
    if (is_power_of_two) return x;
    return ret;
  };

  if ((active_threads / (sm_num << 1)) < max_threads_per_block) {
    expected_threads_per_block =
        RoundUpToPowerOfTwo(active_threads / (sm_num << 1));
  } else if ((active_threads / (sm_num << 2)) < max_threads_per_block) {
    expected_threads_per_block =
        RoundUpToPowerOfTwo(active_threads / (sm_num << 2));
  }

  expected_threads_per_block =
      std::max(expected_threads_per_block, kMinimumSize);
  int64_t expect_block_per_grid =
      ((active_threads + expected_threads_per_block - 1) /
       expected_threads_per_block);

  if (expect_block_per_grid > max_blocks_per_grid) {
    expect_block_per_grid = max_blocks_per_grid;
    expected_threads_per_block =
        (active_threads + expect_block_per_grid - 1) / expect_block_per_grid;
    if (expected_threads_per_block > max_threads_per_block)
      throw std::invalid_argument(
          "Threads required for current input exceed for current GPU!");
    expected_threads_per_block =
        RoundUpToPowerOfTwo(expected_threads_per_block);
    expect_block_per_grid = ((active_threads + expected_threads_per_block - 1) /
                             expected_threads_per_block);
  }

  GPULaunchConfig config;
  config.block.x = expected_threads_per_block;
  config.grid.x = expect_block_per_grid;
  return config;
}

}  // namespace utils
}  // namespace cuda
}  // namespace colossalAI
