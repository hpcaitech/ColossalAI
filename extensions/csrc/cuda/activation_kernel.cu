#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <stdio.h>

#include "type_shim.h"
#include "include/mp_type_traits.h"

template<typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // x * sigmoid(x)
  using MT = typename infer::dtype::MPTypeTrait<T>::Type;
  return static_cast<T>((static_cast<MT>(x)) / (static_cast<MT>(1.0f) + expf(static_cast<MT>(-x))));
}

template<typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
__global__ void act_and_mul_kernel(
  const scalar_t* __restrict__ ins_data,
  scalar_t* __restrict__ outs_data,
  const int64_t numel) {
  using MT = typename infer::dtype::MPTypeTrait<scalar_t>::Type;

  int64_t idx = static_cast<int64_t>(threadIdx.x) + static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x);
  const int64_t grid_size = blockDim.x * gridDim.x;
  if(idx > numel) {
    return;
  }

  for(int64_t i = idx; i < numel; i += grid_size) {
    scalar_t x = ins_data[i];
    scalar_t y = ins_data[i+numel];
    outs_data[i] = static_cast<scalar_t>(static_cast<MT>(ACT_FN(x)) * static_cast<MT>(y));
  }
}

// Note(LiuYang):This func is designed for calculation mode like
// silu(x[:half_1stdim]) * (x[half_1stdim:])
torch::Tensor silu_and_mul(const torch::Tensor& ins)
{
    auto ins_shape = ins.sizes().vec();

    ins_shape[0] = ins_shape[0]/2;
    auto outs = torch::zeros(ins_shape,ins.options());
    auto outs_shape = ins.sizes().vec();

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Note(Liuyang): numel of ins must be divisible by 2
    int64_t numel = ((torch::numel(ins)) >> 1);

    // TODO(LiuYang): Maybe we need to implement a function to get launch config
    dim3 grid((numel+255)/256);
    dim3 block(256);

    DISPATCH_FLOAT_HALF_AND_BFLOAT(
        ins.scalar_type(),
        "silu_and_mul",
        act_and_mul_kernel<scalar_t,silu_kernel<scalar_t>><<<grid, block, 0, stream>>>(
            ins.data_ptr<scalar_t>(),
            outs.data_ptr<scalar_t>(),
            numel
        );)

    AT_CUDA_CHECK(cudaGetLastError());
    return outs;
}
