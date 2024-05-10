#include <torch/extension.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/CUDAContext.h>

#include <cmath>

#include "common/micros.h"
#include "utils/vec_copy.h"
#include "funcs/cast_functor.h"


using colossalAI::cuda::utils::copy;
using colossalAI::cuda::utils::get_vec_size;
using colossalAI::funcs::CastFunctor;

template <typename InT, typename OutT, int VecSize>
__global__ void convert_fp8_kernel(const InT* ins_data, OutT* outs_data, int numel, int tail)
{
  int64_t idx = static_cast<int64_t>(threadIdx.x) + static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x);
  const int64_t grid_size = blockDim.x * gridDim.x;
  if(idx > numel + tail) {
    return;
  }

  for(int64_t i = idx; i < numel; i += grid_size) {
    copy<InT, OutT, VecSize>(ins_data + i * VecSize, outs_data + i * VecSize);
  }
  // Tail process
  if(threadIdx.x == 0)
  {
    for(int i = 0; i < tail; ++i)
    {
      outs_data[i + numel * VecSize] = CastFunctor<InT, OutT>()(ins_data[i + numel * VecSize]);
    }
  }
}

template <typename InT, typename OutT>
void apply_convert_fp8(torch::Tensor& input, torch::Tensor& output)
{
  const int kVecSize = get_vec_size<InT>(input);
  const int kNumel = torch::numel(input);

  const int kVecNumel = (kNumel >> static_cast<int>(std::log2(kVecSize)));
  const int kTail = kNumel & (kVecSize - 1);
  int grid_size = kVecNumel ? (kVecNumel + 255) / 256 : 1;

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(grid_size);
  dim3 block(256);

#define _(VEC_SIZE)                                                   \
    convert_fp8_kernel<InT, OutT, VEC_SIZE>                           \
                    <<<grid, block, 0, stream>>>                      \
                    (reinterpret_cast<const InT*>(input.data_ptr()),  \
                    reinterpret_cast<OutT*>(output.data_ptr()),       \
                    kVecNumel,                                        \
                    kTail)

  switch (kVecSize)
  {
  case 1:
    _(1);
    break;
  case 2:
    _(2);
    break;
  case 4:
    _(4);
    break;
  }
#undef _
  AT_CUDA_CHECK(cudaGetLastError());
}

void convert_fp8(torch::Tensor& input, torch::Tensor& output)
{
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Byte || output.scalar_type() == at::ScalarType::Byte, "Data type of Input or Output should be torch.uint8 for convert_fp8!");
  TORCH_CHECK(input.scalar_type() != output.scalar_type(), "Data type of input and output are the same!");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Byte ||
              input.scalar_type() == at::ScalarType::Float ||
              input.scalar_type() == at::ScalarType::Half ||
              input.scalar_type() == at::ScalarType::BFloat16, "Unsupported dtype of input!");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::Byte ||
              output.scalar_type() == at::ScalarType::Float ||
              output.scalar_type() == at::ScalarType::Half ||
              output.scalar_type() == at::ScalarType::BFloat16, "Unsupported dtype of output!");
  TORCH_CHECK(input.sizes() == output.sizes(), "Shape of input and output should be the same!");

#define _(InT, OutT)                                         \
    apply_convert_fp8<InT, OutT>(input, output)


  if(input.scalar_type() == at::ScalarType::Byte)
  {
    if(output.scalar_type() == at::ScalarType::Float)
    {
      _(uint8_t, float);
    }
    else if(output.scalar_type() == at::ScalarType::Half)
    {
      _(uint8_t, half);
    }
    else if(output.scalar_type() == at::ScalarType::BFloat16)
    {
      _(uint8_t, __nv_bfloat16);
    }
  }
  else
  {
    if(input.scalar_type() == at::ScalarType::Float)
    {
      _(float, uint8_t);
    }
    else if(input.scalar_type() == at::ScalarType::Half)
    {
      _(half, uint8_t);
    }
    else if(input.scalar_type() == at::ScalarType::BFloat16)
    {
      _(__nv_bfloat16, uint8_t);
    }
  }

#undef _
}
