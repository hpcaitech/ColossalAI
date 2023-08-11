#include "inference_cuda_layers.h"
#include <c10/cuda/CUDAStream.h>
#include <stdexcept>
#include <torch/extension.h>
#include <vector>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{

    m.def("gptq_act_linear_fp16",
          &gptq_act_linear_layer<__half, uint64_t>,
          "gptq linear kernel (CUDA)");


    m.def("gptq_act_linear_fp16_w32",
          &gptq_act_linear_layer<__half, uint32_t>,
          "gptq linear kernel (CUDA)");

    m.def("gptq_act_linear_fp16_w8",
          &gptq_act_linear_layer<__half, uint8_t>,
          "gptq linear kernel (CUDA)");

}
