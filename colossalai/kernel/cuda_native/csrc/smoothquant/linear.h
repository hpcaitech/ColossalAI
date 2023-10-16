#include <torch/torch.h>
#include <torch/types.h>

#include <cstdint>
#include <iostream>

torch::Tensor linear_silu_a8_w8_bfp32_ofp32(torch::Tensor input,   // INT8
                                            torch::Tensor weight,  // INT8
                                            torch::Tensor bias,    // FP32
                                            float alpha,           // FP32
                                            float beta             // FP32
);
