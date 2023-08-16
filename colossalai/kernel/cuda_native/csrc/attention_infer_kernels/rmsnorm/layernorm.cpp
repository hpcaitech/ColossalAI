/* Copyright 2021 The Colossal-AI Team
   Copyright (c) 2023, The vLLM team.
   This file is adapted from vllm TEAM: https://github.com/vllm-project/vllm/blob/main/csrc/layernorm.cpp
*/
#include <torch/extension.h>

void rms_norm(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& weight,
  float epsilon);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "rms_norm",
    &rms_norm,
    "Apply Root Mean Square (RMS) Normalization to the input tensor.");
}
