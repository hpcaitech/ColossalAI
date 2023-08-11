// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <cuda.h>
#ifdef BF16_AVAILABLE
#include <cuda_bf16.h>
#endif
#include <cassert>
#include <cuda_fp16.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

template <typename T, typename TW>
at::Tensor gptq_act_linear_layer(at::Tensor& input,
                                 at::Tensor& weight,
                                 at::Tensor& weight_scales,
                                 at::Tensor& weight_zeros,
                                 at::Tensor& bias,
                                 at::Tensor& residual,
                                 int64_t group_size,
                                 int32_t act_type,
                                 int32_t add_bias,
                                 int32_t add_residual,
                                 int32_t qkv_fused,
                                 uint64_t block_size_x,
                                 uint64_t block_size_y);