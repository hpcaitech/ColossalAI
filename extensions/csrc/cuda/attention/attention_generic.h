/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h
 * Copyright (c) 2024, The Colossal-AI team.
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <stdint.h>

// A vector type to store Q, K, V elements.
template <typename T, int VEC_SIZE>
struct VecType {};

// A vector type to store FP32 accumulators.
template <typename T>
struct FloatVec {};

// Template vector operations.
template <typename Acc, typename A, typename B>
inline __device__ Acc mul(A a, B b);

template <typename T>
inline __device__ float sum(T v);

template <typename T, typename TFLOAT>
inline __device__ void from_float(T& dst, TFLOAT src);

template <typename A, typename B, typename C>
inline __device__ C fma(A a, B b, C c);
