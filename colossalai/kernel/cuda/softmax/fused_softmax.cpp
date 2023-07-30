/* coding=utf-8
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_fp16.h>
#include <torch/extension.h>
#include <vector>


torch::Tensor fwd_cuda(
    torch::Tensor const& input, 
    torch::Tensor const& mask,
    float scale_factor);

torch::Tensor fwd(
    torch::Tensor const& input,
    torch::Tensor const& mask,
    float scale_factor) {
  AT_ASSERTM(input.dim() == 4, "expected 4D tensor");
  AT_ASSERTM((input.scalar_type() == at::ScalarType::Half) ||
	     (input.scalar_type() == at::ScalarType::BFloat16), 
      "Only fp16 and bf16 are supported");
  AT_ASSERTM(mask.dim() == 4, "expected 4D tensor");

  return fwd_cuda(input, mask, scale_factor);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scaled_masked_softmax_forward",
        &fwd, 
	"self-multihead attention scaled masked softmax(forward)");
}
