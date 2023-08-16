/* ---------------- LICENSE FOR Colossal-AI ----------------
# coding=utf-8
# Copyright 2021- HPC-AI Technology Inc. team 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
*/
#include <torch/extension.h>

void rotary_embedding_neox(
  torch::Tensor& positions,
  torch::Tensor& query,
  torch::Tensor& key,
  int head_size,
  torch::Tensor& cos_sin_cache);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "rotary_embedding_neox",
    &rotary_embedding_neox,
    "Apply GPT-NeoX style rotary embedding to query and key");
}
