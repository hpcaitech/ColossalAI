<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

## OPT
Meta recently released [Open Pretrained Transformer (OPT)](https://github.com/facebookresearch/metaseq), a 175-Billion parameter AI language model, which stimulates AI programmers to perform various downstream tasks and application deployments.

The following example of [Colossal-AI](https://github.com/hpcaitech/ColossalAI) demonstrates fine-tuning Casual Language Modelling at low cost.

We are using the pre-training weights of the OPT model provided by Hugging Face Hub on the raw WikiText-2 (no tokens were replaced before
the tokenization). This training script is adapted from the [HuggingFace Language Modelling examples](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling).

## Our Modifications
We adapt the OPT training code to ColossalAI by leveraging Gemini and ZeRO DDP.

## Quick Start
You can launch training by using the following bash script

```bash
bash ./run_gemini.sh
```
