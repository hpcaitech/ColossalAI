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

The following example of [Colossal-AI](https://github.com/hpcaitech/ColossalAI) demonstrates fine-tuning Causal Language Modelling at low cost.


## Our Modifications

We are using the pre-training weights of the OPT model provided by Hugging Face Hub on the raw WikiText-2 (no tokens were replaced before
the tokenization).

We adapt the OPT training code to ColossalAI by leveraging [Boosting API](https://colossalai.org/docs/basics/booster_api) loaded with a chosen plugin, where each plugin corresponds to a specific kind of training strategy. This example supports plugins including TorchDDPPlugin, LowLevelZeroPlugin, HybridParallelPlugin and GeminiPlugin.

## Run Demo

By running the following script:
```bash
bash run_demo.sh
```
You will finetune a [facebook/opt-350m](https://huggingface.co/facebook/opt-350m) model on this [dataset](https://huggingface.co/datasets/hugginglearners/netflix-shows), which contains more than 8000 comments on Netflix shows.

The script can be modified if you want to try another set of hyperparameters or change to another OPT model with different size.

The demo code is adapted from this [blog](https://medium.com/geekculture/fine-tune-eleutherai-gpt-neo-to-generate-netflix-movie-descriptions-in-only-47-lines-of-code-40c9b4c32475) and  the [HuggingFace Language Modelling examples](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling).



## Run Benchmark

You can run benchmark for OPT model by running the following script:
```bash
bash run_benchmark.sh
```
The script will test performance (throughput & peak memory usage) for each combination of hyperparameters. You can also play with this script to configure your set of hyperparameters for testing.
