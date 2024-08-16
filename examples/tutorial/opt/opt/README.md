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
# Train OPT model with Colossal-AI


## OPT
Meta recently released [Open Pretrained Transformer (OPT)](https://github.com/facebookresearch/metaseq), a 175-Billion parameter AI language model, which stimulates AI programmers to perform various downstream tasks and application deployments.

The following example of [Colossal-AI](https://github.com/hpcaitech/ColossalAI) demonstrates fine-tuning causal Language Modelling at low cost.

We are using the pre-training weights of the OPT model provided by Hugging Face Hub on the raw WikiText-2 (no tokens were replaced before
the tokenization). This training script is adapted from the [HuggingFace Language Modelling examples](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling).

## Our Modifications
We adapt the OPT training code to ColossalAI by leveraging Gemini and ZeRO DDP.

## 🚀Quick Start for Tutorial
1. Install the dependency
```bash
pip install datasets accelerate
```
2. Run finetuning with synthetic datasets with one GPU
```bash
bash ./run_clm_synthetic.sh
```
3. Run finetuning with 4 GPUs
```bash
bash ./run_clm_synthetic.sh 16 0 125m 4
```

## Quick Start for Practical Use
You can launch training by using the following bash script

```bash
bash ./run_clm.sh <batch-size-per-gpu> <mem-cap> <model> <gpu-num>
```

- batch-size-per-gpu: number of samples fed to each GPU, default is 16
- mem-cap: limit memory usage within a value in GB, default is 0 (no limit)
- model: the size of the OPT model, default is `6.7b`. Acceptable values include `125m`, `350m`, `1.3b`, `2.7b`, `6.7`, `13b`, `30b`, `66b`. For `175b`, you can request
the pretrained weights from [OPT weight downloading page](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT).
- gpu-num: the number of GPUs to use, default is 1.

It uses `wikitext` dataset.

To use synthetic dataset:

```bash
bash ./run_clm_synthetic.sh <batch-size-per-gpu> <mem-cap> <model> <gpu-num>
```

## Remarkable Performance
On a single GPU, Colossal-AI’s automatic strategy provides remarkable performance gains from the ZeRO Offloading strategy by Microsoft DeepSpeed.
Users can experience up to a 40% speedup, at a variety of model scales. However, when using a traditional deep learning training framework like PyTorch, a single GPU can no longer support the training of models at such a scale.

<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/OPT.png" width=1000/>
</p>

Adopting the distributed training strategy with 8 GPUs is as simple as adding a `-nprocs 8` to the training command of Colossal-AI!

More details about behind the scenes can be found on the corresponding [blog](https://medium.com/@yangyou_berkeley/colossal-ai-seamlessly-accelerates-large-models-at-low-costs-with-hugging-face-4d1a887e500d),
and a detailed tutorial will be added in [Documentation](https://www.colossalai.org/docs/get_started/installation) very soon.
