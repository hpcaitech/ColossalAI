<h1 align="center">
  <img width="auto" height="100px", src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/logo_coati.png"/>
  <br/>
  <span>ColossalChat</span>
</h1>


## Table of Contents

- [Table of Contents](#table-of-contents)
- [What is ColossalChat and Coati ?](#what-is-colossalchat-and-coati-)
- [Online demo](#online-demo)
- [Install](#install)
  - [Install the environment](#install-the-environment)
  - [Install the Transformers](#install-the-transformers)
- [How to use?](#how-to-use)
  - [Supervised datasets collection](#supervised-datasets-collection)
  - [RLHF Training Stage1 - Supervised instructs tuning](#RLHF-training-stage1---supervised-instructs-tuning)
  - [RLHF Training Stage2 - Training reward model](#RLHF-training-stage2---training-reward-model)
  - [RLHF Training Stage3 - Training model with reinforcement learning by human feedback](#RLHF-training-stage3---training-model-with-reinforcement-learning-by-human-feedback)
  - [Inference Quantization and Serving - After Training](#inference-quantization-and-serving---after-training)
- [Coati7B examples](#coati7b-examples)
  - [Generation](#generation)
  - [Open QA](#open-qa)
  - [Limitation for LLaMA-finetuned models](#limitation)
  - [Limitation of dataset](#limitation)
- [FAQ](#faq)
  - [How to save/load checkpoint](#faq)
  - [How to train with limited resources](#faq)
- [The Plan](#the-plan)
  - [Real-time progress](#real-time-progress)
- [Invitation to open-source contribution](#invitation-to-open-source-contribution)
- [Quick Preview](#quick-preview)
- [Authors](#authors)
- [Citations](#citations)
- [Licenses](#licenses)
---
## What is ColossalChat and Coati ?

[ColossalChat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat) is the project to implement LLM with RLHF, powered by the [Colossal-AI](https://github.com/hpcaitech/ColossalAI) project.

Coati stands for `ColossalAI Talking Intelligence`. It is the name for the module implemented in this project and is also the name of the large language model developed by the ColossalChat project.

The Coati package provides a unified large language model framework that has implemented the following functions
- Supports comprehensive large-model training acceleration capabilities for ColossalAI, without requiring knowledge of complex distributed training algorithms
- Supervised datasets collection
- Supervised instructions fine-tuning
- Training reward model
- Reinforcement learning with human feedback
- Quantization inference
- Fast model deploying
- Perfectly integrated with the Hugging Face ecosystem, a high degree of model customization

<div align="center">
  <p align="center">
    <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chatgpt/chatgpt.png" width=700/>
  </p>

   Image source: https://openai.com/blog/chatgpt
</div>

**As Colossal-AI is undergoing some major updates, this project will be actively maintained to stay in line with the Colossal-AI project.**


More details can be found in the latest news.
* [2023/03] [ColossalChat: An Open-Source Solution for Cloning ChatGPT With a Complete RLHF Pipeline](https://medium.com/@yangyou_berkeley/colossalchat-an-open-source-solution-for-cloning-chatgpt-with-a-complete-rlhf-pipeline-5edf08fb538b)
* [2023/02] [Open Source Solution Replicates ChatGPT Training Process! Ready to go with only 1.6GB GPU Memory](https://www.hpc-ai.tech/blog/colossal-ai-chatgpt)

## Online demo
<div align="center">
   <a href="https://www.youtube.com/watch?v=HcTiHzApHm0">
   <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/ColossalChat%20YouTube.png" width="700" />
   </a>
</div>

[ColossalChat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat): An open-source solution for cloning [ChatGPT](https://openai.com/blog/chatgpt/) with a complete RLHF pipeline.
[[code]](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat)
[[blog]](https://medium.com/@yangyou_berkeley/colossalchat-an-open-source-solution-for-cloning-chatgpt-with-a-complete-rlhf-pipeline-5edf08fb538b)
[[demo]](https://www.youtube.com/watch?v=HcTiHzApHm0)
[[tutorial]](https://www.youtube.com/watch?v=-qFBZFmOJfg)

<p id="ColossalChat-Speed" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/ColossalChat%20Speed.jpg" width=450/>
</p>

> DeepSpeedChat performance comes from its blog on 2023 April 12, ColossalChat performance can be reproduced on an AWS p4d.24xlarge node with 8 A100-40G GPUs with the following command: torchrun --standalone --nproc_per_node 8 benchmark_opt_lora_dummy.py --num_collect_steps 1 --use_kernels --strategy colossalai_zero2 --experience_batch_size 64 --train_batch_size 32

## Install

### Install the environment

```shell
conda create -n coati
conda activate coati
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI/applications/Chat
pip install .
```

### Install the Transformers

```shell
pip install transformers==4.30.2
```

## How to use?

### Supervised datasets collection

we collected 104K bilingual datasets of Chinese and English, and you can find the datasets in this repo
[InstructionWild](https://github.com/XueFuzhao/InstructionWild)

Here is how we collected the data
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/data-collect.png" width=500/>
</p>

### RLHF Training Stage1 - Supervised instructs tuning

Stage1 is supervised instructs fine-tuning, which uses the datasets mentioned earlier to fine-tune the model.

You can run the `examples/train_sft.sh` to start a supervised instructs fine-tuning.
[[Stage1 tutorial video]](https://www.youtube.com/watch?v=-qFBZFmOJfg)

### RLHF Training Stage2 - Training reward model

Stage2 trains a reward model, which obtains corresponding scores by manually ranking different outputs for the same prompt and supervises the training of the reward model

You can run the `examples/train_rm.sh` to start a reward model training.
[[Stage2 tutorial video]](https://www.youtube.com/watch?v=gMx2CApKhuo)

### RLHF Training Stage3 - Training model with reinforcement learning by human feedback

Stage3 uses reinforcement learning algorithm, which is the most complex part of the training process:

<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/stage-3.jpeg" width=800/>
</p>

You can run the `examples/train_prompts.sh` to start training PPO with human feedback.
[[Stage3 tutorial video]](https://www.youtube.com/watch?v=Z8wwSHxPL9g)

For more details, see [`examples/`](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat/examples).

### Inference Quantization and Serving - After Training

We provide an online inference server and a benchmark. We aim to run inference on single GPU, so quantization is essential when using large models.

We support 8-bit quantization (RTN), 4-bit quantization (GPTQ), and  FP16 inference. You can
Online inference server scripts can help you deploy your own services.

For more details, see [`inference/`](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat/inference).

## Coati7B examples

### Generation

<details><summary><b>E-mail</b></summary>

![phd](https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/Phd.png)
</details>

<details><summary><b>coding</b></summary>

![sort](https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/quick_sort.png)

</details>

<details><summary><b>regex</b></summary>

![regex](https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/regex.png)

</details>

<details><summary><b>Tex</b></summary>

![tex](https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/tex.png)

</details>

<details><summary><b>writing</b></summary>

![writing](https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/writing.png)

</details>

<details><summary><b>Table</b></summary>

![Table](https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/table.png)

</details>

### Open QA
<details><summary><b>Game</b></summary>

![Game](https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/game.png)

</details>

<details><summary><b>Travel</b></summary>

![Travel](https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/travel.png)

</details>

<details><summary><b>Physical</b></summary>

![Physical](https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/physical.png)

</details>

<details><summary><b>Chemical</b></summary>

![Chemical](https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/chemical.png)

</details>

<details><summary><b>Economy</b></summary>

![Economy](https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/economy.png)

</details>

You can find more examples in this [repo](https://github.com/XueFuzhao/InstructionWild/blob/main/comparison.md).

### Limitation
<details><summary><b>Limitation for LLaMA-finetuned models</b></summary>
- Both Alpaca and ColossalChat are based on LLaMA. It is hard to compensate for the missing knowledge in the pre-training stage.
- Lack of counting ability: Cannot count the number of items in a list.
- Lack of Logics (reasoning and calculation)
- Tend to repeat the last sentence (fail to produce the end token).
- Poor multilingual results: LLaMA is mainly trained on English datasets (Generation performs better than QA).
</details>

<details><summary><b>Limitation of dataset</b></summary>
- Lack of summarization ability: No such instructions in finetune datasets.
- Lack of multi-turn chat: No such instructions in finetune datasets
- Lack of self-recognition: No such instructions in finetune datasets
- Lack of Safety:
  - When the input contains fake facts, the model makes up false facts and explanations.
  - Cannot abide by OpenAI's policy: When generating prompts from OpenAI API, it always abides by its policy. So no violation case is in the datasets.
</details>

## FAQ

<details><summary><b>How to save/load checkpoint</b></summary>

We have integrated the Transformers save and load pipeline, allowing users to freely call Hugging Face's language models and save them in the HF format.

```
from coati.models.llama import LlamaLM
from coati.trainer import SFTTrainer

model = LlamaLM(pretrained=args.pretrain)
tokenizer = AutoTokenizer.from_pretrained(args.pretrain)

(model, optim) = strategy.prepare((model, optim))
trainer = SFTTrainer(model=model,
    strategy=strategy,
    optim=optim,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    batch_size=args.batch_size,
    max_epochs=args.max_epochs,
    accumulation_steps = args.accumulation_steps
)

trainer.fit()
# this saves in pytorch format
strategy.save_model(model, args.save_path, only_rank0=True)

# this saves in HF format. ColossalAI strategy with stage-3 doesn't support this method
strategy.save_pretrained(model, args.save_path, only_rank0=True, tokenizer=tokenizer)
```

</details>

<details><summary><b>How to train with limited resources</b></summary>

Here are some examples that can allow you to train a 7B model on a single or multiple consumer-grade GPUs.

If you only have a single 24G GPU, you can use the following script. `batch_size`, `lora_rank` and `grad_checkpoint` are the most important parameters to successfully train the model.
```
torchrun --standalone --nproc_per_node=1 train_sft.py \
    --pretrain "/path/to/LLaMa-7B/" \
    --model 'llama' \
    --strategy ddp \
    --log_interval 10 \
    --save_path  /path/to/Coati-7B \
    --dataset /path/to/data.json \
    --batch_size 1 \
    --accumulation_steps 8 \
    --lr 2e-5 \
    --max_datasets_size 512 \
    --max_epochs 1 \
    --lora_rank 16 \
    --grad_checkpoint
```

`colossalai_gemini` strategy can enable a single 24G GPU to train the whole model without using LoRA if you have sufficient CPU memory. You can use the following script.
```
torchrun --standalone --nproc_per_node=1 train_sft.py \
    --pretrain "/path/to/LLaMa-7B/" \
    --model 'llama' \
    --strategy colossalai_gemini \
    --log_interval 10 \
    --save_path  /path/to/Coati-7B \
    --dataset /path/to/data.json \
    --batch_size 1 \
    --accumulation_steps 8 \
    --lr 2e-5 \
    --max_datasets_size 512 \
    --max_epochs 1 \
    --grad_checkpoint
```

If you have 4x32 GB GPUs, you can even train the whole 7B model using our `colossalai_zero2_cpu` strategy! The script is given as follows.
```
torchrun --standalone --nproc_per_node=4 train_sft.py \
    --pretrain "/path/to/LLaMa-7B/" \
    --model 'llama' \
    --strategy colossalai_zero2_cpu \
    --log_interval 10 \
    --save_path  /path/to/Coati-7B \
    --dataset /path/to/data.json \
    --batch_size 1 \
    --accumulation_steps 8 \
    --lr 2e-5 \
    --max_datasets_size 512 \
    --max_epochs 1 \
    --grad_checkpoint
```
</details>


## The Plan

- [x] implement PPO fine-tuning
- [x] implement training reward model
- [x] support LoRA
- [x] support inference
- [x] support llama from [facebook](https://github.com/facebookresearch/llama)
- [x] implement PPO-ptx fine-tuning
- [ ] integrate with Ray
- [ ] support more RL paradigms, like Implicit Language Q-Learning (ILQL),
- [ ] support chain-of-thought by [langchain](https://github.com/hwchase17/langchain)

### Real-time progress
You will find our progress in github project broad

[Coati](https://github.com/orgs/hpcaitech/projects/17/views/1)

## Invitation to open-source contribution
Referring to the successful attempts of [BLOOM](https://bigscience.huggingface.co/) and [Stable Diffusion](https://en.wikipedia.org/wiki/Stable_Diffusion), any and all developers and partners with computing powers, datasets, models are welcome to join and build the Colossal-AI community, making efforts towards the era of big AI models from the starting point of replicating ChatGPT!

You may contact us or participate in the following ways:
1. [Leaving a Star ⭐](https://github.com/hpcaitech/ColossalAI/stargazers) to show your like and support. Thanks!
2. Posting an [issue](https://github.com/hpcaitech/ColossalAI/issues/new/choose), or submitting a PR on GitHub follow the guideline in [Contributing](https://github.com/hpcaitech/ColossalAI/blob/main/CONTRIBUTING.md).
3. Join the Colossal-AI community on
[Slack](https://join.slack.com/t/colossalaiworkspace/shared_invite/zt-z7b26eeb-CBp7jouvu~r0~lcFzX832w),
and [WeChat(微信)](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/WeChat.png "qrcode") to share your ideas.
4. Send your official proposal to email contact@hpcaitech.com

Thanks so much to all of our amazing contributors!

## Quick Preview
<div align="center">
   <a href="https://chat.colossalai.org/">
   <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/Chat-demo.png" width="700" />
   </a>
</div>

- An open-source low cost solution for cloning [ChatGPT](https://openai.com/blog/chatgpt/) with a complete RLHF pipeline. [[demo]](https://chat.colossalai.org)

<p id="ChatGPT_scaling" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chatgpt/ChatGPT%20scaling.png" width=800/>
</p>

- Up to 7.73 times faster for single server training and 1.42 times faster for single-GPU inference

<p id="ChatGPT-1GPU" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chatgpt/ChatGPT-1GPU.jpg" width=450/>
</p>

- Up to 10.3x growth in model capacity on one GPU
- A mini demo training process requires only 1.62GB of GPU memory (any consumer-grade GPU)

<p id="inference" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chatgpt/LoRA%20data.jpg" width=600/>
</p>

- Increase the capacity of the fine-tuning model by up to 3.7 times on a single GPU
- Keep in a sufficiently high running speed

|  Model Pair   | Alpaca-7B ⚔ Coati-7B | Coati-7B ⚔ Alpaca-7B |
| :-----------: | :------------------: | :------------------: |
| Better Cases  |     38 ⚔ **41**      |     **45** ⚔ 33      |
|   Win Rate    |    48% ⚔ **52%**     |    **58%** ⚔ 42%     |
| Average Score |   7.06 ⚔ **7.13**    |   **7.31** ⚔ 6.82    |
- Our Coati-7B model performs better than Alpaca-7B when using GPT-4 to evaluate model performance. The Coati-7B model we evaluate is an old version we trained a few weeks ago and the new version is around the corner.

## Authors

Coati is developed by ColossalAI Team:
- [Fazzie](https://fazzie-key.cool/about/index.html)
- [FrankLeeeee](https://github.com/FrankLeeeee)
- [BlueRum](https://github.com/ht-zhou)
- [ver217](https://github.com/ver217)
- [ofey404](https://github.com/ofey404)

The Phd student from [(HPC-AI) Lab](https://ai.comp.nus.edu.sg/) also contributed a lot to this project.
- [Zangwei Zheng](https://github.com/zhengzangw)
- [Xue Fuzhao](https://github.com/XueFuzhao)

## Citations

```bibtex
@article{Hu2021LoRALA,
    title   = {LoRA: Low-Rank Adaptation of Large Language Models},
    author  = {Edward J. Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Weizhu Chen},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2106.09685}
}

@article{ouyang2022training,
  title={Training language models to follow instructions with human feedback},
  author={Ouyang, Long and Wu, Jeff and Jiang, Xu and Almeida, Diogo and Wainwright, Carroll L and Mishkin, Pamela and Zhang, Chong and Agarwal, Sandhini and Slama, Katarina and Ray, Alex and others},
  journal={arXiv preprint arXiv:2203.02155},
  year={2022}
}

@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and Martinet, Xavier and Lachaux, Marie-Anne and Lacroix, Timoth{\'e}e and Rozi{\`e}re, Baptiste and Goyal, Naman and Hambro, Eric and Azhar, Faisal and Rodriguez, Aurelien and Joulin, Armand and Grave, Edouard and Lample, Guillaume},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}

@misc{alpaca,
  author = {Rohan Taori and Ishaan Gulrajani and Tianyi Zhang and Yann Dubois and Xuechen Li and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {Stanford Alpaca: An Instruction-following LLaMA model},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/stanford_alpaca}},
}

@misc{instructionwild,
  author = {Fuzhao Xue and Zangwei Zheng and Yang You },
  title = {Instruction in the Wild: A User-based Instruction Dataset},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/XueFuzhao/InstructionWild}},
}
```

## Licenses

Coati is licensed under the [Apache 2.0 License](LICENSE).
