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
  - [Supervised datasets collection](#step-1-data-collection)
  - [RLHF Training Stage1 - Supervised instructs tuning](#rlhf-training-stage1---supervised-instructs-tuning)
  - [RLHF Training Stage2 - Training reward model](#rlhf-training-stage2---training-reward-model)
  - [RLHF Training Stage3 - Training model with reinforcement learning by human feedback](#rlhf-training-stage3---proximal-policy-optimization)
  - [Inference Quantization and Serving - After Training](#inference-quantization-and-serving---after-training)
- [Coati7B examples](#coati7b-examples)
  - [Generation](#generation)
  - [Open QA](#open-qa)
  - [Limitation for LLaMA-finetuned models](#limitation)
  - [Limitation of dataset](#limitation)
- [Alternative Option For RLHF: DPO](#alternative-option-for-rlhf-direct-preference-optimization)
- [Alternative Option For RLHF: SimPO](#alternative-option-for-rlhf-simple-preference-optimization-simpo)
- [Alternative Option For RLHF: ORPO](#alternative-option-for-rlhf-odds-ratio-preference-optimization-orpo)
- [Alternative Option For RLHF: KTO](#alternative-option-for-rlhf-kahneman-tversky-optimization-kto)
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

## What Is ColossalChat And Coati ?

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

- [2023/03] [ColossalChat: An Open-Source Solution for Cloning ChatGPT With a Complete RLHF Pipeline](https://medium.com/@yangyou_berkeley/colossalchat-an-open-source-solution-for-cloning-chatgpt-with-a-complete-rlhf-pipeline-5edf08fb538b)
- [2023/02] [Open Source Solution Replicates ChatGPT Training Process! Ready to go with only 1.6GB GPU Memory](https://www.hpc-ai.tech/blog/colossal-ai-chatgpt)

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

> DeepSpeedChat performance comes from its blog on 2023 April 12, ColossalChat performance can be reproduced on an AWS p4d.24xlarge node with 8 A100-40G GPUs with the following command: `torchrun --standalone --nproc_per_node 8 benchmark_opt_lora_dummy.py --num_collect_steps 1 --use_kernels --strategy colossalai_zero2 --experience_batch_size 64 --train_batch_size 32`

## Install

### Install the Environment

```bash
# Create new environment
conda create -n colossal-chat python=3.10.9 (>=3.8.7)
conda activate colossal-chat

# Clone ColossalAI
git clone https://github.com/hpcaitech/ColossalAI.git

# Install ColossalAI, make sure you have torch installed before using BUILD_EXT=1.
cd $COLOSSAL_AI_ROOT
BUILD_EXT=1 pip install .

# Install ColossalChat
cd $COLOSSAL_AI_ROOT/applications/ColossalChat
pip install .
```

## How To Use?

### RLHF Training Stage1 - Supervised Instructs Tuning

Stage1 is supervised instructs fine-tuning (SFT). This step is a crucial part of the RLHF training process, as it involves training a machine learning model using human-provided instructions to learn the initial behavior for the task at hand. Here's a detailed guide on how to SFT your LLM with ColossalChat. More details can be found in [example guideline](./examples/README.md).

#### Step 1: Data Collection
The first step in Stage 1 is to collect a dataset of human demonstrations of the following format.

```json
[
    {"messages":
      [
        {
          "from": "user",
          "content": "what are some pranks with a pen i can do?"
        },
        {
          "from": "assistant",
          "content": "Are you looking for practical joke ideas?"
        },
      ]
    },
]
```

#### Step 2: Preprocessing
Once you have collected your SFT dataset, you will need to preprocess it. This involves four steps: data cleaning, data deduplication, formatting and tokenization. In this section, we will focus on formatting and tokenization.

In this code, we provide a flexible way for users to set the conversation template for formatting chat data using Huggingface's newest feature--- chat template. Please follow the [example guideline](./examples/README.md) on how to format and tokenize data.

#### Step 3: Training
Choose a suitable model architecture for your task. Note that your model should be compatible with the tokenizer that you used to tokenize the SFT dataset. You can run [train_sft.sh](./examples/training_scripts/train_sft.sh) to start a supervised instructs fine-tuning. More details can be found in [example guideline](./examples/README.md).

### RLHF Training Stage2 - Training Reward Model

Stage2 trains a reward model, which obtains corresponding scores by manually ranking different outputs for the same prompt and supervises the training of the reward model.

#### Step 1: Data Collection
Below shows the preference dataset format used in training the reward model.

```json
[
    {"context": [
        {
          "from": "human",
          "content": "Introduce butterflies species in Oregon."
        }
      ],
      "chosen": [
        {
          "from": "assistant",
          "content": "About 150 species of butterflies live in Oregon, with about 100 species are moths..."
        },
      ],
      "rejected": [
        {
          "from": "assistant",
          "content": "Are you interested in just the common butterflies?  There are a few common ones which will be easy to find..."
        },
      ]
    },
]
```

#### Step 2: Preprocessing
Similar to the second step in the previous stage, we format the reward data into the same structured format as used in step 2 of the SFT stage. You can run [prepare_preference_dataset.sh](./examples/data_preparation_scripts/prepare_preference_dataset.sh) to prepare the preference data for reward model training.

#### Step 3: Training
You can run [train_rm.sh](./examples/training_scripts/train_rm.sh) to start the reward model training. More details can be found in [example guideline](./examples/README.md).

### RLHF Training Stage3 - Proximal Policy Optimization

In stage3 we will use reinforcement learning algorithm--- Proximal Policy Optimization (PPO), which is the most complex part of the training process:

<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/stage-3.jpeg" width=800/>
</p>

#### Step 1: Data Collection
PPO uses two kind of training data--- the prompt data and the sft data (optional). The first dataset is mandatory, data samples within the prompt dataset ends with a line from "human" and thus the "assistant" needs to generate a response to answer to the "human". Note that you can still use conversation that ends with a line from the "assistant", in that case, the last line will be dropped. Here is an example of the prompt dataset format.

```json
[
    {"messages":
      [
        {
          "from": "human",
          "content": "what are some pranks with a pen i can do?"
        }
      ]
    },
]
```

#### Step 2: Data Preprocessing
To prepare the prompt dataset for PPO training, simply run [prepare_prompt_dataset.sh](./examples/data_preparation_scripts/prepare_prompt_dataset.sh)

#### Step 3: Training
You can run the [train_ppo.sh](./examples/training_scripts/train_ppo.sh) to start PPO training. Here are some unique arguments for PPO, please refer to the training configuration section for other training configuration. More detais can be found in [example guideline](./examples/README.md).

```bash
--pretrain $PRETRAINED_MODEL_PATH \
--rm_pretrain $PRETRAINED_MODEL_PATH \ # reward model architectual
--tokenizer_dir $PRETRAINED_TOKENIZER_PATH \
--rm_checkpoint_path $REWARD_MODEL_PATH \ # reward model checkpoint path
--prompt_dataset ${prompt_dataset[@]} \ # List of string, the prompt dataset
--ptx_dataset ${ptx_dataset[@]} \ # List of string, the SFT data used in the SFT stage
--ptx_batch_size 1 \ # batch size for calculate ptx loss
--ptx_coef 0.0 \ # none-zero if ptx loss is enable
--num_episodes 2000 \ # number of episodes to train
--num_collect_steps 1 \
--num_update_steps 1 \
--experience_batch_size 8 \
--train_batch_size 4 \
--accumulation_steps 2
```

Each episode has two phases, the collect phase and the update phase. During the collect phase, we will collect experiences (answers generated by actor), store those in ExperienceBuffer. Then data in ExperienceBuffer is used during the update phase to update parameter of actor and critic.

- Without tensor parallelism,
```
experience buffer size
= num_process * num_collect_steps * experience_batch_size
= train_batch_size * accumulation_steps * num_process
```

- With tensor parallelism,
```
num_tp_group = num_process / tp
experience buffer size
= num_tp_group * num_collect_steps * experience_batch_size
= train_batch_size * accumulation_steps * num_tp_group
```

## Alternative Option For RLHF: Direct Preference Optimization (DPO)
For those seeking an alternative to Reinforcement Learning from Human Feedback (RLHF), Direct Preference Optimization (DPO) presents a compelling option. DPO, as detailed in this [paper](https://arxiv.org/abs/2305.18290), DPO offers an low-cost way to perform RLHF and usually request less computation resources compares to PPO. Read this [README](./examples/README.md) for more information.

### DPO Training Stage1 - Supervised Instructs Tuning

Please refer the [sft section](#dpo-training-stage1---supervised-instructs-tuning) in the PPO part.

### DPO Training Stage2 - DPO Training
#### Step 1: Data Collection & Preparation
For DPO training, you only need the preference dataset. Please follow the instruction in the [preference dataset preparation section](#rlhf-training-stage2---training-reward-model) to prepare the preference data for DPO training.

#### Step 2: Training
You can run the [train_dpo.sh](./examples/training_scripts/train_dpo.sh) to start DPO training. More detais can be found in [example guideline](./examples/README.md).

## Alternative Option For RLHF: Simple Preference Optimization (SimPO)
Simple Preference Optimization (SimPO) from this [paper](https://arxiv.org/pdf/2405.14734) is similar to DPO but it abandons the use of the reference model, which makes the training more efficient. It also adds a reward shaping term called target reward margin to enhance training stability. It also use length normalization to better align with the inference process. Read this [README](./examples/README.md) for more information.

## Alternative Option For RLHF: Odds Ratio Preference Optimization (ORPO)
Odds Ratio Preference Optimization (ORPO) from this [paper](https://arxiv.org/pdf/2403.07691) is a reference model free alignment method that use a mixture of SFT loss and a reinforcement leanring loss calculated based on odds-ratio-based implicit reward to makes the training more efficient and stable. Read this [README](./examples/README.md) for more information.

## Alternative Option For RLHF: Kahneman-Tversky Optimization (KTO)
We support the method introduced in the paper [KTO:Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/pdf/2402.01306) (KTO). Which is a aligment method that directly maximize "human utility" of generation results. Read this [README](./examples/README.md) for more information.

### Inference Quantization and Serving - After Training

We provide an online inference server and a benchmark. We aim to run inference on single GPU, so quantization is essential when using large models.

We support 8-bit quantization (RTN), 4-bit quantization (GPTQ), and FP16 inference.

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

- Option 1: Save the model weights, model config and generation config (Note: tokenizer will not be saved) which can be loaded using HF's from_pretrained method.
```python
# if use lora, you can choose to merge lora weights before saving
if args.lora_rank > 0 and args.merge_lora_weights:
        from coati.models.lora import LORA_MANAGER

        # NOTE: set model to eval to merge LoRA weights
        LORA_MANAGER.merge_weights = True
        model.eval()
# save model checkpoint after fitting on only rank0
booster.save_model(model, os.path.join(args.save_dir, "modeling"), shard=True)

```

- Option 2: Save the model weights, model config, generation config, as well as the optimizer, learning rate scheduler, running states (Note: tokenizer will not be saved) which are needed for resuming training.
```python
from coati.utils import save_checkpoint
# save model checkpoint after fitting on only rank0
save_checkpoint(
        save_dir=actor_save_dir,
        booster=actor_booster,
        model=model,
        optimizer=optim,
        lr_scheduler=lr_scheduler,
        epoch=0,
        step=step,
        batch_size=train_batch_size,
        coordinator=coordinator,
    )
```
To load the saved checkpoint
```python
from coati.utils import load_checkpoint
start_epoch, start_step, sampler_start_idx = load_checkpoint(
        load_dir=checkpoint_path,
        booster=booster,
        model=model,
        optimizer=optim,
        lr_scheduler=lr_scheduler,
    )
```
</details>

<details><summary><b>How to train with limited resources</b></summary>

Here are some suggestions that can allow you to train a 7B model on a single or multiple consumer-grade GPUs.

`batch_size`, `lora_rank` and `grad_checkpoint` are the most important parameters to successfully train the model. To maintain a descent batch size for gradient calculation, consider increase the accumulation_step and reduce the batch_size on each rank.

If you only have a single 24G GPU. Generally, using lora and "zero2-cpu" will be sufficient.

`gemini` and `gemini-auto` can enable a single 24G GPU to train the whole model without using LoRA if you have sufficient CPU memory. But that strategy doesn't support gradient accumulation.

If you have multiple GPUs each has very limited VRAM, say 8GB. You can try the `3d` for the plugin option, which supports tensor parellelism, set `--tp` to the number of GPUs that you have.
</details>

### Real-time progress

You will find our progress in github [project broad](https://github.com/orgs/hpcaitech/projects/17/views/1).

## Invitation to open-source contribution

Referring to the successful attempts of [BLOOM](https://bigscience.huggingface.co/) and [Stable Diffusion](https://en.wikipedia.org/wiki/Stable_Diffusion), any and all developers and partners with computing powers, datasets, models are welcome to join and build the Colossal-AI community, making efforts towards the era of big AI models from the starting point of replicating ChatGPT!

You may contact us or participate in the following ways:

1. [Leaving a Star ⭐](https://github.com/hpcaitech/ColossalAI/stargazers) to show your like and support. Thanks!
2. Posting an [issue](https://github.com/hpcaitech/ColossalAI/issues/new/choose), or submitting a PR on GitHub follow the guideline in [Contributing](https://github.com/hpcaitech/ColossalAI/blob/main/CONTRIBUTING.md).
3. Join the Colossal-AI community on
   [Slack](https://github.com/hpcaitech/public_assets/tree/main/colossalai/contact/slack),
   and [WeChat(微信)](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/WeChat.png "qrcode") to share your ideas.
4. Send your official proposal to email contact@hpcaitech.com

Thanks so much to all of our amazing contributors!

## Quick Preview

<div align="center">
   <a href="https://chat.colossalai.org/">
   <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/Chat-demo.png" width="700" />
   </a>
</div>

- An open-source low-cost solution for cloning [ChatGPT](https://openai.com/blog/chatgpt/) with a complete RLHF pipeline. [[demo]](https://chat.colossalai.org)

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

- [ver217](https://github.com/ver217) Leading the project while contributing to the main framework.
- [FrankLeeeee](https://github.com/FrankLeeeee) Providing ML infra support and also taking charge of both front-end and back-end development.
- [htzhou](https://github.com/ht-zhou) Contributing to the algorithm and development for RM and PPO training.
- [Fazzie](https://fazzie-key.cool/about/index.html) Contributing to the algorithm and development for SFT.
- [ofey404](https://github.com/ofey404) Contributing to both front-end and back-end development.
- [Wenhao Chen](https://github.com/CWHer) Contributing to subsequent code enhancements and performance improvements.
- [Anbang Ye](https://github.com/YeAnbang) Contributing to the refactored PPO version with updated acceleration framework. Add support for DPO, SimPO, ORPO.

The PhD student from [(HPC-AI) Lab](https://ai.comp.nus.edu.sg/) also contributed a lot to this project.
- [Zangwei Zheng](https://github.com/zhengzangw)
- [Xue Fuzhao](https://github.com/XueFuzhao)

We also appreciate the valuable suggestions provided by [Jian Hu](https://github.com/hijkzzz) regarding the convergence of the PPO algorithm.

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

@misc{meng2024simposimplepreferenceoptimization,
      title={SimPO: Simple Preference Optimization with a Reference-Free Reward},
      author={Yu Meng and Mengzhou Xia and Danqi Chen},
      year={2024},
      eprint={2405.14734},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.14734},
}

@misc{rafailov2023directpreferenceoptimizationlanguage,
      title={Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
      author={Rafael Rafailov and Archit Sharma and Eric Mitchell and Stefano Ermon and Christopher D. Manning and Chelsea Finn},
      year={2023},
      eprint={2305.18290},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2305.18290},
}

@misc{hong2024orpomonolithicpreferenceoptimization,
      title={ORPO: Monolithic Preference Optimization without Reference Model},
      author={Jiwoo Hong and Noah Lee and James Thorne},
      year={2024},
      eprint={2403.07691},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2403.07691},
}
```

## Licenses

Coati is licensed under the [Apache 2.0 License](LICENSE).
