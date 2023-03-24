# Examples

## Install requirements

```shell
pip install -r requirements.txt
```

## Train the reward model (Stage 2)
Use these code to train your reward model.
```shell
# Take naive reward model training with opt-350m as example
python train_reward_model.py --pretrain "facebook/opt-350m" --model 'opt' --strategy naive
# use colossalai_zero2
torchrun --standalone --nproc_per_node=2 train_reward_model.py --pretrain "facebook/opt-350m" --model 'opt' --strategy colossalai_zero2
```

### Features and tricks in RM training
- We support [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) and [rm-static](https://huggingface.co/datasets/Dahoas/rm-static) datasets.
- We support 2 kinds of loss_function named 'log_sig'(used by OpenAI) and 'log_exp'(used by Anthropic).
- We change the loss to valid_acc and pair_dist to monitor progress during training.
- We add special token to the end of the sequence to get better result.
- We use cosine-reducing lr-scheduler for RM training.
- We set value_head as 1 liner layer and initialize the weight of value_head using N(0，1/(d_model + 1)) distribution.
- We train a Bloom-560m reward model for 1 epoch and find the test acc of the model achieve the performance mentions in [Anthropics paper](https://arxiv.org/abs/2204.05862).

### Experiment result
Model performance in [Anthropics paper](https://arxiv.org/abs/2204.05862):

<div align=center> <img width="512" alt="image" src="https://user-images.githubusercontent.com/70618399/225263321-8d64c3a8-6877-4cc8-9b61-0e1c52d3d94f.png">

<div align=left>Our training & test result of bloom-560m for 1 epoch:

<div align=center> <img width="512" alt="image" src="https://user-images.githubusercontent.com/70618399/225262950-a7f0a686-25de-44ec-98f2-11b83ea86674.png">

<div align=left>

## Train with dummy prompt data (Stage 3)

This script supports 4 kinds of strategies:

- naive
- ddp
- colossalai_zero2
- colossalai_gemini

It uses random generated prompt data.

Naive strategy only support single GPU training:

```shell
python train_dummy.py --strategy naive
# display cli help
python train_dummy.py -h
```

DDP strategy and ColossalAI strategy support multi GPUs training:

```shell
# run DDP on 2 GPUs
torchrun --standalone --nproc_per_node=2 train_dummy.py --strategy ddp
# run ColossalAI on 2 GPUs
torchrun --standalone --nproc_per_node=2 train_dummy.py --strategy colossalai_zero2
```

## Train with real prompt data (Stage 3)

We use [awesome-chatgpt-prompts](https://huggingface.co/datasets/fka/awesome-chatgpt-prompts) as example dataset. It is a small dataset with hundreds of prompts.

You should download `prompts.csv` first.

This script also supports 4 strategies.

```shell
# display cli help
python train_dummy.py -h
# run naive on 1 GPU
python train_prompts.py prompts.csv --strategy naive
# run DDP on 2 GPUs
torchrun --standalone --nproc_per_node=2 train_prompts.py prompts.csv --strategy ddp
# run ColossalAI on 2 GPUs
torchrun --standalone --nproc_per_node=2 train_prompts.py prompts.csv --strategy colossalai_zero2
```

## Inference example(After Stage3)
We support naive inference demo after training.
```shell
# inference, using pretrain path to configure model
python inference.py --model_path <your actor model path> --model <your model type> --pretrain <your pretrain model name/path>
# example
python inference.py --model_path ./actor_checkpoint_prompts.pt --pretrain bigscience/bloom-560m --model bloom
```

## Attention
The examples is just a demo for testing our progress of RM and PPO training.


#### data
- [x] [rm-static](https://huggingface.co/datasets/Dahoas/rm-static)
- [x] [hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [ ] [openai/summarize_from_feedback](https://huggingface.co/datasets/openai/summarize_from_feedback)
- [ ] [openai/webgpt_comparisons](https://huggingface.co/datasets/openai/webgpt_comparisons)
- [ ] [Dahoas/instruct-synthetic-prompt-responses](https://huggingface.co/datasets/Dahoas/instruct-synthetic-prompt-responses)

## Support Model

### GPT
- [x]  GPT2-S (s)
- [x]  GPT2-M (m)
- [x]  GPT2-L (l)
- [ ]  GPT2-XL (xl)
- [x]  GPT2-4B (4b)
- [ ]  GPT2-6B (6b)
- [ ]  GPT2-8B (8b)
- [ ]  GPT2-10B (10b)
- [ ]  GPT2-12B (12b)
- [ ]  GPT2-15B (15b)
- [ ]  GPT2-18B (18b)
- [ ]  GPT2-20B (20b)
- [ ]  GPT2-24B (24b)
- [ ]  GPT2-28B (28b)
- [ ]  GPT2-32B (32b)
- [ ]  GPT2-36B (36b)
- [ ]  GPT2-40B (40b)
- [ ]  GPT3 (175b)

### BLOOM
- [x] [BLOOM-560m](https://huggingface.co/bigscience/bloom-560m)
- [x] [BLOOM-1b1](https://huggingface.co/bigscience/bloom-1b1)
- [x] [BLOOM-3b](https://huggingface.co/bigscience/bloom-3b)
- [x] [BLOOM-7b](https://huggingface.co/bigscience/bloom-7b1)
- [ ] BLOOM-175b

### OPT
- [x] [OPT-125M](https://huggingface.co/facebook/opt-125m)
- [x] [OPT-350M](https://huggingface.co/facebook/opt-350m)
- [ ] [OPT-1.3B](https://huggingface.co/facebook/opt-1.3b)
- [ ] [OPT-2.7B](https://huggingface.co/facebook/opt-2.7b)
- [ ] [OPT-6.7B](https://huggingface.co/facebook/opt-6.7b)
- [ ] [OPT-13B](https://huggingface.co/facebook/opt-13b)
- [ ] [OPT-30B](https://huggingface.co/facebook/opt-30b)
