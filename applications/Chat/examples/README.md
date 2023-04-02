# Examples

## Install requirements

```shell
pip install -r requirements.txt
```

## Supervised datasets collection

We colllected 104K bilingual dataset of Chinese and English, and you can find the datasets in this repo
[InstructionWild](https://github.com/XueFuzhao/InstructionWild).

The following pic shows how we collected the data.
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/data-collect.png" width=500/>
</p>

## Stage1 - Supervised instructs tuning

Stage1 is supervised instructs fine-tuning, which uses the datasets mentioned earlier to fine-tune the model.

You can run the `examples/train_sft.sh` to start a supervised instructs fine-tuning.

You can also use the following cmd to start a supervised instructs fine-tuning with your own settings.
```
torchrun --standalone --nproc_per_node=4 train_sft.py \
    --pretrain "/path/to/LLaMa-7B/" \
    --model 'llama' \
    --strategy colossalai_zero2 \
    --log_interval 10 \
    --save_path  /path/to/Coati-7B \
    --dataset /path/to/data.json \
    --batch_size 4 \
    --accimulation_steps 8 \
    --lr 2e-5 \
    --max_datasets_size 512 \
    --max_epochs 1 \
```
### Arg List
- --strategy:          the strategy using for training, choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2'], default='naive'
- --model:             model type, choices=['gpt2', 'bloom', 'opt', 'llama'], default='bloom'
- --pretrain:          pretrain model, type=str, default=None
- --max_datasets_size: the max size of dataset, type=int, default=None
- --save_path:         path to save the model, type=str, default='output'
- --need_optim_ckpt:   whether to save optim ckpt, type=bool, default=False
- --max_epochs:        max epochs for training, type=int, default=3
- --batch_size:        batch size while training, type=int, default=4
- --lora_rank:         low-rank adaptation matrices rank, type=int, default=0
- --log_interval:      how many steps to log, type=int, default=100

## Stage2 - Training reward model

We train a reward model in stage 2, which obtains corresponding scores by manually ranking different outputs for the same prompt and supervises the training of the reward model.

You can run the `examples/train_rm.sh` to start a reward model training.

You can also use the following cmd to start training a reward model.
```
torchrun --standalone --nproc_per_node=4 train_reward_model.py
    --pretrain "/path/to/LLaMa-7B/" \
    --model 'llama' \
    --strategy colossalai_zero2 \
    --loss_fn 'log_exp'\
    --save_path 'rmstatic.pt' \
```
### Features and tricks in RM training
- We support [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)and[rm-static](https://huggingface.co/datasets/Dahoas/rm-static) datasets.
- We support 2 kinds of loss_function named 'log_sig'(used by OpenAI) and 'log_exp'(used by Anthropic).
- We change the loss to valid_acc and pair_dist to monitor progress during training.
- We add special token to the end of the sequence to get better result.
- We use cosine-reducing lr-scheduler for RM training.
- We set value_head as 1 liner layer and initialize the weight of value_head using N(0，1/(d_model + 1)) distribution.
- We train a Bloom-560m reward model for 1 epoch and find the test acc of the model achieve the performance mentions in [Anthropics paper](https://arxiv.org/abs/2204.05862).

### Experiment result
Model performance in [Anthropics paper](https://arxiv.org/abs/2204.05862):

<div align=middle> <img width="512" alt="image" src="https://user-images.githubusercontent.com/70618399/225263321-8d64c3a8-6877-4cc8-9b61-0e1c52d3d94f.png">

<div align=left>Our training & test result of bloom-560m for 1 epoch:

<div align=middle> <img width="512" alt="image" src="https://user-images.githubusercontent.com/70618399/225262950-a7f0a686-25de-44ec-98f2-11b83ea86674.png">

<div align=left>We also train the reward model based on LLaMA-7B, which reaches the ACC of 72.06% after 1 epoch, performing almost the same as Anthropic's best RM.

### Arg List
- --strategy:          the strategy using for training, choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2'], default='naive'
- --model:             model type, choices=['gpt2', 'bloom', 'opt', 'llama'], default='bloom'
- --pretrain:          pretrain model, type=str, default=None
- --model_path:        the path of rm model(if continue to train), type=str, default=None
- --save_path:         path to save the model, type=str, default='output'
- --need_optim_ckpt:   whether to save optim ckpt, type=bool, default=False
- --max_epochs:        max epochs for training, type=int, default=3
- --dataset:           dataset name, type=str, choices=['Anthropic/hh-rlhf', 'Dahoas/rm-static']
- --subset:            subset of the dataset, type=str, default=None
- --batch_size:        batch size while training, type=int, default=4
- --lora_rank:         low-rank adaptation matrices rank, type=int, default=0
- --loss_func:         which kind of loss function, choices=['log_sig', 'log_exp']
- --max_len:           max sentence length for generation, type=int, default=512
- --test:              whether is only tesing, if it's ture, the dataset will be small

## Stage3 - Training model using prompts with RL

Stage3 uses reinforcement learning algorithm, which is the most complex part of the training process, as shown below:

<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/stage-3.jpeg" width=800/>
</p>

You can run the `examples/train_prompts.sh` to start PPO training.
You can also use the cmd following to start PPO training.

```
torchrun --standalone --nproc_per_node=4 train_prompts.py \
         --pretrain "/path/to/LLaMa-7B/" \
         --model 'llama' \
         --strategy colossalai_zero2 \
         --prompt_path /path/to/your/prompt_dataset \
         --pretrain_dataset /path/to/your/pretrain_dataset \
         --rm_pretrain /your/pretrain/rm/defination \
         --rm_path /your/rm/model/path
```
### Arg List
- --strategy:          the strategy using for training, choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2'], default='naive'
- --model:             model type of actor, choices=['gpt2', 'bloom', 'opt', 'llama'], default='bloom'
- --pretrain:          pretrain model, type=str, default=None
- --rm_model:          reward model type, type=str, choices=['gpt2', 'bloom', 'opt', 'llama'], default=None
- --rm_pretrain:       pretrain model for reward model, type=str, default=None
- --rm_path:           the path of rm model, type=str, default=None
- --save_path:         path to save the model, type=str, default='output'
- --prompt_path:       path of the prompt dataset, type=str, default=None
- --pretrain_dataset:  path of the ptx dataset, type=str, default=None
- --need_optim_ckpt:   whether to save optim ckpt, type=bool, default=False
- --num_episodes:      num of episodes for training, type=int, default=10
- --max_epochs:        max epochs for training in one episode, type=int, default=5
- --max_timesteps:     max episodes in one batch, type=int, default=10
- --update_timesteps:  timesteps to update, type=int, default=10
- --train_batch_size:  batch size while training, type=int, default=8
- --ptx_batch_size:    batch size to compute ptx loss, type=int, default=1
- --experience_batch_size: batch size to make experience, type=int, default=8
- --lora_rank:         low-rank adaptation matrices rank, type=int, default=0
- --kl_coef:           kl_coef using for computing reward, type=float, default=0.1
- --ptx_coef:          ptx_coef using for computing policy loss, type=float, default=0.9

## Inference example - After Stage3
We support different inference options, including int8 and int4 quantization.
For details, see [`inference/`](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat/inference).


## Attention
The examples are demos for the whole training process.You need to change the hyper-parameters to reach great performance.

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

### BLOOM
- [x] [BLOOM-560m](https://huggingface.co/bigscience/bloom-560m)
- [x] [BLOOM-1b1](https://huggingface.co/bigscience/bloom-1b1)
- [x] [BLOOM-3b](https://huggingface.co/bigscience/bloom-3b)
- [x] [BLOOM-7b](https://huggingface.co/bigscience/bloom-7b1)
- [ ] [BLOOM-175b](https://huggingface.co/bigscience/bloom)

### OPT
- [x] [OPT-125M](https://huggingface.co/facebook/opt-125m)
- [x] [OPT-350M](https://huggingface.co/facebook/opt-350m)
- [ ] [OPT-1.3B](https://huggingface.co/facebook/opt-1.3b)
- [ ] [OPT-2.7B](https://huggingface.co/facebook/opt-2.7b)
- [ ] [OPT-6.7B](https://huggingface.co/facebook/opt-6.7b)
- [ ] [OPT-13B](https://huggingface.co/facebook/opt-13b)
- [ ] [OPT-30B](https://huggingface.co/facebook/opt-30b)

### [LLaMA](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md)
- [x]  LLaMA-7B
- [x]  LLaMA-13B
- [ ]  LLaMA-33B
- [ ]  LLaMA-65B
