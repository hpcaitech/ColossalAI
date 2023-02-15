# Examples

## Install requirements

```shell
pip install -r requirements.txt
```

## Train with dummy prompt data

This script supports 3 strategies:

- naive
- ddp
- colossalai

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
torchrun --standalone --nproc_per_node=2 train_dummy.py --strategy colossalai
```

## Train with real prompt data

We use [awesome-chatgpt-prompts](https://huggingface.co/datasets/fka/awesome-chatgpt-prompts) as example dataset. It is a small dataset with hundreds of prompts.

You should download `prompts.csv` first.

This script also supports 3 strategies.

```shell
# display cli help
python train_dummy.py -h
# run naive on 1 GPU
python train_prompts.py prompts.csv --strategy naive
# run DDP on 2 GPUs
torchrun --standalone --nproc_per_node=2 train_prompts.py prompts.csv --strategy ddp
# run ColossalAI on 2 GPUs
torchrun --standalone --nproc_per_node=2 train_prompts.py prompts.csv --strategy colossalai
```

## Train the reward model
We use [rm-static](https://huggingface.co/datasets/Dahoas/rm-static) as dataset to train our reward model. It is a dataset of chosen & rejected response of the same prompt.

You can download the dataset from huggingface automatically.

Use these code to train your reward model.

```shell
# Naive reward model training
python train_reward_model.py --pretrain <your model path>
# if to use LoRA
python train_reward_model.py --pretrain <your model path> --lora_rank 16
```

## Support Model

### GPT
- [ ]  GPT2-S (s)
- [ ]  GPT2-M (m)
- [ ]  GPT2-L (l)
- [ ]  GPT2-XL (xl)
- [ ]  GPT2-4B (4b)
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
- [ ] [BLOOM-3b](https://huggingface.co/bigscience/bloom-3b)
- [ ] [BLOOM-7b](https://huggingface.co/bigscience/bloomz-7b1)
- [ ] BLOOM-175b

### OPT
- [x] [OPT-125M](https://huggingface.co/facebook/opt-125m)
- [x] [OPT-350M](https://huggingface.co/facebook/opt-350m)
- [ ] [OPT-1.3B](https://huggingface.co/facebook/opt-1.3b)
- [ ] [OPT-2.7B](https://huggingface.co/facebook/opt-2.7b)
- [ ] [OPT-6.7B](https://huggingface.co/facebook/opt-6.7b)
- [ ] [OPT-13B](https://huggingface.co/facebook/opt-13b)
- [ ] [OPT-30B](https://huggingface.co/facebook/opt-30b)
