# Examples

## Table of Contents

- [Examples](#examples)
  - [Table of Contents](#table-of-contents)
  - [Install Requirements](#install-requirements)
  - [Supervised Datasets Collection](#supervised-datasets-collection)
    - [Conversation Dataset Generation](#conversation-dataset-generation)
  - [Task I: Supervised Instruction Tuning](#task-i-supervised-instructs-tuning)
  - [Task II: Reinforcement Learning from Human Feedback](#task-ii-reinforcement-learning-from-human-feedback)
    - [Stage1 - Supervised instructs tuning](#stage1---supervised-instructs-tuning)
      - [Arg List](#arg-list)
    - [Stage2 - Training reward model](#stage2---training-reward-model)
      - [Features and tricks in RM training](#features-and-tricks-in-rm-training)
      - [Experiment result](#experiment-result)
      - [Arg List](#arg-list-1)
    - [Stage3 - Training model using prompts with RL](#stage3---training-model-using-prompts-with-rl)
      - [Arg List](#arg-list-2)
  - [Inference example - After Stage3](#inference-example---after-stage3)
  - [Attention](#attention)
    - [data](#data)
  - [Support Model](#support-model)
    - [GPT](#gpt)
    - [BLOOM](#bloom)
    - [OPT](#opt)
    - [LLaMA](#llama)
  - [Add your own models](#add-your-own-models)
    - [Actor model](#actor-model)
    - [Reward model](#reward-model)
    - [Critic model](#critic-model)

---

## Install requirements

```shell
pip install -r requirements.txt
```

## Get Start with ColossalRun

You can use colossalai run to launch multi-nodes training:
```
colossalai run --nproc_per_node YOUR_GPU_PER_NODE --hostfile YOUR_HOST_FILE \
train.py --OTHER_CONFIGURATIONS
```
Here is a sample hostfile:

```
hostname1
hostname2
hostname3
hostname4
```

Make sure master node can access all nodes (including itself) by ssh without password. Here are some other arguments.

- nnodes: number of nodes used in the training
- nproc-per-node: specifies the number of processes to be launched per node
- rdzv-endpoint: address of the host node


## Supervised datasets collection

We collected 104K bilingual datasets of Chinese and English, and you can find the datasets in this repo
[InstructionWild](https://github.com/XueFuzhao/InstructionWild) and in this [file](https://github.com/XueFuzhao/InstructionWild/blob/main/data/README.md).

Here is how we collected the data

<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/data-collect.png" width=500/>
</p>

### Conversation dataset generation

In order to further improve the model's ability to handle multi-turn conversations, we need to include samples with multi-turn conversations in the dataset. However, the samples in InstructWild and Alpaca datasets currently consist of only single-turn conversations, and their dataset organization is not suitable for storing multi-turn conversations. Additionally, after converting the aforementioned datasets, we also need to include multi-turn conversation datasets like ShareGPT, and we should transform them into the training format supported by ColossalChat.

A sample of conversation dataset should have the following fields:

- `type` (str, optional): The type of the data sample.
- `language` (str, optional): The language of the data sample.
- `dataset` (str, optional): The dataset the data sample originates from.
- `conversations` (str, compulsory): Conversation content of the data sample.
- `id` (int, optional): The ID of the data sample.

A simple example:

```json
{
  "type": "instruction",
  "language": "English",
  "dataset": "Alpaca",
  "conversations": [
    {
      "from": "human",
      "value": "Give three tips for staying healthy."
    },
    {
      "from": "gpt",
      "value": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
    }
  ],
  "id": 1
}
```

> **NOTE:** Only key `conversations` is compulsary for training and other keys serve as metadata. The length of `conversations` varies.

You can run the `examples/generate_conversation_dataset.py` to generate a conversation dataset supported by ColossalChat.

You can use the following cmd to generate conversation dataset.

```bash
python generate_conversation_dataset.py \
    --dataset "All"
    --save_path "/path/to/dataset"
```

## Task I: Supervised Instructs Tuning

In the task of supervised instructs fine-tuning, we will uses the datasets mentioned earlier to fine-tune the model.
[[Stage1 tutorial video]](https://www.youtube.com/watch?v=-qFBZFmOJfg)

You can run the `examples/train_sft.sh` to start a supervised instructs fine-tuning.

You can also use the following cmd to start a supervised instructs fine-tuning with your own settings.

```bash
colossalai run --nproc_per_node 1 --hostfile ./hostfile train_sft.py \
    --pretrain "/path/to/LLaMa-7B/" \
    --model 'llama' \
    --strategy colossalai_zero2 \
    --save_path  /path/to/Coati-7B \
    --dataset /path/to/data.json \
    --batch_size 4 \
    --accumulation_steps 8 \
    --lr 2e-5 \
    --max_datasets_size 512 \
    --max_epochs 1 \
    --grad_checkpoint \
    --use_wandb
```

**Note**: the supervised dataset follows the following format,

```json
[
    {
        "instruction": "Provide a list of the top 10 most popular mobile games in Asia",
        "input": "",
        "output": "The top 10 most popular mobile games in Asia are:\n1) PUBG Mobile\n2) Pokemon Go\n3) Candy Crush Saga\n4) Free Fire\n5) Clash of Clans\n6) Mario Kart Tour\n7) Arena of Valor\n8) Fantasy Westward Journey\n9) Subway Surfers\n10) ARK Survival Evolved",
        "id": 0
    },
    ...
]
```

### Arg List
- `--strategy`: the strategy using for training, choices=['ddp', 'colossalai_gemini', 'colossalai_zero2'], default='colossalai_zero2'
- `--model`: model type, choices=['gpt2', 'bloom', 'opt', 'llama'], default='bloom'
- `--pretrain`: pretrain model, type=str, default=None
- `--max_datasets_size`: the max size of dataset, type=int, default=None
- `--save_path`: path to save the model, type=str, default='output'
- `--need_optim_ckpt`: whether to save optim ckpt, type=bool, default=False
- `--max_epochs`: max epochs for training, type=int, default=3
- `--batch_size`: batch size while training, type=int, default=4
- `--lora_rank`: low-rank adaptation matrices rank, type=int, default=0
- `--grad_checkpoint`: enable gradient checkpointing, type=bool, default=False
- `use_wandb`: whether to use [wandb](https://wandb.ai/site)

## Task II: Reinforcement Learning from Human Feedback
### Stage1 - Supervised Instructs Tuning

The first stage of RLHF is supervised instructs fine-tuning (SFT). This stage is basically the same as the first task, which uses the same datasets but with different prompt format.

You can run the `examples/train_rlhf_sft.sh` to start a supervised instructs fine-tuning.

You can also use the following cmd to start a supervised instructs fine-tuning with your own settings.

```bash
colossalai run --nproc_per_node 1 --hostfile ./hostfile train_rlhf_sft.py \
    --pretrain "gpt2" \
    --model 'gpt2' \
    --strategy colossalai_zero2 \
    --save_path 'path to a directory where you want to stre the weights of the model' \
    --dataset 'path to your dataset, which should be a json file' \
    --batch_size 4 \
    --accumulation_steps 8 \
    --lr 2e-5 \
    --max_datasets_size 60000 \
    --max_epochs 1 \
    --use_wandb
```

**Note**: the supervised dataset follows the same format as in Task I.

### Arg List

The same as in Task I.


### Stage2 - Training reward model

We train a reward model in stage 2, which obtains corresponding scores by manually ranking different outputs for the same prompt and supervises the training of the reward model.
[[Stage2 tutorial video]](https://www.youtube.com/watch?v=gMx2CApKhuo)

You can run the `examples/train_rm.sh` to start a reward model training.

You can also use the following cmd to start training a reward model.

```bash
colossalai run --nproc_per_node 1 --hostfile ./hostfile train_reward_model.py \
    --pretrain "/path/to/LLaMa-7B/" \
    --model 'llama' \
    --strategy colossalai_zero2 \
    --loss_fn 'log_exp'\
    --save_path 'rmstatic.pt' \
```

### Features and tricks in RM training

- We support [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)and[rm-static](https://huggingface.co/datasets/Dahoas/rm-static) datasets.
- We support 2 kinds of loss function named `log_sig`(used by OpenAI) and `log_exp`(used by Anthropic).
- We change the loss to `valid_acc` and `pair_dist` to monitor progress during training.
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

- `--strategy`: the strategy using for training, choices=['ddp', 'colossalai_gemini', 'colossalai_zero2'], default='colossalai_zero2'
- `--model`: model type, choices=['gpt2', 'bloom', 'opt', 'llama'], default='bloom'
- `--pretrain`: pretrain model, type=str, default=None
- `--model_path`: the path of rm model(if continue to train), type=str, default=None
- `--save_path`: path to save the model, type=str, default='output'
- `--need_optim_ckpt`: whether to save optim ckpt, type=bool, default=False
- `--max_epochs`: max epochs for training, type=int, default=3
- `--dataset`: dataset name, type=str, choices=['Anthropic/hh-rlhf', 'Dahoas/rm-static']
- `--subset`: subset of the dataset, type=str, default=None
- `--batch_size`: batch size while training, type=int, default=4
- `--lora_rank`: low-rank adaptation matrices rank, type=int, default=0
- `--loss_func`: which kind of loss function, choices=['log_sig', 'log_exp']
- `--max_len`: max sentence length for generation, type=int, default=512
- `--use_wandb`: whether to use wandb


### Note on Reward Model Training

Before you move on the next stage, please check the following list to ensure that your reward model is stable and robust. You can check the reward chart and the accuracy chart on wandb.
- The mean reward for chosen data is much higher than those for rejected data
- The accuracy is larger than 0.5 by a significant margin (usually should be greater than 0.6)
- Optional：check the reward is positive for chosen data vice versa

Your training reward curves should look similar to the following charts.
<p align="center">
<img width="1000" alt="image" src="https://raw.githubusercontent.com/YeAnbang/imagehostingrepo/main/mean_reward_chart.png">
</p>

## Stage3 - Training model using prompts with RL

Stage3 uses reinforcement learning algorithm, which is the most complex part of the training process, as shown below:

<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/stage-3.jpeg" width=800/>
</p>

You can run the `examples/train_prompts.sh` to start PPO training.

You can also use the cmd following to start PPO training.
[[Stage3 tutorial video]](https://www.youtube.com/watch?v=Z8wwSHxPL9g)


PPO Training Script
```bash
colossalai run --nproc_per_node 1 --hostfile ./hostfile train_prompts.py \
    --pretrain_dataset 'path to sft dataset used in stage 1'  \
    --prompt_dataset 'dataset that contains prompt (queries) for PPO training' \
    --strategy colossalai_zero2 \
    --num_episodes 8000 --num_collect_steps 1 --num_update_steps 1 \
    --experience_batch_size 32 \
    --train_batch_size 32 \
    --save_path 'path to save the trained model' \
    --ptx_coef 0.0 \
    --rm_model 'gpt2' \
    --rm_pretrain 'gpt2' \
    --rm_path 'path to reward model trained in stage 2' \
    --reward_model_tokenizer 'gpt2' \
    --pretrain '/home/lcyab/data/Anthropic_rlhf/actor/pretrain_v3' \
    --use_wandb

```
Prompt dataset: the instruction dataset mentioned in the above figure which includes the instructions, e.g. you can use the [script](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat/examples/generate_prompt_dataset.py) which samples `instinwild_en.json` or `instinwild_ch.json` in [InstructionWild](https://github.com/XueFuzhao/InstructionWild/tree/main/data#instructwild-data) to generate the prompt dataset.
Pretrain dataset: the pretrain dataset including the instruction and corresponding response, e.g. you can use the [InstructWild Data](https://github.com/XueFuzhao/InstructionWild/tree/main/data) in stage 1 supervised instructs tuning.

**Note**: the required datasets follow the following format,

- `pretrain dataset`

  ```json
  [
      {
          "instruction": "Provide a list of the top 10 most popular mobile games in Asia",
          "input": "",
          "output": "The top 10 most popular mobile games in Asia are:\n1) PUBG Mobile\n2) Pokemon Go\n3) Candy Crush Saga\n4) Free Fire\n5) Clash of Clans\n6) Mario Kart Tour\n7) Arena of Valor\n8) Fantasy Westward Journey\n9) Subway Surfers\n10) ARK Survival Evolved",
          "id": 0
      },
      ...
  ]
  ```

- `prompt dataset`

  ```json
  [
      {
          "instruction": "Edit this paragraph to make it more concise: \"Yesterday, I went to the store and bought some things. Then, I came home and put them away. After that, I went for a walk and met some friends.\"",
          "id": 0
      },
      {
          "instruction": "Write a descriptive paragraph about a memorable vacation you went on",
          "id": 1
      },
      ...
  ]
  ```
### Sample Training Results Using Default Script
#### Reward
<p align="center">
<img width="700" alt="image" src="https://raw.githubusercontent.com/YeAnbang/imagehostingrepo/main/reward.png">
</p>

#### Approximate KL Divergence
<p align="center">
<img width="700" alt="image" src="https://raw.githubusercontent.com/YeAnbang/imagehostingrepo/main/KL.png">
</p>

### Note on PPO Training
#### Q1: My reward is nagtive
Answer: Check your reward model trained in stage 1. If the reward model only generate negative reward, we actually will expect a negative reward. However, even though the reward is negative, the reward should go up.

#### Q2: My actor loss is negative
Answer: This is normal for actor loss as PPO doesn't restrict the actor loss to be positive.

#### Q3: My reward doesn't go up (decreases)
Answer: The causes to this problem are two-fold. Check your reward model, make sure that it gives positive and strong reward for good cases and negative, strong reward for bad responses. You should also try different hyperparameter settings.

#### Q4: Generation is garbage
Answer: Yes, this happens and is well documented by other implementations. After training for too many episodes, the actor gradually deviate from its original state, which may leads to decrease in language modeling capabilities. A way to fix this is to add suppervised loss during PPO. Set ptx_coef to a none-zero value (between 0 and 1), which balances PPO loss and sft loss.

### Arg List

- `--strategy`: the strategy using for training, choices=['ddp', 'colossalai_gemini', 'colossalai_zero2'], default='colossalai_zero2'
- `--model`: model type of actor, choices=['gpt2', 'bloom', 'opt', 'llama'], default='bloom'
- `--pretrain`: pretrain model, type=str, default=None
- `--rm_model`: reward model type, type=str, choices=['gpt2', 'bloom', 'opt', 'llama'], default=None
- `--rm_pretrain`: pretrain model for reward model, type=str, default=None
- `--rm_path`: the path of rm model, type=str, default=None
- `--save_path`: path to save the model, type=str, default='output'
- `--prompt_dataset`: path of the prompt dataset, type=str, default=None
- `--pretrain_dataset`: path of the ptx dataset, type=str, default=None
- `--need_optim_ckpt`: whether to save optim ckpt, type=bool, default=False
- `--num_episodes`: num of episodes for training, type=int, default=10
- `--num_update_steps`: number of steps to update policy per episode, type=int
- `--num_collect_steps`: number of steps to collect experience per episode, type=int
- `--train_batch_size`: batch size while training, type=int, default=8
- `--ptx_batch_size`: batch size to compute ptx loss, type=int, default=1
- `--experience_batch_size`: batch size to make experience, type=int, default=8
- `--lora_rank`: low-rank adaptation matrices rank, type=int, default=0
- `--kl_coef`: kl_coef using for computing reward, type=float, default=0.1
- `--ptx_coef`: ptx_coef using for computing policy loss, type=float, default=0.9
- `--use_wandb`

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

- [x] GPT2-S (s)
- [x] GPT2-M (m)
- [x] GPT2-L (l)
- [x] GPT2-XL (xl)
- [x] GPT2-4B (4b)
- [ ] GPT2-6B (6b)

### BLOOM

- [x] [BLOOM-560m](https://huggingface.co/bigscience/bloom-560m)
- [x] [BLOOM-1b1](https://huggingface.co/bigscience/bloom-1b1)
- [x] [BLOOM-3b](https://huggingface.co/bigscience/bloom-3b)
- [x] [BLOOM-7b](https://huggingface.co/bigscience/bloom-7b1)
- [ ] [BLOOM-175b](https://huggingface.co/bigscience/bloom)

### OPT

- [x] [OPT-125M](https://huggingface.co/facebook/opt-125m)
- [x] [OPT-350M](https://huggingface.co/facebook/opt-350m)
- [x] [OPT-1.3B](https://huggingface.co/facebook/opt-1.3b)
- [x] [OPT-2.7B](https://huggingface.co/facebook/opt-2.7b)
- [x] [OPT-6.7B](https://huggingface.co/facebook/opt-6.7b)
- [ ] [OPT-13B](https://huggingface.co/facebook/opt-13b)
- [ ] [OPT-30B](https://huggingface.co/facebook/opt-30b)

### [LLaMA](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md)

- [x] LLaMA-7B
- [x] LLaMA-13B
- [ ] LLaMA-33B
- [ ] LLaMA-65B

## Add your own models

If you want to support your own model in Coati, please refer the pull request for RoBERTa support as an example --[[chatgpt] add pre-trained model RoBERTa for RLHF stage 2 & 3](https://github.com/hpcaitech/ColossalAI/pull/3223), and submit a PR to us.

You should complete the implementation of four model classes, including Reward model, Critic model, LM model, Actor model

here are some example code for a NewModel named `Coati`.
if it is supported in huggingface [transformers](https://github.com/huggingface/transformers), you can load it by `from_pretrained`, o
r you can build your own model by yourself.

### Actor model

```python
from ..base import Actor
from transformers.models.coati import CoatiModel

class CoatiActor(Actor):
    def __init__(self,
                 pretrained: Optional[str] = None,
                 checkpoint: bool = False,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none') -> None:
        if pretrained is not None:
            model = CoatiModel.from_pretrained(pretrained)
        else:
            model = build_model() # load your own model if it is not support in transformers

        super().__init__(model, lora_rank, lora_train_bias)
```

### Reward model

```python
from ..base import RewardModel
from transformers.models.coati import CoatiModel

class CoatiRM(RewardModel):

    def __init__(self,
                 pretrained: Optional[str] = None,
                 checkpoint: bool = False,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none') -> None:
        if pretrained is not None:
            model = CoatiModel.from_pretrained(pretrained)
        else:
            model = build_model() # load your own model if it is not support in transformers

        value_head = nn.Linear(model.config.n_embd, 1)
        value_head.weight.data.normal_(mean=0.0, std=1 / (model.config.n_embd + 1))
        super().__init__(model, value_head, lora_rank, lora_train_bias)
```

### Critic model

```python
from ..base import Critic
from transformers.models.coati import CoatiModel

class CoatiCritic(Critic):
    def __init__(self,
                 pretrained: Optional[str] = None,
                 checkpoint: bool = False,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none') -> None:
        if pretrained is not None:
            model = CoatiModel.from_pretrained(pretrained)
        else:
            model = build_model() # load your own model if it is not support in transformers

        value_head = nn.Linear(model.config.n_embd, 1)
        value_head.weight.data.normal_(mean=0.0, std=1 / (model.config.n_embd + 1))
        super().__init__(model, value_head, lora_rank, lora_train_bias)
```
