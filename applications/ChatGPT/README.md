# RLHF - Colossal-AI

## Table of Contents

- [What is RLHF - Colossal-AI?](#intro)
- [How to Install?](#install)
- [The Plan](#the-plan)
- [How can you partcipate in open source?](#invitation-to-open-source-contribution)
---
## Intro
Implementation of RLHF (Reinforcement Learning with Human Feedback) powered by Colossal-AI. It supports distributed training and offloading, which can fit extremly large models. More details can be found in the [blog](https://www.hpc-ai.tech/blog/colossal-ai-chatgpt).

<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chatgpt/chatgpt.png" width=700/>
</p>

## Training process (step 3)
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chatgpt/experience.jpg" width=500/>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chatgpt/train.jpg" width=500/>
</p>


## Install
```shell
pip install .
```

## Usage

The main entrypoint is `Trainer`. We only support PPO trainer now. We support many training strategies:

- NaiveStrategy: simplest strategy. Train on single GPU.
- DDPStrategy: use `torch.nn.parallel.DistributedDataParallel`. Train on multi GPUs.
- ColossalAIStrategy: use Gemini and Zero of ColossalAI. It eliminates model duplication on each GPU and supports offload. It's very useful when training large models on multi GPUs.

Simplest usage:

```python
from chatgpt.trainer import PPOTrainer
from chatgpt.trainer.strategies import ColossalAIStrategy
from chatgpt.models.gpt import GPTActor, GPTCritic
from chatgpt.models.base import RewardModel
from copy import deepcopy
from colossalai.nn.optimizer import HybridAdam

strategy = ColossalAIStrategy()

with strategy.model_init_context():
  # init your model here
  # load pretrained gpt2
  actor = GPTActor(pretrained='gpt2')
  critic = GPTCritic()
  initial_model = deepcopy(actor).cuda()
  reward_model = RewardModel(deepcopy(critic.model), deepcopy(critic.value_head)).cuda()

actor_optim = HybridAdam(actor.parameters(), lr=5e-6)
critic_optim = HybridAdam(critic.parameters(), lr=5e-6)

# prepare models and optimizers
(actor, actor_optim), (critic, critic_optim), reward_model, initial_model = strategy.prepare(
        (actor, actor_optim), (critic, critic_optim), reward_model, initial_model)

# load saved model checkpoint after preparing
strategy.load_model(actor, 'actor_checkpoint.pt', strict=False)
# load saved optimizer checkpoint after preparing
strategy.load_optimizer(actor_optim, 'actor_optim_checkpoint.pt')

trainer = PPOTrainer(strategy,
                     actor,
                     critic,
                     reward_model,
                     initial_model,
                     actor_optim,
                     critic_optim,
                     ...)

trainer.fit(dataset, ...)

# save model checkpoint after fitting on only rank0
strategy.save_model(actor, 'actor_checkpoint.pt', only_rank0=True)
# save optimizer checkpoint on all ranks
strategy.save_optimizer(actor_optim, 'actor_optim_checkpoint.pt', only_rank0=False)
```

For more details, see `examples/`.

We also support training reward model with true-world data. See `examples/train_reward_model.py`.

## FAQ

### How to save/load checkpoint

To load pretrained model, you can simply use huggingface pretrained models:

```python
# load OPT-350m pretrained model
actor = OPTActor(pretrained='facebook/opt-350m')
```

To save model checkpoint:

```python
# save model checkpoint on only rank0
strategy.save_model(actor, 'actor_checkpoint.pt', only_rank0=True)
```

This function must be called after `strategy.prepare()`.

For DDP strategy, model weights are replicated on all ranks. And for ColossalAI strategy, model weights may be sharded, but all-gather will be applied before returning state dict. You can set `only_rank0=True` for both of them, which only saves checkpoint on rank0, to save disk space usage. The checkpoint is float32.

To save optimizer checkpoint:

```python
# save optimizer checkpoint on all ranks
strategy.save_optimizer(actor_optim, 'actor_optim_checkpoint.pt', only_rank0=False)
```

For DDP strategy, optimizer states are replicated on all ranks. You can set `only_rank0=True`. But for ColossalAI strategy, optimizer states are sharded over all ranks, and no all-gather will be applied. So for ColossalAI strategy, you can only set `only_rank0=False`. That is to say, each rank will save a cehckpoint. When loading, each rank should load the corresponding part.

Note that different stategy may have different shapes of optimizer checkpoint.

To load model checkpoint:

```python
# load saved model checkpoint after preparing
strategy.load_model(actor, 'actor_checkpoint.pt', strict=False)
```

To load optimizer checkpoint:

```python
# load saved optimizer checkpoint after preparing
strategy.load_optimizer(actor_optim, 'actor_optim_checkpoint.pt')
```

## The Plan

- [x] implement PPO fine-tuning
- [x] implement training reward model
- [x] support LoRA
- [x] support inference
- [ ] open source the reward model weight
- [ ] support llama from [facebook](https://github.com/facebookresearch/llama)
- [ ] support BoN(best of N sample)
- [ ] implement PPO-ptx fine-tuning
- [ ] integrate with Ray
- [ ] support more RL paradigms, like Implicit Language Q-Learning (ILQL),
- [ ] support chain of throught by [langchain](https://github.com/hwchase17/langchain)

### Real-time progress
You will find our progress in github project broad

[Open ChatGPT](https://github.com/orgs/hpcaitech/projects/17/views/1)

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
```
