# RLHF - ColossalAI

Implementation of RLHF (Reinforcement Learning with Human Feedback) powered by ColossalAI. It supports distributed training and offloading, which can fit extremly large models.

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

strategy = ColossalAIStrategy()

with strategy.model_init_context():
  # init your model here
  actor = Actor()
  critic = Critic()

trainer = PPOTrainer(actor = actor, critic= critic, strategy, ...)

trainer.fit(dataset, ...)
```

For more details, see `examples/`.

We also support training reward model with true-world data. See `examples/train_reward_model.py`.

## Todo

- [x] implement PPO training
- [x] implement training reward model
- [x] support LoRA
- [ ] implement PPO-ptx fine-tuning
- [ ] integrate with Ray
- [ ] support more RL paradigms, like Implicit Language Q-Learning (ILQL)

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
