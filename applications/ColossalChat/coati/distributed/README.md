# Distributed RL Framework for Language Model Fine-Tuning

This repository implements a distributed Reinforcement Learning (RL) training framework designed to fine-tune large language models using algorithms such as **GRPO** and **DAPO**. It supports multi-node and multi-GPU setups, scalable rollout generation, and policy optimization using libraries like  VLLM.

---

## 🚀 Features

* **Distributed Training with Ray**: Scalable to multiple machines and GPUs.
* **Support for GRPO and DAPO**: Choose your preferred policy optimization algorithm.
* **Model Backends**: Support `vllm` as inference backends.
* **Rollout and Policy Decoupling**: Efficient generation and consumption of data through parallel inferencer-trainer architecture.
* **Evaluation Integration**: Easily plug in task-specific eval datasets.
* **Checkpoints and Logging**: Configurable intervals and directories.

---

## 🛠 Installation

### Prepare Develop Environment

Install Colossalai & ColossalChat
```bash
git clone https://github.com/hpcaitech/ColossalAI.git
git checkout grpo-latest-ascend
pip install -e .

cd ./applications/ColossalChat
pip install -e .
```
Install Fuyao Ray.
Please update CANN before install fuyao ray
```bash
# Install CANN
source /usr/local/Ascend/ascend-toolkit/set_env.sh
./Ascend-cann-kernels-910b_8.1.RC1.alpha001_linux-aarch64.run  --devel

# Clone Fuyao Ray. Fuyao Ray is not an open source project, it will be inherited in the ColossalRL images.
git clone https://gitee.com/openfuyao/ray.git
cd ray
git pull origin pull/5/head

# Install ray
pip install ray==2.43.0 --no-cache-dir

# Create soft-link from fuyao-ray to ray site-package
cd ..
ln -s ./ray/python/ray/ /usr/local/python3.10/lib/python3.10/site-packages/ray

# Install Fuyao Ray
cd ray
python python/ray/setup-dev.py
```

Prepare Model & dataset
```bash
huggingface-cli download --local-dir-use-symlinks False Qwen/Qwen2.5-7B --local-dir /models/Qwen/Qwen2.5-7B
```

### Set Distributed Config
Now, we need to set distributed config for multi-node.

First, we set host ip config.
For example. I need to configure a cluster of 4 nodes, then I do
```bash
vim /etc/hosts
```
Then write IP node map to /etc/hosts
```bash
10.0.0.3 npu-3
10.0.0.4 npu-4
10.0.0.5 npu-5
10.0.0.6 npu-6
```

Set Ascend Multi-Node Config

```bash
export ATB_LLM_HCCL_ENABLE=1
export ATB_LLM_COMM_BACKEND="hccl"
export HCCL_CONNECT_TIMEOUT=7200
export WORLD_SIZE=32
export HCCL_EXEC_TIMEOUT=7200
export HCCL_SOCKET_IFNAME=eno0
export RAY_COLLECTIVE_MEET_TIMEOUT_SECONDS=7200
```

## 🧠 Data Format

Each data sample in the training or evaluation `.jsonl` file should follow this format:

```json
{
  "messages": {
    "role": "user",
    "content": "Simplify $\\sqrt[3]{1+8} \\cdot \\sqrt[3]{1+\\sqrt[3]{8}}$. Let's think step by step and output the final answer within \\boxed{}."
  },
  "gt_answer": "3"
}
```

---

## ⚙️ Hyperparameters & Arguments

| Argument         | Description                             | Example           |
| ---------------- | --------------------------------------- | ----------------- |
| `--model`        | Model path or identifier                | `/path/to/model` |
| `--dataset`      | Path to training `.jsonl`               | `/path/to/train_data.jsonl`      |
| `--eval-dataset` | JSON of task\:eval\_dataset\_path pairs | `{'eval_1':'/path/to/eval_1.jsonl'}`            |
| `--project`      | Project name                            | `Project1`            |
| `--num-episodes` | Number of training episodes             | `1`               |

### Distributed Training

| Argument                      | Description                           | Example |
| ----------------------------- | ------------------------------------- | ------- |
| `--num-trainers`              | Number of trainer processes           | `4`     |
| `--num-inferencer`            | Number of inferencer processes        | `4`     |
| `--inference-batch-size`      | Prompts per inference step            | `8`    |
| `--inference-microbatch-size` | Per-GPU batch size for inference      | `8`     |
| `--train-batch-size`          | Prompts per trainer step per dp group | `8`    |
| `--train-minibatch-size`      | Mini-batch size before forward pass   | `8`     |
| `--train-microbatch-size`     | Per-GPU batch size for training       | `2`     |

### Sampling

| Argument              | Description           | Example        |
| --------------------- | --------------------- | -------------- |
| `--backend`           | Generation backend, choose from `vllm`     | `vllm` |
| `--temperature`       | Sampling temperature for generation  | `1.0`          |
| `--top-k`             | Top-K sampling parameter for generation        | `None`         |
| `--top-p`             | Top-P sampling parameter for generation        | `1.0`          |
| `--system-prompt`     | System prompt, default to the system prompt for `think_answer_tags` format         | `Please reason step by step, and put your final answer within \\boxed{}.`         |
| `--max-new-tokens`    | Max generation tokens | `3584`         |
| `--max-prompt-tokens` | Max prompt tokens     | `512`          |

### GRPO Specific

| Argument          | Description                  | Example             |
| ----------------- | ---------------------------- | ------------------- |
| `--algo`          | Algorithm (`GRPO` or `DAPO`), for more customization refer to [GRPO Settings](#️-grpo-settings) | `GRPO`              |
| `--learning-rate` | Learning rate                | `1e-6`              |
| `--kl-coeff`      | KL penalty coefficient       | `0.01`              |
| `--reward-type`   | Reward signal type (choose from 'think_answer_tags', 'boxed')          | `think_answer_tags` |
| `--eval-interval` | Evaluation interval in number of training steps (positive value to enable evaluation)         | `100`               |

### Logging and Checkpointing

| Argument             | Description               | Example      |
| -------------------- | ------------------------- | ------------ |
| `--save-interval`    | Training steps between checkpoints | `20`         |
| `--save-dir`         | Checkpoint directory      | `./model`    |
| `--eval-save-dir`    | Evaluation save path      | `./eval`     |
| `--rollout-save-dir` | Rollout logs directory    | `./rollouts` |

### Miscellaneous

| Argument           | Description                             | Example |
| ------------------ | --------------------------------------- | ------- |
| `--ray_dir`        | Custom Ray temp dir of a running Ray cluster (optional)                   | `None`  |
| `--master_address` | Master address of a running Ray cluster | `None`  |
| `--master_port`    | Master port for torch DDP                            | `29506` |

---

## ⚙️ GRPO Settings

In addition to the two default training settings we provided--- original `GRPO` and `DAPO`, users can customize their training by changing the following hyperparameters in `grpo_config` in `rl_example.py`.

| Argument Name                 | Description                      | Default                                                                                                                                                   |
| ----------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `filter_range`                | Filters out rollout group if the success rate within that group is out of this range.| `[0.01, 0.99]`                                  |
| `dynamic_batching`            | Enables dynamic batching as described in the [DAPO paper](https://arxiv.org/abs/2503.14476).                                                                      | `True`                                         |
| `clip_eps_low`                | epsilon_low in DAPO in equation in [DAPO paper](https://arxiv.org/abs/2503.14476)                                                   | `0.2`                                           |
| `clip_eps_high`               | epsilon_high in DAPO equation in [DAPO paper](https://arxiv.org/abs/2503.14476)                                                 | `0.28`                                           |
| `skip_threshold`              | If ratio is above this threshold, the sample is skipped to avoid instability.                                                   | `20.0`                                             |
| `loss_variation`              | Type of loss variation. Supports `"token_level"` for token-wise policy gradient loss and `sample_level` for original GRPO loss.                                         |  `"token_level"`                                        |
| `soft_over_length_punishment` | Whether to use soft overlength penalty in [DAPO paper](https://arxiv.org/abs/2503.14476) or not.                                                               | `True`                                             |
| `cache_length`                | `L_cache` parameter for soft overlength penalty in e.q. 13 in [DAPO paper](https://arxiv.org/abs/2503.14476)                                                                          | `min(1024, int(args.max_new_tokens / 4))`                 |
| `filter_truncated_response`    | Mask out truncated responses in loss calculation.                                       | `True`                                         |



## 🔄 Constraints and Notes

* `num_inferencer + num_trainer == NUM_GPUs`
* `num_inferencer % num_trainer == 0`
* `(num_inferencer * inference_batch_size) % (num_trainer * train_batch_size) == 0`
* `train_batch_size >= train_minibatch_size >= train_microbatch_size`
* `inference_batch_size >= inference_microbatch_size`
* Set microbatch sizes based on **VRAM capacity**
* To use tensor parallelism on inferencer
  * set backend to `vllm`
  * change `tensor_parallel_size` in `inference_model_config` in rl_example.py
  * set `num_inferencer = NUM_INFERENCE_GPUs / tensor_parallel_size`
* To set tensor parallelism / pipeline parallelism / zero stage
  * change corresponding settings in `plugin_config` in rl_example.py
* Ensure rollout generation rate matches trainer consumption:

  ```
  num_inferencer * inference_batch_size % (
    num_trainer * train_batch_size /
    train_pipeline_parallelism_size /
    train_tensor_parallelism_size
  ) == 0
  ```
* Model weights sync every:

  ```
  (num_inferencer * inference_batch_size) /
  (num_trainer * train_batch_size /
    train_pipeline_parallelism_size /
    train_tensor_parallelism_size)
  ```

---

## 🧪 Example: single machine 8-GPU Zero2 Strategy

```bash
python rl_example.py \
  --dataset /path/to/train_data.jsonl \
  --model /path/to/Qwen2.5-3B/ \
  -t 4 -i 4 \
  -b vllm \
  -ibs 2 -tbs 4 -tMbs 1 -tmbs 4 -imbs 1 \
  -rt boxed \
  -g 4 \
  -ibs 1 \
  -tbs 2 \
  -tMbs 1 \
  -tmbs 2 \
  -imbs 1 \
  -s "Please reason step by step, and put your final answer within \\boxed{}." \
  -tMbs 8 \
  -p GRPO-Train-Align-Debug \
```

## 🧪 Example: multi-machine TP+PP Strategy

### Create ray cluster on multi-machine
For example, now we have 4 nodes and their IPs are 10.0.0.3, 10.0.0.4, 10.0.0.5, 10.0.0.6.
We use 10.0.0.3 as master node. First we start a ray cluster on 10.0.0.3:
```bash
ray start --head --node-ip-address=10.0.0.3
```

Then, for each slave node (10.0.0.4/10.0.0.5/10.0.0.6), we add to the ray cluser by following code:
```bash
ray start --address='10.0.0.3:6379'
```

Modify plugin_config in ./applications/ColossalChat/rl_example.py
```python
plugin_config={
  "tp_size": 4,
  "pp_size": 2,
  "microbatch_size": max(
    1, args.train_microbatch_size // 2
  ),  # microbatch size should be set to train_microbatch_size // pp_size
  "zero_stage": 1,
  "max_norm": 1.0,
  },  # for pp, tp
```

```bash
# Hint1: replace /models/Qwen/Qwen2.5-7B to your model path
#        replace /datasets/train-alignment.jsonl to your dataset path
python rl_example.py
  -m /path/to/Qwen2.5-Math-7B/ \
  -d /path/to/train_data.jsonl \
  --master_address '10.0.0.3'
  -t 16 \
  -i 16 \
  -p GRPO-Train-Align-Debug \
  -g 2 \
  -ibs 1 \
  -tbs 2 \
  -tMbs 1 \
  -tmbs 2 \
  -imbs 1 \
  -b vllm \
  -e 2 \
  -rt boxed \
  -s "Please reason step by step, and put your final answer within \\boxed{}."
```

## Acknowledgement

---
