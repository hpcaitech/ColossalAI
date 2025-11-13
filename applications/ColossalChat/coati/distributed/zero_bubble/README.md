# Zero Bubble Distributed RL Framework for Language Model Fine-Tuning

This folder contains code for the Zero Bubble distributed RL framework. It currently supports **GRPO** and **DAPO**. See the [main README](../README.md) for general installation instructions and usage.

**Note:** This project is under active development — expect changes.

## 🛠 Installation

1. Follow the general installation guide in the [main README](../README.md).
2. Install [pygloo](https://github.com/ray-project/pygloo). Build pygloo for Ray from source following the instructions in its repository README.

## Design idea

We aim to reduce the *“bubble”* — the idle time that occurs between rollouts and training steps (illustrated in Fig. 1).

<div align="center">
  <p align="center">
    <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/all_sync.png" width=700/>
  </p>
</div>

**Fig. 1** - In an all-sync online RL framework, rollout workers wait for the trainer to finish training and synchronize weights, and the trainer waits for rollouts. This causes large GPU idle time.

<div align="center">
  <p align="center">
    <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/zero_bubble.png" width=700/>
  </p>
</div>

**Fig. 2** - Our Zero Bubble pipeline follows a producer–consumer pattern:

* A global **data buffer** temporarily stores rollouts produced by inference workers.
* A **weights distributor** buffers updated model weights and distributes them to inference workers.
* When the data buffer has enough data, the trainer continuously consumes from it and pushes updated weights to the weights distributor.
* After finishing a mini-batch, each inference worker checks the weights distributor and synchronizes to a newer weight version if available.

Under ideal conditions (inference workers produce data at the same rate the trainer consumes it), the pipeline eliminates idle time. We call it *zero bubble* because, with an unlimited data buffer, inference and training can run indefinitely without waiting. In practice, to avoid wasted compute and stale/off-policy data, we set a bounded buffer size so inference workers will briefly wait when the buffer is full.

## Usage

In addition to the general parameters (see the main README), the Zero Bubble pipeline introduces one additional parameter:

* **`data_actor_buffer_size_limit`** - Maximum number of rollout batches the data buffer may hold. Defaults to **twice** the trainer’s mini-batch size. Avoid setting this too large — a very large buffer increases off-policy training. For DAPO, since only effective prompts count, you may need to raise `data_actor_buffer_size_limit` depending on sample utility.

Example: RL training on 8 GPUs with Zero Bubble (zero2)

```bash
python rl_example_zero_bubble.py \
  --dataset /path/to/your/dataset.jsonl \
  --model /path/to/your/model \
  -t 4 -i 4 -b vllm -a DAPO \
  -imbs 8 -ibs 8 -tbs 8 -e 2 -rt boxed \
  -si 25 -s "Please reason step by step, and put your final answer within \\boxed{}." \
  -tMbs 2 -tmbs 2 -p Rebase_Experiments -zero 2 -mpt 512 -mnt 3584
```

## Performance

<div align="center">
  <p align="center">
    <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/zero_bubble_gpu_util.png" width=700/>
  </p>
</div>

**Fig. 3** - Performance of the Zero Bubble pipeline tested with an unlimited buffer size.
