# Finetune BERT on GLUE

## 🚀 Quick Start

This example provides a training script, which provides an example of finetuning BERT on GLUE dataset.

- Training Arguments
  - `-t`, `--task`: GLUE task to run. Defaults to `mrpc`.
  - `-p`, `--plugin`: Plugin to use. Choices: `torch_ddp`, `torch_ddp_fp16`, `gemini`, `low_level_zero`. Defaults to `torch_ddp`.
  - `--target_f1`: Target f1 score. Raise exception if not reached. Defaults to `None`.


### Install requirements

```bash
pip install -r requirements.txt
```

### Train

```bash
# train with torch DDP with fp32
colossalai run --nproc_per_node 4 finetune.py

# train with torch DDP with mixed precision training
colossalai run --nproc_per_node 4 finetune.py -p torch_ddp_fp16

# train with gemini
colossalai run --nproc_per_node 4 finetune.py -p gemini

# train with low level zero
colossalai run --nproc_per_node 4 finetune.py -p low_level_zero
```

Expected F1-score will be:

| Model             | Single-GPU Baseline FP32 | Booster DDP with FP32 | Booster DDP with FP16 | Booster Gemini | Booster Low Level Zero |
| ----------------- | ------------------------ | --------------------- | --------------------- |--------------- | ---------------------- |
| bert-base-uncased | 0.86                     | 0.88                  | 0.87                  | 0.88           | 0.89                   |
