# A script of pretraining LLaMA

## Dataset

Different from the original LLaMA, we use [RedPajama](https://www.together.xyz/blog/redpajama) dataset, which is a reproduction of the LLaMA training dataset containing over 1.2 trillion tokens. The full dataset is ~5TB unzipped on disk and ~3TB to download compressed.

A smaller, more consumable random sample can be downloaded through [Hugging Face](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T). If you just want to try out the pretraining script, you can use a 1B-token sample subset of RedPajama, which is available at [Hugging Face](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample).

RedPajama-Data-1T consists of seven data slices:

|               | RedPajama    | LLaMA         |
|---------------|--------------|---------------|
| CommonCrawl   | 878 billion  | 852 billion   |
| C4            | 175 billion  | 190 billion   |
| Github        | 59 billion   | 100 billion   |
| Books         | 26 billion   | 25 billion    |
| ArXiv         | 28 billion   | 33 billion    |
| Wikipedia     | 24 billion   | 25 billion    |
| StackExchange | 20 billion   | 27 billion    |
| Total         | 1.2 trillion | 1.25 trillion |

## Training

We follow the hyperparameter settings from the original LLaMA paper. We use AdamW with $beta1=0.9$ and $beta2=0.95$. We use a cosine learning rate schedule, such that the final learning rate is equal to 10% of the maximal learning rate. We use a weight decay of 0.1 and gradient clipping of 1.0. We use 2,000 warmup steps.

| params | learning rate | batch size |
|--------|---------------|------------|
| 6.7B   | 3.0e-4        | 4M         |
| 13.0B  | 3.0e-4        | 4M         |
| 32.5B  | 1.5e-4        | 4M         |
| 65.2B  | 1.5e-4        | 4M         |

## Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

The dataset can be automatically downloaded by using `huggingface/datasets`. You can specify the dataset path by `-d` or `--dataset`. The default dataset is `togethercomputer/RedPajama-Data-1T-Sample`.

### 3. Commad line arguments

- Model configuration: `-c`, `--config`. `7b`, `13b`, `30b` and `65b` are supported.
- Booster plugin: `-p`, `--plugin`. `gemini`, `gemini_cpu`, `zero2` and `zero2_cpu` are supported. For more details, please refer to [Booster plugins](https://colossalai.org/docs/basics/booster_plugins).
- Dataset path: `-d`, `--dataset`. The default dataset is `togethercomputer/RedPajama-Data-1T-Sample`. It support any dataset from `datasets` with the same data format as RedPajama.
- Number of epochs: `-e`, `--num_epochs`. The default value is 1.
- Local batch size: `-b`, `--batch_size`. Batch size per GPU. The default value is 2.
- Learning rate: `--lr`. The default value is 3e-4.
- Weight decay: `-w`, `--weight_decay`. The default value is 0.1.
- Warmup steps: `-s`, `--warmup_steps`. The default value is 2000.
- Gradient checkpointing: `-g`, `--gradient_checkpoint`. The default value is `False`.
- Max length: `-l`, `--max_length`. The default value is 2048.
- Mixed precision: `-x`, `--mixed_precision`. The default value is "fp16". "fp16" and "bf16" are supported.
- Save interval: `-i`, `--save_interval`. The interval (steps) of saving checkpoints. The default value is 1000.
- Checkpoint directory: `-o`, `--save_dir`. The directoty path to save checkpoints. The default value is `checkpoint`.
- Checkpoint to load: `-f`, `--load`. The checkpoint path to load. The default value is `None`.
- Gradient clipping: `--gradient_clipping`. The default value is 1.0.
- Tensorboard log directory: `-t`, `--tensorboard_dir`. The directory path to save tensorboard logs. The default value is `tb_logs`.
- Flash attention: `-a`, `--flash_attention`. If you want to use flash attention, you must install [xformers](https://github.com/facebookresearch/xformers) first. The default value is `False`.
