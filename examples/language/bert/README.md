## Overview

This directory includes two parts: Using the Booster API finetune Huggingface Bert and AlBert models and benchmarking Bert and AlBert models with different Booster Plugin.

## Finetune
```
bash test_ci.sh
```

## Benchmark
```
bash benchmark.sh
```

Now include these metrics in benchmark: CUDA mem occupy, throughput and the number of model parameters. If you have custom metrics, you can add them to benchmark_util.

## Results

### Bert

|       | max cuda mem | throughput(sample/s) | params |
| :-----| -----------: | :--------: | :----: |
| ddp | 21.44 GB | 3.0 | 87M |
| ddp_fp16 | 16.26 GB | 11.6 | 87M |
| gemini | 11.3 GB | 15.2 | 87M |
| low_level_zero | 11.3 GB | 14.3 | 87M |

### AlBert
|       | max cuda mem | throughput(sample/s) | params |
| :-----| -----------: | :--------: | :----: |
| ddp | OOM |  | |
| ddp_fp16 | OOM |  | |
| gemini | 69.39 G | 1.4 | 219M |
| low_level_zero | 56.80 G | 1.4 | 219M |