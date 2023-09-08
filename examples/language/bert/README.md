## Overview

This directory includes two parts: Using the Booster API finetune Huggingface Bert and AlBert models and benchmarking Bert and AlBert models with different Booster Plugin.

## Finetune
```
bash test_ci.sh
```

### Bert-Finetune Results

| Plugin         | Accuracy | F1-score | GPU number |
| -------------- | -------- | -------- | -------- |
| torch_ddp      | 84.4%    | 88.6%    |    2     |
| torch_ddp_fp16 | 84.7%    | 88.8%    |    2     |
| gemini         | 84.0%    | 88.4%    |    2     |
| hybrid_parallel | 84.5%    | 88.6%    |    4     |


## Benchmark
```
bash benchmark.sh
```

Now include these metrics in benchmark: CUDA mem occupy, throughput and the number of model parameters. If you have custom metrics, you can add them to benchmark_util.

### Results

#### Bert

|       | max cuda mem | throughput(sample/s) | params |
| :-----| -----------: | :--------: | :----: |
| ddp | 21.44 GB | 3.0 | 82M |
| ddp_fp16 | 16.26 GB | 11.3 | 82M |
| gemini | 11.0 GB | 12.9 | 82M |
| low_level_zero | 11.29 G | 14.7 | 82M |

#### AlBert
|       | max cuda mem | throughput(sample/s) | params |
| :-----| -----------: | :--------: | :----: |
| ddp | OOM |  | |
| ddp_fp16 | OOM |  | |
| gemini | 69.39 G | 1.3 | 208M |
| low_level_zero | 56.89 G | 1.4 | 208M |
