# Handson 3: Auto-Parallelism with ResNet

## Prepare Dataset

We use CIFAR10 dataset in this example. You should invoke the `donwload_cifar10.py` in the tutorial root directory or directly run the `auto_parallel_with_resnet.py`.
The dataset will be downloaded to `colossalai/examples/tutorials/data` by default.
If you wish to use customized directory for the dataset. You can set the environment variable `DATA` via the following command.

```bash
export DATA=/path/to/data
```


## Run on 2*2 device mesh

```bash
colossalai run --nproc_per_node 4 auto_parallel_with_resnet.py
```

## Auto Checkpoint Benchmarking

We prepare three demos for you to test the performance of auto checkpoint, the test `demo_resnet50.py` and `demo_gpt2_medium.py` will show you the ability of solver to search checkpoint strategy that could fit in the given budget.

The usage of the above two test
```bash
python demo_resnet50.py --help
usage: ResNet50 Auto Activation Benchmark [-h] [--batch_size BATCH_SIZE] [--num_steps NUM_STEPS] [--sample_points SAMPLE_POINTS] [--free_memory FREE_MEMORY]
                                          [--start_factor START_FACTOR]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        batch size for benchmark, default 128
  --num_steps NUM_STEPS
                        number of test steps for benchmark, default 5
  --sample_points SAMPLE_POINTS
                        number of sample points for benchmark from start memory budget to maximum memory budget (free_memory), default 15
  --free_memory FREE_MEMORY
                        maximum memory budget in MB for benchmark, default 11000 MB
  --start_factor START_FACTOR
                        start memory budget factor for benchmark, the start memory budget will be free_memory / start_factor, default 4

# run with default settings
python demo_resnet50.py

python demo_gpt2_medium.py --help
usage: GPT2 medium Auto Activation Benchmark [-h] [--batch_size BATCH_SIZE] [--num_steps NUM_STEPS] [--sample_points SAMPLE_POINTS] [--free_memory FREE_MEMORY]
                                             [--start_factor START_FACTOR]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        batch size for benchmark, default 8
  --num_steps NUM_STEPS
                        number of test steps for benchmark, default 5
  --sample_points SAMPLE_POINTS
                        number of sample points for benchmark from start memory budget to maximum memory budget (free_memory), default 15
  --free_memory FREE_MEMORY
                        maximum memory budget in MB for benchmark, default 56000 MB
  --start_factor START_FACTOR
                        start memory budget factor for benchmark, the start memory budget will be free_memory / start_factor, default 10

# run with default settings
python demo_gpt2_medium.py
```

There are some results for your reference

### ResNet 50
![](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/tutorial/resnet50_benchmark.png)

### GPT2 Medium
![](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/tutorial/gpt2_benchmark.png)

We also prepare the demo `demo_resnet152.py` to manifest the benefit of auto activation with large batch, the usage is listed as follows
```bash
python demo_resnet152.py --help
usage: ResNet152 Auto Activation Through Put Benchmark [-h] [--num_steps NUM_STEPS]

optional arguments:
  -h, --help            show this help message and exit
  --num_steps NUM_STEPS
                        number of test steps for benchmark, default 5

# run with default settings
python demo_resnet152.py
```

here are some results on our end for your reference
```bash
===============test summary================
batch_size: 512, peak memory: 73314.392 MB, through put: 254.286 images/s
batch_size: 1024, peak memory: 73316.216 MB, through put: 397.608 images/s
batch_size: 2048, peak memory: 72927.837 MB, through put: 277.429 images/s
```

The above tests will output the test summary and a plot of the benchmarking results.
