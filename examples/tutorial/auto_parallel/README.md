# Auto-Parallelism with ResNet

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

We prepare two bechmarks for you to test the performance of auto checkpoint

The first test `auto_ckpt_solver_test.py` will show you the ability of solver to search checkpoint strategy that could fit in the given budget (test on GPT2 Medium and ResNet 50). It will output the benchmark summary and data visualization of peak memory vs. budget memory and relative step time vs. peak memory.

The second test `auto_ckpt_batchsize_test.py` will show you the advantage of fitting larger batchsize training into limited GPU memory with the help of our activation checkpoint solver (test on ResNet152). It will output the benchmark summary.

The usage of the above two test
```bash
# run auto_ckpt_solver_test.py on gpt2 medium
python auto_ckpt_solver_test.py --model gpt2

# run auto_ckpt_solver_test.py on resnet50
python auto_ckpt_solver_test.py --model resnet50

# tun auto_ckpt_batchsize_test.py
python auto_ckpt_batchsize_test.py
```

There are some results for your reference

## Auto Checkpoint Solver Test

### ResNet 50
![](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/tutorial/resnet50_benchmark.png)

### GPT2 Medium
![](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/tutorial/gpt2_benchmark.png)

## Auto Checkpoint Batch Size Test
```bash
===============test summary================
batch_size: 512, peak memory: 73314.392 MB, through put: 254.286 images/s
batch_size: 1024, peak memory: 73316.216 MB, through put: 397.608 images/s
batch_size: 2048, peak memory: 72927.837 MB, through put: 277.429 images/s
```
