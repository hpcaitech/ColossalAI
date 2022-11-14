# Auto-Parallelism with ResNet

## 🚀Quick Start
### Auto-Parallel Tutorial
1. Install `pulp` and `coin-or-cbc` for the solver.
```bash
pip install pulp
conda install -c conda-forge coin-or-cbc
```
2. Run the auto parallel resnet example with 4 GPUs with synthetic dataset.
```bash
colossalai run --nproc_per_node 4 auto_parallel_with_resnet.py -s
```

You should expect to the log like this. This log shows the edge cost on the computation graph as well as the sharding strategy for an operation. For example, `layer1_0_conv1 S01R = S01R X RR` means that the first dimension (batch) of the input and output is sharded while the weight is not sharded (S means sharded, R means replicated), simply equivalent to data parallel training.
![](https://raw.githubusercontent.com/hpcaitech/public_assets/main/examples/tutorial/auto-parallel%20demo.png)


### Auto-Checkpoint Tutorial
1. Stay in the `auto_parallel` folder.
2. Install the dependencies.
```bash
pip install matplotlib transformers
```
3. Run a simple resnet50 benchmark to automatically checkpoint the model.
```bash
python auto_ckpt_solver_test.py --model resnet50
```

You should expect the log to be like this
![](https://raw.githubusercontent.com/hpcaitech/public_assets/main/examples/tutorial/auto-ckpt%20demo.png)

This shows that given different memory budgets, the model is automatically injected with activation checkpoint and its time taken per iteration. You can run this benchmark for GPT as well but it can much longer since the model is larger.
```bash
python auto_ckpt_solver_test.py --model gpt2
```

4. Run a simple benchmark to find the optimal batch size for checkpointed model.
```bash
python auto_ckpt_batchsize_test.py
```

You can expect the log to be like
![](https://raw.githubusercontent.com/hpcaitech/public_assets/main/examples/tutorial/auto-ckpt%20batchsize.png)


## Prepare Dataset

We use CIFAR10 dataset in this example. You should invoke the `donwload_cifar10.py` in the tutorial root directory or directly run the `auto_parallel_with_resnet.py`.
The dataset will be downloaded to `colossalai/examples/tutorials/data` by default.
If you wish to use customized directory for the dataset. You can set the environment variable `DATA` via the following command.

```bash
export DATA=/path/to/data
```

## extra requirements to use autoparallel

```bash
pip install pulp
conda install coin-or-cbc
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
