# Overview

MoE is a new technique to enlarge neural networks while keeping the same throughput in our training. 
It is designed to improve the performance of our models without any additional time penalty. But now using 
our temporary moe parallelism will cause a moderate computation overhead and additoinal communication time.
The communication time depends on the topology of network in running environment. At present, moe parallelism
may not meet what you want. Optimized version of moe parallelism will come soon.

This is a simple example about how to run widenet-tiny on cifar10. More information about widenet can be 
found [here](https://arxiv.org/abs/2107.11817).

# How to run

On a single server, you can directly use torchrun to start pre-training on multiple GPUs in parallel. 
If you use the script here to train, just use follow instruction in your terminal. `n_proc` is the 
number of processes which commonly equals to the number GPUs.

```shell
torchrun --nnodes=1 --nproc_per_node=4 train.py \
    --config ./config.py
```

If you want to use multi servers, please check our document about environment initialization.

Make sure to initialize moe running environment by `moe_set_seed` before building the model.

# Result

The result of training widenet-tiny on cifar10 from scratch is 89.93%. Since moe makes the model larger
than other vit-tiny models, mixup and rand augmentation is needed.