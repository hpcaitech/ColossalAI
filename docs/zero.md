# Zero Redundancy optimizer and zero offload

The Zero Redundancy Optimizer (ZeRO) removes the memory redundancies across data-parallel processes by partitioning three 
model states (optimizer states, gradients, and parameters) instead of replicating them. 
By doing so, memory efficiency is boosted drastically compared to classic data parallelism while the computational granularity 
and communication efficiency are retained.

1. **ZeRO Level 1**: The optimizer states (e.g., for [Adam optimizer](https://arxiv.org/abs/1412.6980), 32-bit weights, and the 
first and second momentum estimates) are partitioned across the processes, so that each process updates only its partition.
2. **ZeRO Level 2**: The reduced 32-bit gradients for updating the model weights are also partitioned such that each process 
only stores the gradients corresponding to its partition of the optimizer states.
3. **ZeRO Level 3**: The 16-bit model parameters are partitioned across the processes. ZeRO-3 will automatically collect and 
partition them during the forward and backward passes.

## Getting Started with ZeRO

If you are training models with Colossal-AI, enabling ZeRO DP and Offloading is easy by addding several lines in your configuration file. We support configration for level 2 and 3. You have use [PyTorch native implementation](https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html) for level 1 optimizer.
Below are a few examples of ZeRO-3 configurations.

### Example of ZeRO-3 Configurations

Here we use `Adam` as the initial optimizer.

1. Use ZeRO to partition the optimizer states, gradients (level 2), and parameters (level 3).
    ```python
    zero = dict(
        level=3,
        dynamic_loss_scale=True,
        clip_grad=1.0
    )
    ```

2. Additionally offload the optimizer states and computations to the CPU.
    ```python
    zero = dict(
        level=3,
        offload_optimizer_config=dict(
            device='cpu',
            pin_memory=True,
            fast_init=True
        ),
        ...
    )
    ```
3. Save even more memory by offloading parameters to the CPU memory.
    ```python
    zero = dict(
        level=3,
        offload_optimizer_config=dict(
            device='cpu',
            pin_memory=True,
            fast_init=True
        ),
        offload_param_config=dict(
            device='cpu',
            pin_memory=True,
            fast_init=OFFLOAD_PARAM_MAX_IN_CPU
        ),
        ...
    )
    ```
4. Save even MORE memory by offloading to NVMe (if available on your system):
    ```python
    zero = dict(
        level=3,
        offload_optimizer_config=dict(
            device='nvme',
            pin_memory=True,
            fast_init=True,
            nvme_path='/nvme_data'
        ),
        offload_param_config=dict(
            device='nvme',
            pin_memory=True,
            max_in_cpu=OFFLOAD_PARAM_MAX_IN_CPU,
            nvme_path='/nvme_data'
        ),
        ...
    )
    ```

Note that `fp16` is automatically enabled when using ZeRO. This relies on `AMP_TYPE.NAIVE` in Colossal-AI AMP module.

### Training

Once you have completed your configuration, just use `colossalai.initialize()` to initialize your training.
