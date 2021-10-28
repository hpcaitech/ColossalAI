# Zero Redundancy Optimizer and Zero Offload

The Zero Redundancy Optimizer (ZeRO) removes the memory redundancies across data-parallel processes by partitioning the three model states (optimizer states, gradients, and parameters) across data-parallel processes instead of replicating them. By doing this, it boosts memory efficiency compared to classic data-parallelism while retaining its computational granularity and communication efficiency.

1. **ZeRO Level 1**: The optimizer states (e.g., for [Adam optimizer](https://arxiv.org/abs/1412.6980), 32-bit weights, and the first, and second moment estimates) are partitioned across the processes, so that each process updates only its partition.
2. **ZeRO Level 2**: The reduced 32-bit gradients for updating the model weights are also partitioned such that each process retains only the gradients corresponding to its portion of the optimizer states.
3. **ZeRO Level 3**: The 16-bit model parameters are partitioned across the processes. ZeRO-3 will automatically collect and partition them during the forward and backward passes.

## Getting Started

Once you are training with ColossalAI, enabling ZeRO-3 offload is as simple as enabling it in your ColossalAI configuration! Below are a few examples of ZeRO-3 configurations. 

### Example ZeRO-3 Configurations

Here we use ``Adam`` as the initial optimizer.

1. Use ZeRO to partition the optimizer states (level 1), gradients (level 2), and parameters (level 3).
    ```python
    optimizer = dict(
        type='Adam',
        lr=0.001,
        weight_decay=0
    )

    zero = dict(
        type='ZeroRedundancyOptimizer_Level_3',
        dynamic_loss_scale=True,
        clip_grad=1.0
    )
    ```
2. Additionally offload the optimizer states and computations to the CPU.
    ```python
    zero = dict(
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

Note that ``fp16`` is automatically enabled when using ZeRO. 

### Training

Once you complete your configuration, just use `colossalai.initialize()` to initialize your training. All you need to do is to write your configuration.