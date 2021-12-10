# Mixed precision training

In Colossal-AI, we have incorporated different implementations of mixed precision training:
1. torch.cuda.amp
2. apex.amp
3. naive amp

The first two rely on the original implementation of [PyTorch](https://pytorch.org/docs/stable/amp.html)
(version 1.6 and above) and [Nvidia Apex](https://github.com/NVIDIA/apex). The last mehtod is simialr to Apex O2 level.

Among these methods, apex.amp is not compatible with tensor parallelism. This is because that tensors are split across devices 
in tensor parallelism, thus, it is required to communicate among different processes to check if `inf` or `nan` occurs in the 
whole model weights. **We modified the torch amp implementation so that it is compatible with tensor parallelism now.**

To use mixed precision training, you can easily specify the `fp16` field in the config file to be True. Currently, PyTorch and 
Apex amp cannot be guaranteed to work with tensor and pipeline parallelism. We recommend you to use torch amp as it generally 
gives better accuracy than naive amp.

The AMP module is designed to be completely modular and can be used independently from other colossalai modules.
If you wish to only use amp in your code base without `colossalai.initialize`, you can use `colossalai.amp.convert_to_amp`.

```python
from colossalai.amp import AMP_TYPE

# exmaple of using torch amp
model, optimizer, criterion = colossalai.amp.convert_to_amp(model, 
                                                            optimizer, 
                                                            criterion,
                                                            AMP_TYPE.TORCH)
```

## PyTorch AMP

PyTorch provides mixed precision training in version 1.6 and above. It provides an easy way to cast data to `fp16` format 
while keeping some operations such as reductions in `fp32`. You can configure the gradient scaler in the config file.

```python
from colossalai.amp import AMP_TYPE

fp16=dict(
    mode=AMP_TYPE.TORCH,
    # below are default values for grad scaler
    init_scale=2.**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
    enabled=True
)
```

## Apex AMP

For this mode, we rely on the [Apex](https://nvidia.github.io/apex/) implementation for mixed precision training. We support 
this plugin because it allows for finer control on the granularity of mixed precision. For example, `O2` level (optimization level 2) 
will keep batch normalization in `fp32`.

The following code block shows a config file for Apex AMP.

```python
from colossalai.amp import AMP_TYPE

fp16 = dict(
    mode=AMP_TYPE.APEX,
    # below are the default values
    enabled=True, 
    opt_level='O1', 
    cast_model_type=None, 
    patch_torch_functions=None, 
    keep_batchnorm_fp32=None, 
    master_weights=None, 
    loss_scale=None, 
    cast_model_outputs=None,
    num_losses=1, 
    verbosity=1, 
    min_loss_scale=None, 
    max_loss_scale=16777216.0
)
```

## Tensor Parallel AMP

We leveraged the Megatron-LM implementation to achieve mixed precision training while maintaining compatibility with complex tensor 
and pipeline parallelism.

The following conde block show a config file for this mode.

```python
from colossalai.amp import AMP_TYPE

fp16 = dict(
    mode=AMP_TYPE.NAIVE,
    # below are the default values
    clip_grad=0,
    log_num_zeros_in_grad=False,
    initial_scale=2 ** 32,
    min_scale=1,
    growth_factor=2,
    backoff_factor=0.5,
    growth_interval=1000,
    hysteresis=2
)
```