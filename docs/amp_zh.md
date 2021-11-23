# 混合精度训练

Colossal-AI可以使用如下三种不同的混合精度训练方式：
1. torch.cuda.amp
2. apex.amp
3. 张量并行AMP

前两种混合精度训练方式依赖于[PyTorch](https://pytorch.org/docs/stable/amp.html)的原生实现（1.6或以上版本）以及[Nvidia Apex](https://github.com/NVIDIA/apex)，但这两种方法与张量并行并不兼容，因为在张量并行中我们需要将张量进行切分并保存在不同的设备上，因此，实现兼容张量并行的混合精度训练需要在不同进程之间不断通信来交流`inf`以及`nan`是否存在于模型参数中，因此我们采用了[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)的实现方式。

您可以简单地将配置文件中的`fp16`字段设置为True来使用混合精度训练。目前，PyTorch与Apex的amp不能保证与张量和流水线并行兼容，因此，我们推荐您使用最后一种混合精度训练方式。

## PyTorch AMP

PyTorch在1.6及以上版本中提供了混合精度训练，它可以在保持一些操作的精度为`fp32`的同时，将数据转换成`fp16`格式，您可以在配置文件中配置使用。

```python
from colossalai.engine import AMP_TYPE

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

我们使用了[Apex](https://nvidia.github.io/apex/)中的混合精度训练，因为该模式提供了细粒度的混合精度控制，例如，`O2`级（第二级优化器）将会保持批标准化在`fp32`上进行。下面的代码块展示了使用Apex AMP的配置文件。

```python
from colossalai.engine import AMP_TYPE

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

## 张量并行AMP

我们借鉴了Megatron-LM的混合精度训练实现，该实现方式与张量并行、流水线并行相兼容。下面的代码块展示了使用张量并行AMP的配置文件。

```python
from colossalai.engine import AMP_TYPE

fp16 = dict(
    mode=AMP_TYPE.PARALLEL,
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