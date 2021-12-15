# Config file

Here is a config file example showing how to train a ViT model on the CIFAR10 dataset using Colossal-AI:

```python
# optional
# three keys: pipeline, tensor
# data parallel size is inferred
parallel = dict(
    pipeline=dict(size=1),
    tensor=dict(size=4, mode='2d'),
)

# optional
# pipeline or no pipeline schedule
fp16 = dict(
    mode=AMP_TYPE.NAIVE,
    initial_scale=2 ** 8
)

# optional
# configuration for zero
# you can refer to the Zero Redundancy optimizer and zero offload section for details
# https://www.colossalai.org/zero.html
zero = dict(
    level=<int>,
    ...
)

# optional
# if you are using complex gradient handling
# otherwise, you do not need this in your config file
# default gradient_handlers = None
gradient_handlers = [dict(type='MyHandler', arg1=1, arg=2), ...]

# optional
# specific gradient accumulation size
# if your batch size is not large enough
gradient_accumulation = <int>

# optional
# add gradient clipping to your engine
# this config is not compatible with zero and AMP_TYPE.NAIVE
# but works with AMP_TYPE.TORCH and AMP_TYPE.APEX
# defautl clip_grad_norm = 0.0
clip_grad_norm = <float>

# optional
# cudnn setting
# default is like below
cudnn_benchmark = False,
cudnn_deterministic=True,

```