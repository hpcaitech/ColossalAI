# 梯度 Handler

作者: Shenggui Li, Yongbin Li

**前置教程**
- [定义配置文件](../basics/define_your_config.md)
- [在训练中使用Engine和Trainer](../basics/engine_trainer.md)

**示例代码**
- [ColossalAI-Examples Gradient Handler](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/features/gradient_handler)

## 引言

在分布式训练中，每次迭代结束时都需要梯度同步。这很重要，因为我们需要确保在不同的机器中使用相同的梯度更新参数，以便生成的参数都一样。这通常在数据并行中看到，因为在数据并行中的模型是直接复制的。

在 Colossal-AI 中，我们为用户提供了一个接口来定制他们想要如何处理同步。这为实现新的并行方法等情况带来了灵活性。

当梯度 Handler 被使用时, PyTorch 的 `DistributedDataParallel` 将不再被使用，因为它会自动同步梯度.

## 定制你的梯度 Handler

要实现定制的梯度Handler，需要遵循以下步骤。
1. 继承Colossal-AI中的 `BaseGradientHandler`
2. 将梯度Handler注册进 `GRADIENT_HANDLER`
3. 实现 `handle_gradient`

```python
from colossalai.registry import GRADIENT_HANDLER
from colossalai.engine.gradient_handler import BaseGradientHandler


@GRADIENT_HANDLER.register_module
class MyGradientHandler(BaseGradientHandler):

    def handle_gradient(self):
        do_something()


```


## 使用

要使用梯度 Handler，需要在配置文件中指定梯度 Handler。梯度 Handler 将自动构建并连接到 Engine。

```python
gradient_handler = [dict(type='MyGradientHandler')]
```


### 实例

我们提供了一个 [运行实例](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/features/gradient_handler)
展现梯度 Handler 的使用. 在这个例子中，我们使用 `DataParallelGradientHandler` 而不是 PyTorch 的
`DistributedDataParallel` 实现数据并行.

```shell
python -m torch.distributed.launch --nproc_per_node 4 --master_addr localhost --master_port 29500  train_with_engine.py
```
