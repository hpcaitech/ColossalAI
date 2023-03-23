# 梯度裁剪

作者: Boxiang Wang, Haichen Huang, Yongbin Li

**前置教程**
- [定义配置文件](../basics/define_your_config.md)
- [在训练中使用Engine和Trainer](../basics/engine_trainer.md)

**示例代码**
- [ColossalAI-Examples Gradient Clipping](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/features/gradient_clipping)

**相关论文**
- [On the difficulty of training Recurrent Neural Networks](https://arxiv.org/abs/1211.5063)

## 引言

为了加快训练过程和寻求全局最优以获得更好的性能，越来越多的学习率调度器被提出。人们通过控制学习率来调整训练中的下降速度。这使得梯度向量在每一步都能更好地统一。在这种情况下，下降速度可以按预期被控制。
因此，梯度裁剪，一种可以将梯度向量归一化，以将其限制在统一长度的技术，对于那些希望模型性能更好的人来说是不可或缺的。

在使用 Colossal-AI 时，你不必担心实现梯度剪裁，我们以一种有效而方便的方式支持梯度剪裁。你所需要的只是在你的配置文件中增加一个命令。

## 为什么应该使用 Colossal-AI 中的梯度裁剪

我们不建议用户自己编写梯度剪裁，因为朴素的梯度剪裁在应用张量并行、流水线并行、MoE 等功能时可能会失败。

根据下图，每个 GPU 只拥有线性层中权重的一部分参数。为了得到线性层权重的梯度向量的正确范数，每个 GPU 中的每个梯度向量的范数应该相加。更复杂的是，偏置的分布不同于权重的分布。通信组在求和运算中有所不同。

(注: 这种情况是旧版本的 2D 并行，在代码中的实现是不一样的。但这是一个很好的例子，能够说明在梯度剪裁中统一所有通信的困难。)

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/01/28/KXiJPHt3Dum82cA.png"/>
<figcaption>参数分布</figcaption>
</figure>

不用担心它，因为 Colossal-AI 已经为你处理好。

### 使用
要使用梯度裁剪，只需在配置文件中添加梯度裁剪范数即可。

```python
clip_grad_norm = 1.0
```

### 实例

我们提供了一个展现梯度裁剪的[运行实例](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/features/gradient_clipping)
。在本例中，我们将梯度裁剪范数设置为1.0，你可以使用以下命令运行脚本：

```shell
python -m torch.distributed.launch --nproc_per_node 1 --master_addr localhost --master_port 29500  train_with_engine.py
```
