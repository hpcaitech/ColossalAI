# 梯度累积 (旧版本)

作者: Shenggui Li, Yongbin Li

**前置教程**
- [定义配置文件](../basics/define_your_config.md)
- [在训练中使用Engine和Trainer](../basics/engine_trainer.md)

**示例代码**
- [ColossalAI-Examples Gradient Accumulation](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/features/gradient_accumulation)

## 引言

梯度累积是一种常见的增大训练 batch size 的方式。 在训练大模型时，内存经常会成为瓶颈，并且 batch size 通常会很小（如2），这导致收敛性无法保证。梯度累积将多次迭代的梯度累加，并仅在达到预设迭代次数时更新参数。

## 使用

在 Colossal-AI 中使用梯度累积非常简单，仅需将下列配置添加进 config 文件。其中，整数值代表期望梯度累积的次数。

```python
gradient_accumulation = <int>
```

## 实例

我们提供了一个 [运行实例](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/features/gradient_accumulation)
来展现梯度累积。在这个例子中，梯度累积次数被设置为4，你可以通过一下命令启动脚本

```shell
python -m torch.distributed.launch --nproc_per_node 1 --master_addr localhost --master_port 29500  run_resnet_cifar10_with_engine.py
```

你将会看到类似下方的文本输出。这展现了梯度虽然在前3个迭代中被计算，但直到最后一次迭代，参数才被更新。

```text
iteration 0, first 10 elements of param: tensor([-0.0208,  0.0189,  0.0234,  0.0047,  0.0116, -0.0283,  0.0071, -0.0359, -0.0267, -0.0006], device='cuda:0', grad_fn=<SliceBackward0>)
iteration 1, first 10 elements of param: tensor([-0.0208,  0.0189,  0.0234,  0.0047,  0.0116, -0.0283,  0.0071, -0.0359, -0.0267, -0.0006], device='cuda:0', grad_fn=<SliceBackward0>)
iteration 2, first 10 elements of param: tensor([-0.0208,  0.0189,  0.0234,  0.0047,  0.0116, -0.0283,  0.0071, -0.0359, -0.0267, -0.0006], device='cuda:0', grad_fn=<SliceBackward0>)
iteration 3, first 10 elements of param: tensor([-0.0141,  0.0464,  0.0507,  0.0321,  0.0356, -0.0150,  0.0172, -0.0118, 0.0222,  0.0473], device='cuda:0', grad_fn=<SliceBackward0>)
```
<!-- doc-test-command: torchrun --standalone --nproc_per_node=1 gradient_accumulation.py  -->
