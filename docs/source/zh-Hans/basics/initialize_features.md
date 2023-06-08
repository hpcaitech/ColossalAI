# 初始化功能

作者: Shenggui Li, Siqi Mai

> ⚠️ 此页面上的信息已经过时并将被废弃。请在[Booster API](../basics/booster_api.md)页面查阅更新。

**预备知识:**
- [分布式训练](../concepts/distributed_training.md)
- [Colossal-AI 总览](../concepts/colossalai_overview.md)

## 简介

在本教程中，我们将介绍 `colossalai.initialize` 的使用。 它包含了如何将特征(例如，模型、优化器、数据加载器）无缝注入您的训练组件中。 调用 `colossalai.initialize` 是您进入训练循环前的基本操作。

在下面一节中，我们将介绍 `colossalai.initialize` 是如何工作的以及使用中我们要注意的细节。

## 使用

在一个典型的工作流程中，我们将在训练脚本的开始启动分布式环境。
之后，我们将实例化我们的对象，如模型、优化器、损失函数、数据加载器等。此时，我们可以使用 `colossalai.initialize` 便捷地为这些对象注入特征。
具体细节请看以下的伪代码例子。

```python
import colossalai
import torch
...


# launch distributed environment
colossalai.launch(config='./config.py', ...)

# create your objects
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
train_dataloader = MyTrainDataloader()
test_dataloader = MyTrainDataloader()

# initialize features
engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model,
                                                                     optimizer,
                                                                     criterion,
                                                                     train_dataloader,
                                                                     test_dataloader)
```

 `colossalai.initialize` 将返回一个 `Engine` 对象。 该对象把模型、优化器和损失函数封装起来。 **`Engine` 对象会以配置文件中指定的特征运行。**
关于 `Engine` 的更多使用细节可以在 [在训练中使用Engine和Trainer](./engine_trainer.md) 中获取。
