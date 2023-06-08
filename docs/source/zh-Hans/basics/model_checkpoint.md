# 模型Checkpoint

作者 : Guangyang Lu

> ⚠️ 此页面上的信息已经过时并将被废弃。请在[Booster Checkpoint](../basics/booster_checkpoint.md)页面查阅更新。

**预备知识:**
- [Launch Colossal-AI](./launch_colossalai.md)
- [Initialize Colossal-AI](./initialize_features.md)

**示例代码:**
- [ColossalAI-Examples Model Checkpoint](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/utils/checkpoint)

**函数是经验函数.**

## 简介

本教程将介绍如何保存和加载模型Checkpoint。

为了充分利用Colossal-AI的强大并行策略，我们需要修改模型和张量，可以直接使用 `torch.save` 或者 `torch.load` 保存或加载模型Checkpoint。在Colossal-AI中，我们提供了应用程序接口实现上述同样的效果。

但是，在加载时，你不需要使用与存储相同的保存策略。

## 使用方法

### 保存

有两种方法可以使用Colossal-AI训练模型，即使用engine或使用trainer。
**注意我们只保存 `state_dict`.** 因此，在加载Checkpoint时，需要首先定义模型。

#### 同 engine 保存

```python
from colossalai.utils import save_checkpoint
model = ...
engine, _, _, _ = colossalai.initialize(model=model, ...)
for epoch in range(num_epochs):
    ... # do some training
    save_checkpoint('xxx.pt', epoch, model)
```

#### 用 trainer 保存
```python
from colossalai.trainer import Trainer, hooks
model = ...
engine, _, _, _ = colossalai.initialize(model=model, ...)
trainer = Trainer(engine, ...)
hook_list = [
            hooks.SaveCheckpointHook(1, 'xxx.pt', model)
            ...]

trainer.fit(...
            hook=hook_list)
```

### 加载

```python
from colossalai.utils import load_checkpoint
model = ...
load_checkpoint('xxx.pt', model)
... # train or test
```
