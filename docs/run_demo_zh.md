# 快速上手

ColossalAI是一个大规模深度学习框架，其中包含高效的并行技术。该框架可以在多GPU的分布式系统上使用并行技术有效地加速模型训练，同时该框架也可以运行在
带有GPU的非分布式系统上。下面是ColossalAI的快速上手指南。

## 单GPU系统

在带有GPU的非分布式系统上进行模型训练时，ColossalAI可以达到当前的基线效率。
[这里](https://colab.research.google.com/drive/1fJnqqFzPuzZ_kn1lwCpG2nh3l2ths0KE?usp=sharing#scrollTo=cQ_y7lBG09LS)我们给出一个Google Colab
示例展现如何使用ColossalAI与CIFAR10数据集在非分布式系统上训练一个LeNet模型。

## 多GPU系统

在多GPU的分布式系统上训练深度学习模型时，ColossalAI可以使用高效的并行技术来显著地加速训练过程，这些技术将在下面的[并行技术](parallelization.md)章节中被详述。
下面的代码将在拥有四个GPU的分布式系统上训练一个ViT模型，其中`HOST`变量为您分布式系统的IP地址。请注意下面的代码使用了
[Slurm](https://slurm.schedmd.com/documentation.html)作业调度系统。

```bash
HOST=xxx.xxx.xxx.xxx srun ./scripts/slurm_dist_train.sh ./examples/run_trainer.py ./configs/vit/vit_2d.py
```

`./configs/vit/vit_2d.py`是一个[配置文件](config.md)，ColossalAI使用配置文件来定义训练过程中需要用到的参数，比如模型类型、数据集、以及优化器、学习率调度器等。
您可以通过编写配置文件的方式来训练不同的模型。`./examples/run_trainer.py`是一个标准的训练脚本，具体代码已经附在下面。该脚本可以读入配置文件中的训练参数并训练模型。

```python
import colossalai
from colossalai.core import global_context as gpc
from colossalai.engine import Engine
from colossalai.logging import get_global_dist_logger
from colossalai.trainer import Trainer

def run_trainer():
    model, train_dataloader, test_dataloader, criterion, optimizer, schedule, lr_scheduler = colossalai.initialize()
    logger = get_global_dist_logger()
    schedule.data_sync = False
    engine = Engine(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        schedule=schedule
    )
    logger.info("engine is built", ranks=[0])

    trainer = Trainer(engine=engine,
                      hooks_cfg=gpc.config.hooks,
                      verbose=True)
    logger.info("trainer is built", ranks=[0])

    logger.info("start training", ranks=[0])
    trainer.fit(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        max_epochs=gpc.config.num_epochs,
        display_progress=True,
        test_interval=2
    )

if __name__ == '__main__':
    run_trainer()
```

上面代码中的`model`变量可以被替换为一个自定义的模型或者`Model Zoo`中一个事先定义的模型，以此来达到训练不同模型的目的，[这里](model.md)详述了如何进行这样的替换。

## 系统功能

ColossalAI提供了一系列并行组件来加速您的模型训练，我们在下面的章节提供了关于这些并行组件的介绍。我们的目标是使您的分布式深度学习模型开发像单卡深度学习模型开发那样方便。

- [数据并行](parallelization.md)
- [1D、2D、2.5D、3D张量并行以及序列并行](parallelization.md)
- [流水线并行](parallelization.md)
- [训练器以及引擎](trainer_engine.md)
- [自定义您的并行模式](add_your_parallel.md)
- [混合精度训练](amp.md)
- [ZeRO优化器](zero.md)
