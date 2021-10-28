# Quick demo

ColossalAI is an integrated large-scale deep learning framework with efficient parallelization techniques. The framework
can accelerate model training on distributed systems with multiple GPUs by applying parallelization techniques. The
framework can also run on systems with only one GPU. Quick demos showing how to use ColossalAI are given below.

## Single GPU

ColossalAI can be used to train deep learning models on systems with only one GPU and achieve baseline
performances. [Here](https://colab.research.google.com/drive/1fJnqqFzPuzZ_kn1lwCpG2nh3l2ths0KE?usp=sharing#scrollTo=cQ_y7lBG09LS)
is an example showing how to train a LeNet model on the CIFAR10 dataset using ColossalAI.

## Multiple GPUs

ColossalAI can be used to train deep learning models on distributed systems with multiple GPUs and accelerate the
training process drastically by applying efficient parallelization techiniques, which will be elaborated in
the [Parallelization](parallelization.md) section below. Run the code below on your distributed system with 4 GPUs,
where `HOST` is the IP address of your system. Note that we use
the [Slurm](https://slurm.schedmd.com/documentation.html) job scheduling system here.

```bash
HOST=xxx.xxx.xxx.xxx srun ./scripts/slurm_dist_train.sh ./example/train_vit_2d.py ./configs/vit/vit_2d.py
```

`./configs/vit/vit_2d.py` is a config file, which is introduced in the [Config file](config.md) section below. These
config files are used by ColossalAI to define all kinds of training arguments, such as the model, dataset and training
method (optimizer, lr_scheduler, epoch, etc.). Config files are highly customizable and can be modified so as to train
different models.
`./example/run_trainer.py` contains a standard training script and is presented below, it reads the config file and
realizes the training process.

```python
import colossalai
from colossalai.engine import Engine
from colossalai.trainer import Trainer
from colossalai.core import global_context as gpc

model, train_dataloader, test_dataloader, criterion, optimizer, schedule, lr_scheduler = colossalai.initialize()
engine = Engine(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    schedule=schedule
)

trainer = Trainer(engine=engine,
                  hooks_cfg=gpc.config.hooks,
                  verbose=True)
trainer.fit(
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    max_epochs=gpc.config.num_epochs,
    display_progress=True,
    test_interval=5
)
```

Alternatively, the `model` variable can be substituted with a self-defined model or a pre-defined model in our Model
Zoo. The detailed substitution process is elaborated [here](model.md).

## Features

ColossalAI provides a collection of parallel training components for you. We aim to support you with your development of
distributed deep learning models just like how you write single-GPU deeo learning models. We provide friendly tools to
kickstart distributed training in a few lines.

- [Data Parallelism](parallelization.md)
- [Pipeline Parallelism](parallelization.md)
- [1D, 2D, 2.5D, 3D and sequence parallelism](parallelization.md)
- [Friendly trainer and engine](trainer_engine.md)
- [Extensible for new parallelism](add_your_parallel.md)
- [Mixed Precision Training](amp.md)
- [Zero Redundancy Optimizer (ZeRO)](zero.md)
