# Quick demo

Colossal-AI is an integrated large-scale deep learning system with efficient parallelization techniques. The system can
accelerate model training on distributed systems with multiple GPUs by applying parallelization techniques. The system
can also run on systems with only one GPU. Quick demos showing how to use Colossal-AI are given below.

## Single GPU

Colossal-AI can be used to train deep learning models on systems with only one GPU and achieve baseline
performances. [Here](https://colab.research.google.com/drive/1fJnqqFzPuzZ_kn1lwCpG2nh3l2ths0KE?usp=sharing#scrollTo=cQ_y7lBG09LS)
is an example showing how to train a LeNet model on the CIFAR10 dataset using Colossal-AI.

## Multiple GPUs

Colossal-AI can be used to train deep learning models on distributed systems with multiple GPUs and accelerate the
training process drastically by applying efficient parallelization techiniques, which will be elaborated in
the [Parallelization](parallelization.md) section below. Run the code below on your distributed system with 4 GPUs,
where `HOST` is the IP address of your system. Note that we use
the [Slurm](https://slurm.schedmd.com/documentation.html) job scheduling system here.

```bash
HOST=xxx.xxx.xxx.xxx srun ./scripts/slurm_dist_train.sh ./examples/run_trainer.py ./configs/vit/vit_2d.py
```

`./configs/vit/vit_2d.py` is a config file, which is introduced in the [Config file](config.md) section below. These
config files are used by Colossal-AI to define all kinds of training arguments, such as the model, dataset and training
method (optimizer, lr_scheduler, epoch, etc.). Config files are highly customizable and can be modified so as to train
different models.
`./examples/run_trainer.py` contains a standard training script and is presented below, it reads the config file and
realizes the training process.

```python
import colossalai
from colossalai.core import global_context as gpc
from colossalai.logging import get_global_dist_logger
from colossalai.trainer import Trainer


def run_trainer():
    engine, train_dataloader, test_dataloader = colossalai.initialize()
    logger = get_global_dist_logger()

    logger.info("engine is built", ranks=[0])

    trainer = Trainer(engine=engine,
                      verbose=True)
    logger.info("trainer is built", ranks=[0])

    logger.info("start training", ranks=[0])
    trainer.fit(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        epochs=gpc.config.num_epochs,
        hooks_cfg=gpc.config.hooks,
        display_progress=True,
        test_interval=2
    )


if __name__ == '__main__':
    run_trainer()
```

Alternatively, the `model` variable can be substituted with a self-defined model or a pre-defined model in our Model
Zoo. The detailed substitution process is elaborated [here](model.md).

## Features

Colossal-AI provides a collection of parallel training components for you. We aim to support you with your development
of distributed deep learning models just like how you write single-GPU deep learning models. We provide friendly tools
to kickstart distributed training in a few lines.

- [Data Parallelism](parallelization.md)
- [Pipeline Parallelism](parallelization.md)
- [1D, 2D, 2.5D, 3D and sequence parallelism](parallelization.md)
- [Friendly trainer and engine](trainer_engine.md)
- [Extensible for new parallelism](add_your_parallel.md)
- [Mixed Precision Training](amp.md)
- [Zero Redundancy Optimizer (ZeRO)](zero.md)
