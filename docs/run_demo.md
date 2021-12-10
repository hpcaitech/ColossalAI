# Quick demo

Colossal-AI is an integrated large-scale deep learning system with efficient parallelization techniques. The system can
accelerate model training on distributed systems with multiple GPUs by applying parallelization techniques. The system
can also run on systems with only one GPU. Quick demos showing how to use Colossal-AI are given below.

## Single GPU

Colossal-AI can be used to train deep learning models on systems with only one GPU and achieve baseline
performances. We provided an example to train ResNet on CIFAR10 data with only one GPU. You can find this example in 
`examples\resnet_cifar10_data_parallel` in the repository. Detailed instructions can be found in its `README.md`.

## Multiple GPUs

Colossal-AI can be used to train deep learning models on distributed systems with multiple GPUs and accelerate the
training process drastically by applying efficient parallelization techiniques, which will be elaborated in
the [Parallelization](parallelization.md) section below. 

You can turn the resnet example mentioned above into a multi-GPU training by setting `--nproc_per_node` to be the number of 
GPUs you have on your system. We also provide an example of Vision Transformer which relies on
training with more GPUs. You can visit this example in `examples\vit_b16_imagenet_data_parallel`. It has a detailed instructional 
`README.md` for you too.


## Sample Training Script

Below is a typical way of how you train the model using 

```python
import colossalai
from colossalai.amp import AMP_TYPE
from colossalai.logging import get_dist_logger
from colossalai.trainer import Trainer, hooks
from colossalai.utils import get_dataloader


CONFIG = dict(
    parallel=dict(
        pipeline=1,
        tensor=1, mode=None
    ),
    fp16 = dict(
        mode=AMP_TYPE.TORCH
    ),
    gradient_accumulation=4,
    clip_grad_norm=1.0
)

def run_trainer():
    parser = colossalai.get_default_parser()
    args = parser.parse_args()
    colossalai.launch(config=CONFIG,
                      rank=args.rank,
                      world_size=args.world_size,
                      host=args.host,
                      port=args.port,
                      backend=args.backend)

    logger = get_dist_logger()

    # instantiate your compoentns
    model = MyModel()
    optimizer = MyOptimizer(model.parameters(), ...)
    train_dataset = TrainDataset()
    test_dataset = TestDataset()
    train_dataloader = get_dataloader(train_dataset, ...)
    test_dataloader = get_dataloader(test_dataset, ...)
    lr_scheduler = MyScheduler()
    logger.info("components are built")

    engine, train_dataloader, test_dataloader, lr_scheduler = colossalai.initialize(model, 
                                                                                    optimizer, 
                                                                                    criterion, 
                                                                                    train_dataloader, 
                                                                                    test_dataloader, 
                                                                                    lr_scheduler)

    trainer = Trainer(engine=engine,
                      verbose=True)

    hook_list = [
        hooks.LossHook(),
        hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=False),
        hooks.AccuracyHook(),
        hooks.TensorboardHook(log_dir='./tb_logs', ranks=[0]),
        hooks.LogMetricByEpochHook(logger),
        hooks.LogMemoryByEpochHook(logger),
        hooks.SaveCheckpointHook(checkpoint_dir='./ckpt')
    ]

    trainer.fit(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        epochs=NUM_EPOCH,
        hooks=hook_list,
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
