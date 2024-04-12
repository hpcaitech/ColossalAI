import os

import torch
from titans.model.vit.vit import _create_vit_model
from tqdm import tqdm

import colossalai
from colossalai.legacy.context import ParallelMode
from colossalai.legacy.core import global_context as gpc
from colossalai.legacy.nn import CrossEntropyLoss
from colossalai.legacy.pipeline.pipelinable import PipelinableContext
from colossalai.logging import get_dist_logger
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.utils import is_using_pp


class DummyDataloader:
    def __init__(self, length, batch_size):
        self.length = length
        self.batch_size = batch_size

    def generate(self):
        data = torch.rand(self.batch_size, 3, 224, 224)
        label = torch.randint(low=0, high=10, size=(self.batch_size,))
        return data, label

    def __iter__(self):
        self.step = 0
        return self

    def __next__(self):
        if self.step < self.length:
            self.step += 1
            return self.generate()
        else:
            raise StopIteration

    def __len__(self):
        return self.length


def main():
    # launch from torch
    parser = colossalai.legacy.get_default_parser()
    args = parser.parse_args()
    colossalai.legacy.launch_from_torch(config=args.config)

    # get logger
    logger = get_dist_logger()
    logger.info("initialized distributed environment", ranks=[0])

    if hasattr(gpc.config, "LOG_PATH"):
        if gpc.get_global_rank() == 0:
            log_path = gpc.config.LOG_PATH
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            logger.log_to_file(log_path)

    use_pipeline = is_using_pp()

    # create model
    model_kwargs = dict(
        img_size=gpc.config.IMG_SIZE,
        patch_size=gpc.config.PATCH_SIZE,
        hidden_size=gpc.config.HIDDEN_SIZE,
        depth=gpc.config.DEPTH,
        num_heads=gpc.config.NUM_HEADS,
        mlp_ratio=gpc.config.MLP_RATIO,
        num_classes=10,
        init_method="jax",
        checkpoint=gpc.config.CHECKPOINT,
    )

    if use_pipeline:
        pipelinable = PipelinableContext()
        with pipelinable:
            model = _create_vit_model(**model_kwargs)
        pipelinable.to_layer_list()
        pipelinable.policy = "uniform"
        model = pipelinable.partition(1, gpc.pipeline_parallel_size, gpc.get_local_rank(ParallelMode.PIPELINE))
    else:
        model = _create_vit_model(**model_kwargs)

    # count number of parameters
    total_numel = 0
    for p in model.parameters():
        total_numel += p.numel()
    if not gpc.is_initialized(ParallelMode.PIPELINE):
        pipeline_stage = 0
    else:
        pipeline_stage = gpc.get_local_rank(ParallelMode.PIPELINE)
    logger.info(f"number of parameters: {total_numel} on pipeline stage {pipeline_stage}")

    # use synthetic dataset
    # we train for 10 steps and eval for 5 steps per epoch
    train_dataloader = DummyDataloader(length=10, batch_size=gpc.config.BATCH_SIZE)
    test_dataloader = DummyDataloader(length=5, batch_size=gpc.config.BATCH_SIZE)

    # create loss function
    criterion = CrossEntropyLoss(label_smoothing=0.1)

    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=gpc.config.LEARNING_RATE, weight_decay=gpc.config.WEIGHT_DECAY)

    # create lr scheduler
    lr_scheduler = CosineAnnealingWarmupLR(
        optimizer=optimizer, total_steps=gpc.config.NUM_EPOCHS, warmup_steps=gpc.config.WARMUP_EPOCHS
    )

    # initialize
    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
    )

    logger.info("Engine is built", ranks=[0])

    for epoch in range(gpc.config.NUM_EPOCHS):
        # training
        engine.train()
        data_iter = iter(train_dataloader)

        if gpc.get_global_rank() == 0:
            description = "Epoch {} / {}".format(epoch, gpc.config.NUM_EPOCHS)
            progress = tqdm(range(len(train_dataloader)), desc=description)
        else:
            progress = range(len(train_dataloader))
        for _ in progress:
            engine.zero_grad()
            engine.execute_schedule(data_iter, return_output_label=False)
            engine.step()
            lr_scheduler.step()
    gpc.destroy()


if __name__ == "__main__":
    main()
