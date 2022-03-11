from functools import partial

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.amp.amp_type import AMP_TYPE
from colossalai.logging import get_dist_logger
from colossalai.trainer import Trainer
from colossalai.utils import MultiTimer, free_port
from tests.components_to_test.registry import non_distributed_component_funcs

BATCH_SIZE = 16
IMG_SIZE = 32
NUM_EPOCHS = 200

CONFIG = dict(
    # Config
    fp16=dict(mode=AMP_TYPE.TORCH))


def run_trainer_no_pipeline(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    test_models = ['repeated_computed_layers', 'resnet18', 'nested_model']
    for name in test_models:
        get_components_func = non_distributed_component_funcs.get_callable(name)
        model_builder, train_dataloader, test_dataloader, optimizer_builder, criterion = get_components_func()
        model = model_builder()
        optimizer = optimizer_builder(model)
        engine, train_dataloader, *_ = colossalai.initialize(model=model,
                                                             optimizer=optimizer,
                                                             criterion=criterion,
                                                             train_dataloader=train_dataloader)

        logger = get_dist_logger()
        logger.info("engine is built", ranks=[0])

        timer = MultiTimer()
        trainer = Trainer(engine=engine, logger=logger, timer=timer)
        logger.info("trainer is built", ranks=[0])

        logger.info("start training", ranks=[0])
        trainer.fit(train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    epochs=NUM_EPOCHS,
                    max_steps=5,
                    display_progress=True,
                    test_interval=5)
        torch.cuda.empty_cache()


@pytest.mark.dist
def test_trainer_no_pipeline():
    world_size = 4
    run_func = partial(run_trainer_no_pipeline, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_trainer_no_pipeline()
