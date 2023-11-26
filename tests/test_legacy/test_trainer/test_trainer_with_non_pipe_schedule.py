import pytest
import torch

import colossalai
from colossalai.legacy.amp.amp_type import AMP_TYPE
from colossalai.legacy.trainer import Trainer
from colossalai.logging import get_dist_logger
from colossalai.testing import DummyDataloader, parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils import MultiTimer
from tests.kit.model_zoo import model_zoo

BATCH_SIZE = 4
IMG_SIZE = 32
NUM_EPOCHS = 200

CONFIG = dict(fp16=dict(mode=AMP_TYPE.TORCH))


@parameterize("model_name", ["custom_repeated_computed_layers", "torchvision_resnet18", "custom_nested_model"])
def run_trainer(model_name):
    model_builder, data_gen_fn, *_ = next(iter(model_zoo.get_sub_registry(model_name).values()))
    model = model_builder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_dataloader = DummyDataloader(data_gen_fn)
    test_dataloader = DummyDataloader(data_gen_fn)
    criterion = lambda x: x.sum()
    engine, train_dataloader, *_ = colossalai.legacy.initialize(
        model=model, optimizer=optimizer, criterion=criterion, train_dataloader=train_dataloader
    )

    logger = get_dist_logger()
    logger.info("engine is built", ranks=[0])

    timer = MultiTimer()
    trainer = Trainer(engine=engine, logger=logger, timer=timer)
    logger.info("trainer is built", ranks=[0])

    logger.info("start training", ranks=[0])
    trainer.fit(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        epochs=NUM_EPOCHS,
        max_steps=3,
        display_progress=True,
        test_interval=5,
    )
    torch.cuda.empty_cache()


def run_dist(rank, world_size, port):
    colossalai.legacy.launch(
        config=CONFIG, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl"
    )


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_trainer_no_pipeline():
    world_size = 4
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_trainer_no_pipeline()
