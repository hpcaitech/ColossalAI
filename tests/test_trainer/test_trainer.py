import colossalai
from colossalai.core import global_context as gpc
from colossalai.logging import get_global_dist_logger
from colossalai.trainer import Trainer


def test_trainer():
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
        hooks_cfg=gpc.config.hooks,
        epochs=gpc.config.num_epochs,
        display_progress=False,
        test_interval=5
    )


if __name__ == '__main__':
    test_trainer()
