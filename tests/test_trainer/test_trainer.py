import colossalai
from colossalai.core import global_context as gpc
from colossalai.engine import Engine
from colossalai.logging import get_global_dist_logger
from colossalai.trainer import Trainer


def test_trainer():
    model, train_dataloader, test_dataloader, criterion, optimizer, schedule, lr_scheduler = colossalai.initialize()
    logger = get_global_dist_logger()

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
        display_progress=False,
        test_interval=5
    )


if __name__ == '__main__':
    test_trainer()
