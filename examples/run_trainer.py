#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import colossalai
from colossalai.core import global_context as gpc
from colossalai.logging import get_global_dist_logger
from colossalai.trainer import Trainer


def run_trainer():
    engine, train_dataloader, test_dataloader = colossalai.initialize()
    logger = get_global_dist_logger()
    engine.schedule.data_sync = False

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
