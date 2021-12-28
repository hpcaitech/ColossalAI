import os.path as osp

import pytest
import torch
import torch.multiprocessing as mp

from colossalai.builder.pipeline import build_pipeline_model_from_cfg
from colossalai.core import global_context
from colossalai.initialize import launch
from colossalai.logging import get_dist_logger
from functools import partial
from colossalai.utils import free_port

DIR_PATH = osp.dirname(osp.realpath(__file__))
CONFIG_PATH = osp.join(DIR_PATH, 'resnet_config.py')


def run_partition(rank, world_size, port):
    launch(config=CONFIG_PATH,
           rank=rank,
           world_size=world_size,
           host='localhost',
           port=port,
           backend='nccl'
           )
    logger = get_dist_logger()
    logger.info('finished initialization')

    # build model
    model = build_pipeline_model_from_cfg(global_context.config.model, 1, verbose=True)
    assert isinstance(model, torch.nn.Module)
    logger.info('model is created')

    global_context.destroy()
    logger.info('training finished')
    torch.cuda.empty_cache()


@pytest.mark.dist
def test_partition():
    world_size = 4
    run_func = partial(run_partition, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_partition()
