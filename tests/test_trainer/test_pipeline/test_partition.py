import os.path as osp

import pytest
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

<<<<<<< HEAD
from colossalai.builder.pipeline import build_pipeline_model_from_cfg
=======
from colossalai.builder.pipeline import PipelineModel
>>>>>>> 75c1a14... integrated parallel layers for ease of building models
from colossalai.core import global_context
from colossalai.initialize import launch
from colossalai.logging import get_dist_logger
from functools import partial
import model

DIR_PATH = osp.dirname(osp.realpath(__file__))
CONFIG_PATH = osp.join(DIR_PATH, 'resnet_config.py')


def run_partition(rank, world_size):
    launch(config=CONFIG_PATH,
           rank=rank,
           world_size=world_size,
           host='localhost',
           port=29933,
           backend='nccl'
           )
    logger = get_dist_logger()
    logger.info('finished initialization')

    # build model
<<<<<<< HEAD
    model = build_pipeline_model_from_cfg(global_context.config.model, 1, verbose=True)
=======
    model = PipelineModel(global_context.config.model, 1, verbose=True)()
>>>>>>> 75c1a14... integrated parallel layers for ease of building models
    assert isinstance(model, torch.nn.Module)
    logger.info('model is created')

    global_context.destroy()
    logger.info('training finished')
    torch.cuda.empty_cache()


@pytest.mark.dist
def test_partition():
    world_size = 4
    run_func = partial(run_partition, world_size=world_size)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_partition()
