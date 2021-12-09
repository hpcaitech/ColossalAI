import os.path as osp

import pytest
import torch
from torch.utils.data import DataLoader

from colossalai.builder import build_dataset, ModelInitializer
from colossalai.core import global_context
from colossalai.initialize import init_dist
from colossalai.logging import get_dist_logger

DIR_PATH = osp.dirname(osp.realpath(__file__))
CONFIG_PATH = osp.join(DIR_PATH, '../configs/pipeline_vanilla_resnet.py')


@pytest.mark.skip("This test should be invoked using the test.sh provided")
@pytest.mark.dist
def test_partition():
    init_dist(CONFIG_PATH)
    logger = get_dist_logger()
    logger.info('finished initialization')

    # build model
    model = ModelInitializer(global_context.config.model, 1, verbose=True).model_initialize()
    logger.info('model is created')

    dataset = build_dataset(global_context.config.train_data.dataset)
    dataloader = DataLoader(dataset=dataset, **global_context.config.train_data.dataloader)
    logger.info('train data is created')

    global_context.destroy()
    torch.cuda.synchronize()
    logger.info('training finished')


if __name__ == '__main__':
    test_partition()
