#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pytest
import torch
from common import CONFIG

import colossalai
from colossalai.logging import get_dist_logger
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils.cuda import get_current_device
from colossalai.utils.memory import colo_device_memory_used
from colossalai.zero.gemini.memory_tracer.utils import colo_model_mem_usage
from colossalai.zero.legacy.init_ctx import ZeroInitContext
from colossalai.zero.legacy.shard_utils import BucketTensorShardStrategy, TensorShardStrategy
from tests.components_to_test.registry import non_distributed_component_funcs


@parameterize("init_device_type", ['cpu', 'cuda'])
@parameterize("shard_strategy_class", [TensorShardStrategy, BucketTensorShardStrategy])
def run_model_test(init_device_type, shard_strategy_class):
    logger = get_dist_logger("test_zero_init")

    for name, get_components_func in non_distributed_component_funcs._registry.items():
        # because the ZeroInitContext automatically turns parameters to fp16
        # and the beit model use tensor.erfinv_() function to initialize weights
        # tensor.erfinv_() doesn't support Half in CPU, we omit the beit model
        if name == 'beit':
            continue
        model_builder, _, _, _, _ = get_components_func()
        if init_device_type == 'cuda':
            init_device = get_current_device()
        elif init_device_type == 'cpu':
            init_device = torch.device("cpu")
        else:
            continue

        model_numel_tensor = torch.zeros(1, dtype=torch.int)
        with ZeroInitContext(target_device=init_device,
                             shard_strategy=shard_strategy_class(),
                             shard_param=True,
                             model_numel_tensor=model_numel_tensor):
            model = model_builder(checkpoint=True)

        for param in model.parameters():
            assert hasattr(param, 'colo_attr')
            assert param.colo_attr.sharded_data_tensor.dtype == torch.half
            assert param.colo_attr.sharded_data_tensor.is_sharded
            assert param.colo_attr.data_payload.device.type == init_device.type, \
                f'{param.colo_attr.data_payload.device.type} vs. {init_device.type}'

        cuda_mem_use, _ = colo_model_mem_usage(model)
        model_data_cuda_mem_MB = cuda_mem_use / 1e6
        logger.info(f"Existing ZeRO Context.\nModel Data CUDA Memory {model_data_cuda_mem_MB} MB", ranks=[0])
        sys_cuda_mem_MB = colo_device_memory_used(get_current_device()) / 1e6
        logger.info(f"System CUDA Memory Usage {sys_cuda_mem_MB} MB", ranks=[0])
        logger.info(f"Model Number Parameter {model_numel_tensor.numpy()[0]/1e6} M", ranks=[0])


def run_dist(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_model_test()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 4])
@rerun_if_address_is_in_use()
def test_zero_init_context(world_size):
    spawn(run_dist, world_size)


if __name__ == '__main__':
    test_zero_init_context(1)
