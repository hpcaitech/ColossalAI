import random
from contextlib import nullcontext
from typing import Any, Callable, Iterator, List, Optional, Tuple

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.logging import disable_existing_loggers
from colossalai.pipeline.schedule import OneForwardOneBackwardSchedule
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.testing import (
    assert_hf_output_close,
    clear_cache_before_run,
    parameterize,
    rerun_if_address_is_in_use,
    spawn,
)
from tests.kit.model_zoo import model_zoo
from tests.test_shardformer.test_model._utils import build_model, build_pipeline_model, run_forward

DP_AXIS, PP_AXIS, TP_AXIS = 0, 1, 2


class PipelineOptimizer(OptimizerWrapper):

    def __init__(self, optim: Optimizer, model: Module):
        super().__init__(optim)
        params = set(model.parameters())
        new_param_groups = []
        for group in optim.param_groups:
            params = [p for p in group['params'] if p in params]
            new_param_groups.append({**group, 'params': params})
        optim.__setstate__({'param_groups': new_param_groups})
        # TODO: support amp


class PipelinedModel(ModelWrapper):

    def __init__(self, module: Module, shard_config: ShardConfig, stage_manager: PipelineStageManager) -> None:
        self.stage_manager = stage_manager
        shardformer = ShardFormer(shard_config)
        module, self.shared_params = shardformer.optimize(module)
        self.shared_param_process_groups = []
        super().__init__(module)


def prepare_dataloader(dataset, batch_size, shuffle=False, seed=1024, drop_last=False, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(
        dataset,
    #rank=self.pg_mesh.coordinate(DP_AXIS),
        shuffle=shuffle)

    # Deterministic dataloader
    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        worker_init_fn=seed_worker,
        drop_last=drop_last,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )


def execute_pipeline(
    data_iter: Iterator,
    model: PipelinedModel,
    criterion: Callable[[Any, Any], torch.Tensor],
    optimizer: PipelineOptimizer,
    return_loss: bool = True,
    return_outputs: bool = False,
    schedule: OneForwardOneBackwardSchedule = None,
) -> dict:
    # return loss or outputs if needed
    outputs = schedule.forward_backward_step(model, optimizer, data_iter, criterion, return_loss, return_outputs)
    return outputs


class data_iter():

    def __getitem__(self, x):
        return torch.randint(0, 100, (4, 128)).cuda()


def loss(x, y):
    return (x[0].float().mean() - y[0].float().mean())


@parameterize('enable_fused_normalization', [False])
@parameterize('enable_tensor_parallelism', [False])
@parameterize('use_lazy_init', [False])
def run_llama_test(enable_fused_normalization, enable_tensor_parallelism, use_lazy_init):
    PP_DIM = 0
    PP_SIZE = 2
    RANK_TO_COORDINATE = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, 1),
    }
    PP_RANKS_IN_GROUP = {
        0: [0, 1],
        1: [0, 1],
        2: [2, 3],
        3: [2, 3],
    }

    pg_mesh = ProcessGroupMesh(PP_SIZE)
    stage_manager = PipelineStageManager(pg_mesh, PP_DIM)
    sub_model_zoo = model_zoo.get_sub_registry('transformers_llama')
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        num_microbatches = 2
        org_model = model_fn().cuda()
        optimizer = torch.optim.AdamW(org_model.parameters(), lr=1e-3)
        #dataloader=prepare_dataloader(dataset=dataset['train'],batch_size=4)
        schedule = OneForwardOneBackwardSchedule(num_microbatches, stage_manager)
        shard_config = ShardConfig(enable_fused_normalization=enable_fused_normalization,
                                   enable_tensor_parallelism=enable_tensor_parallelism,
                                   pipeline_stage_manager=stage_manager)
        pipelined_model = PipelinedModel(org_model, shard_config, stage_manager)
        pp_optimizer = PipelineOptimizer(optimizer, pipelined_model)
        data_it = iter(data_iter())
        results = execute_pipeline(data_it, pipelined_model, loss, pp_optimizer, schedule=schedule)
        if stage_manager.is_last_stage():
            assert results['loss'] is not None
        assert results['outputs'] is None
    torch.cuda.empty_cache()


def check_llama(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_llama_test()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_llama():
    spawn(check_llama, 2)


if __name__ == "__main__":
    test_llama()
