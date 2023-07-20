import copy
from contextlib import nullcontext

import torch
from torch.nn import Module

from colossalai.lazy import LazyInitContext
from colossalai.shardformer import ShardConfig, ShardFormer


def build_model(model_fn, enable_fused_normalization=True, enable_tensor_parallelism=True, use_lazy_init: bool = False):
    ctx = LazyInitContext() if use_lazy_init else nullcontext()
    with ctx:
        # create new model
        org_model = model_fn()
        model_copy = copy.deepcopy(org_model)
    if use_lazy_init:
        ctx.materialize(org_model)
    # shard model
    shard_config = ShardConfig(enable_fused_normalization=enable_fused_normalization,
                               enable_tensor_parallelism=enable_tensor_parallelism)
    shard_former = ShardFormer(shard_config=shard_config)
    sharded_model, shared_params = shard_former.optimize(model_copy)
    return org_model.cuda(), sharded_model.cuda()


def build_pipeline_model(model_fn,
                         stage_manager=None,
                         enable_fused_normalization=False,
                         enable_tensor_parallelism=False,
                         use_lazy_init: bool = False):
    ctx = LazyInitContext() if use_lazy_init else nullcontext()
    with ctx:
        # create new model
        org_model = model_fn()
        model_copy = copy.deepcopy(org_model)
    if use_lazy_init:
        ctx.materialize(org_model)

    # shard model
    shard_config = ShardConfig(enable_fused_normalization=enable_fused_normalization,
                               enable_tensor_parallelism=enable_tensor_parallelism,
                               pipeline_stage_manager=stage_manager)

    shard_former = ShardFormer(shard_config=shard_config)
    sharded_model, shared_params = shard_former.optimize(model_copy)
    return org_model.cuda(), sharded_model.cuda()


def run_forward(original_model, sharded_model, data_gen_fn, output_transform_fn, loss_fn):
    # prepare input
    data = data_gen_fn()
    data = {k: v.cuda() for k, v in data.items()}
    # switch to train mode
    original_model.train()
    sharded_model.train()
    # run forward
    org_output = original_model(**data)
    org_output = output_transform_fn(org_output)
    org_loss = loss_fn(org_output)

    shard_output = sharded_model(**data)
    shard_output = output_transform_fn(shard_output)
    shard_loss = loss_fn(shard_output)
    return org_output, org_loss, shard_output, shard_loss


def check_state_dict(org_model: Module, sharded_model: Module, name: str = ''):
    org_sd = org_model.state_dict()
    shard_sd = sharded_model.state_dict()
    for k, v in org_sd.items():
        assert k in shard_sd, f'{name} {k} not in sharded model'
        shard_v = shard_sd[k]
        assert v.shape == shard_v.shape, f'{name} {k} shape mismatch, {v.shape} vs {shard_v.shape}'
        assert v.dtype == shard_v.dtype, f'{name} {k} dtype mismatch, {v.dtype} vs {shard_v.dtype}'
        assert torch.equal(v, shard_v), f'{name} {k} value mismatch'
