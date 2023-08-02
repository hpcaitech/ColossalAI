import copy
from contextlib import nullcontext

import torch
from torch import distributed as dist
from torch.distributed import ProcessGroup
from torch.nn import Module
from torch.optim import Adam, Optimizer

from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.lazy import LazyInitContext
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.tensor.d_tensor.api import is_customized_distributed_tensor, is_distributed_tensor


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


def build_model_from_hybrid_plugin(model_fn: callable, loss_fn: callable, test_config: dict):

    use_lazy_init = False
    if 'use_lazy_init' in test_config:
        use_lazy_init = test_config.pop('use_lazy_init')

    if use_lazy_init:
        ctx = LazyInitContext()
    else:
        ctx = nullcontext()

    plugin = HybridParallelPlugin(**test_config)
    booster = Booster(plugin=plugin)

    with ctx:
        org_model = model_fn().cuda()
        sharded_model = copy.deepcopy(org_model).cuda()
        # print('inctx',sharded_model.embeddings.word_embeddings.weight.device)
        # print(use_lazy_init)
    if use_lazy_init:
        org_model = ctx.materialize(org_model)

    org_optimizer = Adam(org_model.parameters(), lr=1e-3)
    sharded_optimizer = Adam(sharded_model.parameters(), lr=1e-3)
    criterion = loss_fn

    sharded_model, sharded_optimizer, criterion, _, _ = booster.boost(sharded_model, sharded_optimizer, criterion)

    return org_model, org_optimizer, sharded_model, sharded_optimizer, criterion, booster


def run_forward_backward_with_hybrid_plugin(org_model: Module, sharded_model: Module, sharded_optimizer: Optimizer,
                                            data_gen_fn: callable, output_transform_fn: callable, criterion: callable,
                                            booster: Booster):
    org_model.cuda()
    sharded_model.cuda()

    def _criterion(outputs, inputs):
        outputs = output_transform_fn(outputs)
        loss = criterion(outputs)
        return loss

    data = data_gen_fn()
    sharded_model.train()
    if booster.plugin.stage_manager is not None:
        data = {
            k: v.to('cuda').repeat(4, 1) if torch.is_tensor(v) or 'Tensor' in v.__class__.__name__ else v
            for k, v in data.items()
        }
        data_iter = iter([data])
        sharded_output = booster.execute_pipeline(data_iter,
                                                  sharded_model,
                                                  _criterion,
                                                  sharded_optimizer,
                                                  return_loss=True,
                                                  return_outputs=True)
        sharded_loss = sharded_output['loss']
    else:
        data = {k: v.cuda() for k, v in data.items()}
        sharded_output = sharded_model(**data)
        sharded_loss = criterion(sharded_output)
        sharded_loss.backward()

    org_model.train()
    org_output = org_model(**data)
    org_loss = criterion(org_output)
    org_loss.backward()

    return org_loss, org_output, sharded_loss, sharded_output


def check_output_hidden_state(org_output, sharded_output, stage_manager=None, atol=1e-5, rtol=1e-3):

    org_hidden_state = org_output.last_hidden_state

    if stage_manager is None:
        sharded_hidden_state = sharded_output.last_hidden_state

    if stage_manager and stage_manager.is_last_stage():
        sharded_hidden_state = torch.cat([output.last_hidden_state for output in sharded_output['outputs']], dim=0)

    assert torch.allclose(org_hidden_state, sharded_hidden_state, atol=atol, rtol=rtol), \
        f"shard model's output hidden state is not equal to origin model's last hidden state\n{org_hidden_state}\n{sharded_hidden_state}"


def check_loss(org_loss, sharded_loss, atol=1e-5, rtol=1e-3):
    assert torch.allclose(org_loss, sharded_loss, atol=atol, rtol=rtol), \
        f"shard model loss is not equal to origin model loss\n{org_loss}\n{sharded_loss}"


def check_weight(org_param: Module, sharded_param: Module, tp_group: ProcessGroup = None, atol=1e-5, rtol=1e-3):

    org_weight = org_param.weight
    sharded_weight = sharded_param.weight

    if is_distributed_tensor(sharded_weight) or is_customized_distributed_tensor(sharded_weight):
        sharded_weight_list = [
            torch.zeros([*sharded_weight.shape]).to('cuda') for _ in range(dist.get_world_size(tp_group))
        ]
        dist.all_gather(sharded_weight_list, sharded_weight, tp_group)
        sharded_weight = torch.cat(sharded_weight_list, dim=0)

    assert torch.allclose(org_weight, sharded_weight, atol=atol, rtol=rtol), \
        f"shard model weight is not equal to origin model weight\n{org_weight}\n{sharded_weight}"


def check_gradient(org_param: Module, sharded_param: Module, tp_group: ProcessGroup = None, atol=1e-5, rtol=1e-3):

    sharded_weight = sharded_param.weight
    org_grad = org_param.weight.grad
    sharded_grad = sharded_param.weight.grad
    print(org_grad.shape)
    if is_distributed_tensor(sharded_weight) or is_customized_distributed_tensor(sharded_weight):
        sharded_grad_list = [
            torch.zeros([*sharded_grad.shape]).to('cuda') for _ in range(dist.get_world_size(tp_group))
        ]
        dist.all_gather(sharded_grad_list, sharded_grad, tp_group)
        sharded_grad = torch.cat(sharded_grad_list, dim=0)
        print(sharded_grad.shape)

    assert torch.allclose(org_grad, sharded_grad, atol=atol, rtol=rtol), \
        f"shard model grad is not equal to origin model grad\n{org_grad}\n{sharded_grad}"
