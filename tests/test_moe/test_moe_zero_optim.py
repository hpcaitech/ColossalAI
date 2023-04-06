import pytest
import torch

import colossalai
from colossalai.amp import convert_to_apex_amp
from colossalai.context import MOE_CONTEXT
from colossalai.engine.gradient_handler import MoeGradientHandler
from colossalai.nn import MoeLoss
from colossalai.nn.optimizer import CPUAdam
from colossalai.testing import assert_equal_in_group, parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils import get_current_device
from colossalai.zero.legacy.init_ctx import ZeroInitContext
from colossalai.zero.legacy.shard_utils import BucketTensorShardStrategy, TensorShardStrategy
from colossalai.zero.legacy.sharded_model import ShardedModelV2
from colossalai.zero.legacy.sharded_model.utils import col_model_deepcopy
from colossalai.zero.legacy.sharded_optim import ShardedOptimizerV2
from colossalai.zero.low_level._utils import has_inf_or_nan
from tests.components_to_test.registry import non_distributed_component_funcs
from tests.test_moe.test_moe_zero_init import MoeModel
from tests.test_zero.test_legacy.common import CONFIG, check_sharded_model_params


def _run_step(model, optimizer, data, label, criterion, grad_handler):
    model.train()
    optimizer.zero_grad()

    if criterion:
        y = model(data)
        loss = criterion(y, label)
    else:
        loss = model(data, label)

    loss = loss.float()
    if isinstance(model, ShardedModelV2):
        optimizer.backward(loss)
    else:
        loss.backward()

    if grad_handler is not None:
        grad_handler.handle_gradient()

    optimizer.step()


@parameterize("cpu_offload", [True])
@parameterize("use_cpuadam", [True])    # We do not use Hybrid Adam right now, since it has a little bug
@parameterize("reuse_fp16_shard", [True, False])
@parameterize("shard_strategy_class", [TensorShardStrategy, BucketTensorShardStrategy])
def _run_test_sharded_optim_v2(cpu_offload,
                               shard_strategy_class,
                               use_cpuadam,
                               reuse_fp16_shard,
                               gpu_margin_mem_ratio=0.0):
    shard_strategy = shard_strategy_class()
    if use_cpuadam and cpu_offload is False:
        return
    MOE_CONTEXT.reset_loss()
    get_components_func = non_distributed_component_funcs.get_callable('hanging_param_model')
    _, train_dataloader, _, optimizer_class, _ = get_components_func()
    criterion = MoeLoss(aux_weight=0.01, loss_fn=torch.nn.CrossEntropyLoss)

    with ZeroInitContext(target_device=torch.device('cpu') if cpu_offload else get_current_device(),
                         shard_strategy=shard_strategy,
                         shard_param=True):
        zero_model = MoeModel(checkpoint=True)

    zero_model = ShardedModelV2(zero_model,
                                shard_strategy,
                                tensor_placement_policy='cpu' if cpu_offload else 'cuda',
                                reuse_fp16_shard=reuse_fp16_shard)

    # check whether parameters are identical in ddp
    for name, p in zero_model.named_parameters():
        if not p.colo_attr.param_is_sharded and p.colo_attr.is_replicated:
            assert_equal_in_group(p.colo_attr.data_payload.to(get_current_device()))

    model = MoeModel(checkpoint=True).half()
    col_model_deepcopy(zero_model, model)
    model = model.cuda().float()

    if use_cpuadam:
        optimizer_class = CPUAdam
    optim = optimizer_class(model.parameters(), lr=1e-3)
    sharded_optim = optimizer_class(zero_model.parameters(), lr=1e-3)
    sharded_optim = ShardedOptimizerV2(zero_model,
                                       sharded_optim,
                                       initial_scale=2**5,
                                       gpu_margin_mem_ratio=gpu_margin_mem_ratio)

    amp_config = dict(opt_level='O2', keep_batchnorm_fp32=False)
    apex_model, apex_optimizer = convert_to_apex_amp(model, optim, amp_config)
    apex_grad_handler = MoeGradientHandler(model)

    for i, (data, label) in enumerate(train_dataloader):
        if i > 5:
            break
        data, label = data.cuda(), label.cuda()
        _run_step(apex_model, apex_optimizer, data, label, criterion, apex_grad_handler)
        _run_step(zero_model, sharded_optim, data, label, criterion, None)
        check_sharded_model_params(model, zero_model, loose=True, reuse_fp16_shard=use_cpuadam)
        for param in model.parameters():
            assert not has_inf_or_nan(param)


def _run_dist(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    MOE_CONTEXT.setup(seed=42)
    _run_test_sharded_optim_v2()


# use_cpuadam = True can be used with cpu_offload = False
@pytest.mark.dist
@pytest.mark.parametrize("world_size", [2])
@rerun_if_address_is_in_use()
def test_moe_zero_optim(world_size):
    spawn(_run_dist, world_size)


if __name__ == '__main__':
    test_moe_zero_optim(world_size=4)
