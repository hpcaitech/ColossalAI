import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.context import MOE_CONTEXT
from colossalai.engine.gradient_handler import MoeGradientHandler
from colossalai.nn import MoeLoss
from colossalai.testing import assert_equal_in_group, parameterize, rerun_if_address_is_in_use, spawn
from colossalai.zero.legacy.init_ctx import ZeroInitContext
from colossalai.zero.legacy.shard_utils import BucketTensorShardStrategy, TensorShardStrategy
from colossalai.zero.legacy.sharded_model import ShardedModelV2
from colossalai.zero.legacy.sharded_model._utils import cast_tensor_to_fp16
from colossalai.zero.legacy.sharded_model.utils import col_model_deepcopy
from tests.components_to_test.registry import non_distributed_component_funcs
from tests.test_moe.test_moe_zero_init import MoeModel

def allclose(tensor_a: torch.Tensor, tensor_b: torch.Tensor, loose=False) -> bool:
    if loose:
        return torch.allclose(tensor_a, tensor_b, atol=1e-2, rtol=1e-3)
    return torch.allclose(tensor_a, tensor_b)

CONFIG = dict(fp16=dict(mode=None,),
              zero=dict(level=3,
                        verbose=False,
                        offload_optimizer_config=dict(device='cpu', pin_memory=True, buffer_count=5, fast_init=False),
                        offload_param_config=dict(device='cpu',
                                                  pin_memory=True,
                                                  buffer_count=5,
                                                  buffer_size=1e8,
                                                  max_in_cpu=1e9)),
              parallel=dict(pipeline=dict(size=1), tensor=dict(size=1, mode=None)))

def check_grads_padding(model, zero_model, loose=False):
    rank = dist.get_rank()
    for (name, p), (zero_name, zero_p) in zip(model.named_parameters(), zero_model.named_parameters()):
        # zero_grad = zero_p.grad.clone().to(p.device)
        if zero_p.colo_attr.is_replicated:
            zero_grad = zero_p.colo_attr.grad_payload.clone().to(p.device)
            chunks = torch.flatten(p.grad).chunk(dist.get_world_size())
            if rank >= len(chunks):
                continue
            grad = chunks[rank].float()
            if zero_grad.size(0) > grad.size(0):
                zero_grad = zero_grad[:grad.size(0)]
        else:
            zero_grad = zero_p.colo_attr.grad_payload
            grad = p.grad.to(zero_grad.dtype)

        assert grad.dtype == zero_grad.dtype
        assert allclose(grad, zero_grad, loose=loose), f'diff: {grad - zero_grad}'

def run_fwd_bwd(model, data, label, criterion, enable_autocast=False):
    model.train()
    with torch.cuda.amp.autocast(enabled=enable_autocast):
        if criterion:
            y = model(data)
            loss = criterion(y, label)
        else:
            loss = model(data, label)
        loss = loss.float()
    if isinstance(model, ShardedModelV2):
        model.backward(loss)
    else:
        loss.backward()


@parameterize("enable_autocast", [False])
@parameterize("shard_strategy_class", [TensorShardStrategy, BucketTensorShardStrategy])
def run_model_test(enable_autocast, shard_strategy_class):
    shard_strategy = shard_strategy_class()

    get_components_func = non_distributed_component_funcs.get_callable('hanging_param_model')
    _, train_dataloader, _, optimizer_class, _ = get_components_func()
    criterion = MoeLoss(aux_weight=0.01, loss_fn=torch.nn.CrossEntropyLoss)

    with ZeroInitContext(target_device=torch.device('cuda', torch.cuda.current_device()),
                         shard_strategy=shard_strategy,
                         shard_param=True):
        zero_model = MoeModel(checkpoint=True)
    zero_model = ShardedModelV2(zero_model, shard_strategy)

    # check whether parameters are identical in ddp
    for name, p in zero_model.named_parameters():
        if not p.colo_attr.param_is_sharded and p.colo_attr.is_replicated:
            assert_equal_in_group(p.colo_attr.data_payload)

    model = MoeModel(checkpoint=True).half()
    col_model_deepcopy(zero_model, model)
    model = model.cuda()
    grad_handler = MoeGradientHandler(model)

    for i, (data, label) in enumerate(train_dataloader):
        if i > 5:
            break

        data, label = cast_tensor_to_fp16(data).cuda(), label.cuda()
        run_fwd_bwd(model, data, label, criterion, enable_autocast)
        run_fwd_bwd(zero_model, data, label, criterion, enable_autocast)
        grad_handler.handle_gradient()

        check_grads_padding(model, zero_model, loose=True)


def run_dist(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    MOE_CONTEXT.setup(seed=42)
    run_model_test()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [2])
@rerun_if_address_is_in_use()
def test_moe_zero_model(world_size):
    spawn(run_dist, world_size)


if __name__ == '__main__':
    test_moe_zero_model(world_size=2)
