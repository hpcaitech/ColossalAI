import pytest
import torch

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
from tests.test_zero.test_legacy.common import CONFIG, check_grads_padding, run_fwd_bwd


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
