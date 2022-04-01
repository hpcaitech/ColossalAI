from functools import partial

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.testing import parameterize, rerun_on_exception
from colossalai.utils import free_port
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.zero.shard_utils import (BucketTensorShardStrategy, TensorShardStrategy)
from colossalai.zero.sharded_model import ShardedModelV2
from colossalai.zero.sharded_model._utils import cast_tensor_to_fp16
from colossalai.zero.sharded_model.utils import col_model_deepcopy
from tests.components_to_test.registry import non_distributed_component_funcs
from colossalai.engine.gradient_handler import MoeGradientHandler
from colossalai.context import MOE_CONTEXT
from colossalai.testing import assert_equal_in_group

from tests.test_zero_data_parallel.common import CONFIG, check_grads_padding, run_fwd_bwd
from tests.test_moe.test_moe_zero_init import MoeModel


@parameterize("enable_autocast", [False])
@parameterize("shard_strategy_class", [TensorShardStrategy, BucketTensorShardStrategy])
def run_model_test(enable_autocast, shard_strategy_class):
    shard_strategy = shard_strategy_class()

    get_components_func = non_distributed_component_funcs.get_callable('no_leaf_module')
    _, train_dataloader, _, _, criterion = get_components_func()

    rm_torch_payload_on_the_fly = False

    with ZeroInitContext(target_device=torch.cuda.current_device(),
                         shard_strategy=shard_strategy,
                         shard_param=True,
                         rm_torch_payload_on_the_fly=rm_torch_payload_on_the_fly):
        zero_model = MoeModel()
    zero_model = ShardedModelV2(zero_model, shard_strategy, use_memory_tracer=True)

    # check whether parameters are identical in ddp
    for name, p in zero_model.named_parameters():
        if not p.colo_attr.param_is_sharded and p.is_replicated:
            assert_equal_in_group(p.data)

    model = MoeModel().half()
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
    MOE_CONTEXT.reset_loss()
    run_model_test()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [2])
@rerun_on_exception(exception_type=mp.ProcessRaisedException, pattern=".*Address already in use.*")
def test_moe_zero_model(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_moe_zero_model(world_size=2)
