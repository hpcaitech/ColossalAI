from typing import Tuple

import pytest
import torch
import torch.distributed
import torch.distributed as dist
from torch.testing import assert_close

import colossalai
from colossalai.booster.plugin import MoeHybridParallelPlugin
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.testing.random import seed_all
from tests.kit.model_zoo.transformers.deepseek_v3 import (
    data_gen_for_lm,
    init_deepseek,
    loss_fn_for_lm,
    output_transform_fn,
)
from tests.test_shardformer.test_model._utils import (
    build_model_from_hybrid_plugin,
    run_forward_backward_with_hybrid_plugin,
)


def check_forward_backward(model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config):
    enable_gradient_checkpointing = test_config.pop("enable_gradient_checkpointing", False)
    seed_all(42)
    org_model, org_optimizer, sharded_model, sharded_optimizer, criterion, booster = build_model_from_hybrid_plugin(
        model_fn, loss_fn, test_config, pluggin_cls=MoeHybridParallelPlugin
    )
    if enable_gradient_checkpointing:
        # org_model.gradient_checkpointing_enable()
        sharded_model.unwrap().gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    org_model = org_model.to(torch.bfloat16)
    org_model.eval()
    sharded_model.eval()

    org_loss, org_output, sharded_loss, sharded_output = run_forward_backward_with_hybrid_plugin(
        org_model, sharded_model, sharded_optimizer, data_gen_fn, output_transform_fn, criterion, booster
    )

    assert_close(org_loss, sharded_loss)

    param_dict = {n: p for n, p in org_model.named_parameters()}
    for n, p in sharded_model.unwrap().named_parameters():
        if n in param_dict:
            if booster.plugin.zero_stage == 0:
                grad = p.grad
                target_grad = param_dict[n].grad
            else:
                grad = sharded_optimizer.get_working_grad_by_param_id(id(p))
                pg = sharded_optimizer.param_to_pg[p]
                target_grad = param_dict[n].grad
                if target_grad is None:
                    continue
                target_grad = target_grad.view(-1).chunk(dist.get_world_size(pg))[dist.get_rank(pg)]
            assert_close(grad, target_grad, atol=5e-1, rtol=0)


@parameterize(
    "config",
    [
        # zero 1
        (1, 4),
        (1, 2),
    ],
)
def run_deepseek_v3_test(config: Tuple[int, ...]):
    zero_stage, ep_size = config
    plugin_config = dict(
        pp_size=1,
        tp_size=1,
        ep_size=ep_size,
        zero_stage=zero_stage,
        overlap_communication=False,
        precision="bf16",
        find_unused_parameters=True,
    )

    check_forward_backward(
        init_deepseek,
        data_gen_for_lm,
        output_transform_fn,
        loss_fn_for_lm,
        plugin_config,
    )


def check_deepseek_v3(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_deepseek_v3_test()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [4])
@rerun_if_address_is_in_use()
def test_deepseek_v3(world_size):
    spawn(check_deepseek_v3, world_size)


if __name__ == "__main__":
    test_deepseek_v3(world_size=4)
