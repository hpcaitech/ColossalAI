import pytest
import torch
from torch.testing import assert_close

import colossalai
from colossalai.nn.optimizer.came import CAME
from colossalai.nn.optimizer.distributed_came import DistributedCAME
from colossalai.shardformer.layer.utils import Randomizer
from colossalai.tensor.d_tensor.api import clear_layout_converter
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo
from tests.test_shardformer.test_model._utils import (
    build_model_from_hybrid_plugin,
    check_weight,
    run_forward_backward_with_hybrid_plugin,
    unwrap_model,
)


def check_bert_fwd_bwd(
    model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config, optim_class, sharded_optim_class
):
    org_model, org_optimizer, sharded_model, sharded_optimizer, criterion, booster = build_model_from_hybrid_plugin(
        model_fn, loss_fn, test_config, optim_class, sharded_optim_class
    )
    org_optimizer.zero_grad()
    sharded_optimizer.zero_grad()
    org_loss, org_output, sharded_loss, sharded_output = run_forward_backward_with_hybrid_plugin(
        org_model, sharded_model, sharded_optimizer, data_gen_fn, output_transform_fn, criterion, booster
    )

    stage_manager = booster.plugin.stage_manager
    tp_group = booster.plugin.tp_group

    # assign grad
    # for (name1, ori_param), (name2, dist_param) in zip(org_model.named_parameters(), sharded_model.named_parameters()):
    #     if ori_param.requires_grad:
    #         try:
    #             sharding_spec = api.get_sharding_spec(dist_param)
    #             clip_dim = 0 if 0 in sharding_spec.dim_partition_dict.keys() else 1
    #         except:
    #             clip_dim = -1
    #         # 为原始参数生成随机梯度
    #         random_grad = torch.randn_like(dist_param.data)
    #         dist_param.grad = random_grad.clone()

    #         # 根据进程排名和总进程数分片梯度
    #         if clip_dim == -1:
    #             grad_chunk = random_grad
    #         else:
    #             shard_grad_list = [torch.zeros_like(random_grad).to("cuda") for _ in range(dist.get_world_size(tp_group))]
    #             dist.all_gather(shard_grad_list, random_grad, tp_group)
    #             full_grad = torch.cat(shard_grad_list, dim=clip_dim)
    #         if dist.get_rank() == 0:
    #             print(dist.get_rank(), full_grad.data.shape, random_grad.shape)
    #         try:
    #             ori_param.grad = full_grad.clone()
    #         except:
    #             ori_param.grad = torch.zeros_like(ori_param.data)

    bert = unwrap_model(org_model, "BertModel", "bert")
    sharded_bert = unwrap_model(sharded_model, "BertModel", "bert")
    weight_layer_for_check = [
        "encoder.layer[0].output.dense",
        "encoder.layer[1].output.dense",
    ]

    # check grad
    # check_grad(bert, sharded_bert, ["encoder.layer[0].output.dense"], tp_group, atol=1e-4, rtol=1e-4, dim=1)

    # optimizer executes step
    org_optimizer.step()
    sharded_optimizer.step()

    # check weights
    if test_config["precision"] == "bf16":
        atol, rtol = 5e-4, 1e-4
    else:
        atol, rtol = 5e-4, 5e-4
    if stage_manager is None or stage_manager.is_first_stage(ignore_chunk=True):
        check_weight(bert, sharded_bert, weight_layer_for_check, tp_group, atol=atol, rtol=rtol, dim=1)
    # check optim states
    for group in org_optimizer.param_groups:
        for p in group["params"]:
            sharded_state = sharded_optimizer.optim.state[p]
            state = org_optimizer.state[p]
            for key in sharded_state:
                assert_close(state[key], sharded_state[key], rtol=1e-5, atol=1e-5)
    torch.cuda.empty_cache()


@parameterize(
    "test_config",
    [
        # {
        #     "tp_size": 1,
        #     "num_microbatches": 4,
        #     "zero_stage": 2,
        #     "precision": "bf16",
        # },
        # {
        #     "tp_size": 2,
        #     "num_microbatches": 4,
        #     "zero_stage": 2,
        #     "precision": "bf16",
        # },
        # {
        #     "tp_size": 4,
        #     "num_microbatches": 4,
        #     "zero_stage": 2,
        #     "precision": "bf16",
        # },
        # {
        #     "tp_size": 1,
        #     "num_microbatches": 4,
        #     "zero_stage": 2,
        #     "precision": "fp16",
        # },
        # {
        #     "tp_size": 2,
        #     "num_microbatches": 4,
        #     "zero_stage": 2,
        #     "precision": "fp16",
        # },
        {
            "tp_size": 4,
            "num_microbatches": 0,
            "zero_stage": 0,
            "precision": "fp16",
        },
    ],
)
def run_bert_test(test_config, optim_class, sharded_optim_class):
    """Just call this if you've initialized distributed backend and spawned procs"""
    sub_model_zoo = model_zoo.get_sub_registry("transformers_bert")
    test_config["use_lazy_init"] = False
    test_config["pp_size"] = 1  # Do NOT test Pipeline Parallel
    test_config["initial_scale"] = 2**15  # avoid overflow

    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        check_bert_fwd_bwd(
            model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config, optim_class, sharded_optim_class
        )
        break

    clear_layout_converter()
    Randomizer.reset_index()
    torch.cuda.empty_cache()


def _run_bert_test(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_bert_test(optim_class=CAME, sharded_optim_class=DistributedCAME)


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def check_optim_on_bert():
    spawn(_run_bert_test, 4)


if __name__ == "__main__":
    check_optim_on_bert()
