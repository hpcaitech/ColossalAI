import copy
import math
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch import distributed as dist
from torch.distributed import ProcessGroup
from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.testing import assert_close

from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.booster.plugin.hybrid_parallel_plugin import HybridParallelModule
from colossalai.lazy import LazyInitContext
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer._utils import getattr_
from colossalai.shardformer.policies.auto_policy import Policy
from colossalai.tensor.d_tensor.api import is_customized_distributed_tensor, is_distributed_tensor


def print_rank(prompt, value, rank=0):
    if dist.get_rank() == rank:
        print(f"rank-{rank}, {prompt}: {value}")


def build_model(
    model_fn,
    enable_fused_normalization=True,
    enable_tensor_parallelism=True,
    enable_flash_attention=False,
    enable_jit_fused=False,
    enable_sequence_parallelism=False,
    use_lazy_init: bool = False,
):
    # create new model
    ctx = LazyInitContext() if use_lazy_init else nullcontext()
    with ctx:
        # create new model
        org_model = model_fn()
        model_copy = copy.deepcopy(org_model)
    if use_lazy_init:
        ctx.materialize(org_model)
    # shard model
    shard_config = ShardConfig(
        enable_fused_normalization=enable_fused_normalization,
        enable_tensor_parallelism=enable_tensor_parallelism,
        enable_flash_attention=enable_flash_attention,
        enable_jit_fused=enable_jit_fused,
        enable_sequence_parallelism=enable_sequence_parallelism,
    )
    model_copy = copy.deepcopy(org_model)
    shard_former = ShardFormer(shard_config=shard_config)
    sharded_model, shared_params = shard_former.optimize(model_copy)
    return org_model.cuda(), sharded_model.cuda()


def build_pipeline_model(
    model_fn,
    stage_manager=None,
    enable_fused_normalization=False,
    enable_tensor_parallelism=False,
    use_lazy_init: bool = False,
    policy: Optional[Policy] = None,
):
    ctx = LazyInitContext() if use_lazy_init else nullcontext()
    with ctx:
        # create new model
        org_model = model_fn()
        model_copy = copy.deepcopy(org_model)
    if use_lazy_init:
        ctx.materialize(org_model)

    # shard model
    shard_config = ShardConfig(
        enable_fused_normalization=enable_fused_normalization,
        enable_tensor_parallelism=enable_tensor_parallelism,
        pipeline_stage_manager=stage_manager,
    )

    shard_former = ShardFormer(shard_config=shard_config)
    sharded_model, shared_params = shard_former.optimize(model_copy, policy=policy)
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


def check_state_dict(org_model: Module, sharded_model: Module, name: str = ""):
    org_sd = org_model.state_dict()
    shard_sd = sharded_model.state_dict()
    for k, v in org_sd.items():
        assert k in shard_sd, f"{name} {k} not in sharded model"
        shard_v = shard_sd[k]
        assert v.shape == shard_v.shape, f"{name} {k} shape mismatch, {v.shape} vs {shard_v.shape}"
        assert v.dtype == shard_v.dtype, f"{name} {k} dtype mismatch, {v.dtype} vs {shard_v.dtype}"
        assert torch.equal(v, shard_v), f"{name} {k} value mismatch"


def build_model_from_hybrid_plugin(model_fn: Callable, loss_fn: Callable, test_config: Dict[str, Any]):
    use_lazy_init = False
    if "use_lazy_init" in test_config:
        use_lazy_init = test_config.pop("use_lazy_init")

    ctx = LazyInitContext() if use_lazy_init else nullcontext()
    with ctx:
        org_model = model_fn()
        sharded_model = copy.deepcopy(org_model)
    if use_lazy_init:
        ctx.materialize(org_model)
    org_model = org_model.cuda()
    org_optimizer = Adam(org_model.parameters(), lr=1e-3)
    sharded_optimizer = Adam(sharded_model.parameters(), lr=1e-3)
    criterion = loss_fn

    plugin = HybridParallelPlugin(**test_config)
    booster = Booster(plugin=plugin)

    sharded_model, sharded_optimizer, criterion, _, _ = booster.boost(sharded_model, sharded_optimizer, criterion)
    return org_model, org_optimizer, sharded_model, sharded_optimizer, criterion, booster


def run_forward_backward_with_hybrid_plugin(
    org_model: Module,
    sharded_model: Module,
    sharded_optimizer: Optimizer,
    data_gen_fn: Callable,
    output_transform_fn: Callable,
    criterion: Callable,
    booster: Booster,
):
    org_model.cuda()
    sharded_model.cuda()

    def _criterion(outputs, inputs):
        outputs = output_transform_fn(outputs)
        loss = criterion(outputs)
        return loss

    data = data_gen_fn()

    if (
        booster.plugin.shard_config.enable_sequence_parallelism
        and booster.plugin.shard_config.sequence_parallelism_mode in ["1", "2"]
        and booster.plugin.tp_size != 0
    ):
        seq_len = data["input_ids"].shape[-1]
        lcm = booster.plugin.tp_size * seq_len // math.gcd(booster.plugin.tp_size, seq_len)
        times = lcm // seq_len
        input_shape = data["input_ids"].shape
        for k, v in data.items():
            if v.shape == input_shape:
                data[k] = v.repeat((1,) * (v.dim() - 1) + (times,))

    shard_test_data = {}
    for k, v in data.items():
        if k == "labels":
            shard_test_data[k] = data[k].clone()
        else:
            shard_test_data[k] = (
                torch.chunk(data[k].clone(), booster.plugin.shard_config.sequence_parallel_size, dim=1)[dist.get_rank()]
                if booster.plugin.shard_config.enable_sequence_parallelism
                and booster.plugin.shard_config.sequence_parallelism_mode in ["2", "3"]
                else data[k].clone()
            )
    unshard_test_data = {}
    for k, v in data.items():
        unshard_test_data[k] = data[k].clone()

    sharded_model.train()
    if booster.plugin.stage_manager is not None:
        for k, v in shard_test_data.items():
            if torch.is_tensor(v) or "Tensor" in v.__class__.__name__:
                new_shape = [1] * v.dim()
                new_shape[0] = 4
                shard_test_data[k] = v.to("cuda").repeat(*new_shape)

        data_iter = iter([shard_test_data])
        sharded_output = booster.execute_pipeline(
            data_iter, sharded_model, _criterion, sharded_optimizer, return_loss=True, return_outputs=True
        )
        sharded_loss = sharded_output["loss"]

    else:
        shard_test_data = {k: v.cuda() for k, v in shard_test_data.items()}
        sharded_output = sharded_model(**shard_test_data)
        sharded_loss = criterion(sharded_output)
        sharded_optimizer.backward(sharded_loss)

    org_model.train()
    if booster.plugin.stage_manager is not None:
        for k, v in unshard_test_data.items():
            if torch.is_tensor(v) or "Tensor" in v.__class__.__name__:
                new_shape = [1] * v.dim()
                new_shape[0] = 4
                unshard_test_data[k] = v.to("cuda").repeat(*new_shape)
    unshard_test_data = {k: v.cuda() for k, v in unshard_test_data.items()}
    org_output = org_model(**unshard_test_data)
    org_loss = criterion(org_output)
    org_loss.backward()

    return org_loss, org_output, sharded_loss, sharded_output


def check_output_hidden_state(
    org_output: Tensor,
    sharded_output: Tensor,
    stage_manager: Optional[PipelineStageManager] = None,
    atol: float = 1e-5,
    rtol: float = 1e-3,
):
    org_hidden_state = org_output.last_hidden_state

    if stage_manager and stage_manager.is_last_stage(ignore_chunk=True):
        sharded_hidden_state = sharded_output["outputs"]["last_hidden_state"]
    else:
        sharded_hidden_state = sharded_output.last_hidden_state

    assert_close(org_hidden_state.float(), sharded_hidden_state.float(), atol=atol, rtol=rtol)


def check_loss(org_loss: Tensor, sharded_loss: Tensor, atol: float = 1e-5, rtol: float = 1e-3):
    assert torch.allclose(org_loss.float(), sharded_loss.float(), atol=atol, rtol=rtol)


def check_weight(
    org_model: Module,
    sharded_model: Module,
    layer_suffix: List[str],
    tp_group: Optional[ProcessGroup] = None,
    dim: int = 0,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    verbose: bool = False,
):
    for suffix in layer_suffix:
        org_weight = getattr_(org_model, suffix).weight
        sharded_weight = getattr_(sharded_model, suffix).weight

        # skip if layer is not held by this process
        if sharded_weight is None:
            continue

        if is_distributed_tensor(sharded_weight) or is_customized_distributed_tensor(sharded_weight):
            sharded_weight_list = [
                torch.zeros_like(sharded_weight).to("cuda") for _ in range(dist.get_world_size(tp_group))
            ]
            dist.all_gather(sharded_weight_list, sharded_weight, tp_group)
            sharded_weight = torch.cat(sharded_weight_list, dim=dim)

        if verbose and dist.get_rank() == 0:
            print(f"'{suffix}' weight: {org_weight}, {sharded_weight}")

        assert_close(org_weight.float(), sharded_weight.float(), atol=atol, rtol=rtol)


def get_grad_tensors_for_check(
    org_model: Module,
    sharded_model: Module,
    layer_suffix: List[str],
    tp_group: ProcessGroup = None,
    dim: int = 0,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    verbose: bool = False,
    name: str = None,
):
    grad_to_check = {}
    for suffix in layer_suffix:
        org_grad = getattr_(org_model, suffix).weight.grad
        shard_grad = getattr_(sharded_model, suffix).weight.grad
        shard_weight = getattr_(sharded_model, suffix).weight
        if is_distributed_tensor(shard_weight) or is_customized_distributed_tensor(shard_weight):
            shard_grad_list = [torch.zeros_like(shard_grad).to("cuda") for _ in range(dist.get_world_size(tp_group))]
            dist.all_gather(shard_grad_list, shard_grad, tp_group)
            shard_grad = torch.cat(shard_grad_list, dim=dim)

        # embedding may be resized when using tensor parallel
        if shard_grad.shape[0] > org_grad.shape[0]:
            shard_grad = shard_grad[: org_grad.shape[0], :]
        if verbose and dist.get_rank() == 0:
            print(f"'{suffix}' grad: {org_grad}, {shard_grad}")

        grad_to_check[suffix] = {
            "org_grad": org_grad.float(),
            "shard_grad": shard_grad.float(),
            "rtol": rtol,
            "atol": atol,
        }

    return grad_to_check


# used by sam/blip2
def check_grad(
    org_model: Module,
    sharded_model: Module,
    layer_suffix: List[str],
    tp_group: ProcessGroup = None,
    dim: int = 0,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    verbose: bool = False,
):
    for suffix in layer_suffix:
        org_grad = getattr_(org_model, suffix).weight.grad
        shard_grad = getattr_(sharded_model, suffix).weight.grad
        shard_weight = getattr_(sharded_model, suffix).weight
        if is_distributed_tensor(shard_weight) or is_customized_distributed_tensor(shard_weight):
            shard_grad_list = [torch.zeros_like(shard_grad).to("cuda") for _ in range(dist.get_world_size(tp_group))]
            dist.all_gather(shard_grad_list, shard_grad, tp_group)
            shard_grad = torch.cat(shard_grad_list, dim=dim)

        # embedding may be resized when using tensor parallel
        if shard_grad.shape[0] > org_grad.shape[0]:
            shard_grad = shard_grad[: org_grad.shape[0], :]
        if verbose and dist.get_rank() == 0:
            print(f"'{suffix}' grad: {org_grad}, {shard_grad}")

        assert_close(org_grad.float(), shard_grad.float(), rtol=rtol, atol=atol)


def unwrap_model(
    module: Module, base_model_class_name: Optional[str] = None, base_model_attribute_name: Optional[str] = None
):
    if isinstance(module, HybridParallelModule):
        module = module.unwrap()
    if base_model_class_name is None:
        return module
    if module.__class__.__name__ == base_model_class_name:
        return module
    return getattr(module, base_model_attribute_name, None)


def check_all_grad_tensors(check_tensors):
    """
    "org_grad": tensor to be compared from the original model
    "shard_grad": tensor to be compared from the sharded model
    """
    for suffix, check_info in check_tensors.items():
        org_grad = check_info["org_grad"]
        shard_grad = check_info["shard_grad"]
        rtol = check_info["rtol"]
        atol = check_info["atol"]
        assert_close(org_grad, shard_grad, atol=atol, rtol=rtol)
