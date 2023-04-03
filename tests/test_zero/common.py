from functools import partial

import torch
import torch.distributed as dist

from colossalai.logging import get_dist_logger
from colossalai.utils import checkpoint
from colossalai.zero.legacy.shard_utils import TensorShardStrategy
from colossalai.zero.legacy.sharded_model import ShardedModelV2

LOGGER = get_dist_logger('zero_test')

MP_PARALLEL_CONFIG = dict(fp16=dict(mode=None,), parallel=dict(pipeline=dict(size=1), tensor=dict(size=2, mode=None)))

_ZERO_MODEL_CONFIG = dict(reduce_scatter_bucket_size_mb=25,
                          fp32_reduce_scatter=False,
                          tensor_placement_policy='cuda',
                          gradient_predivide_factor=1.0,
                          shard_strategy=TensorShardStrategy(),
                          reuse_fp16_shard=False)

_ZERO_OPTIMIZER_CONFIG = dict(initial_scale=2**5,
                              min_scale=1,
                              growth_factor=2,
                              backoff_factor=0.5,
                              growth_interval=1000,
                              hysteresis=2,
                              max_scale=2**32)

ZERO_PARALLEL_CONFIG = dict(fp16=dict(mode=None,),
                            zero=dict(
                                model_config=_ZERO_MODEL_CONFIG,
                                optimizer_config=_ZERO_OPTIMIZER_CONFIG,
                            ),
                            parallel=dict(pipeline=dict(size=1), tensor=dict(size=1, mode=None)))

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


def checkpoint_wrapper(module, enable=True):
    if enable:
        module.forward = partial(checkpoint, module.forward)
    return module


def allclose(tensor_a: torch.Tensor, tensor_b: torch.Tensor, loose=False) -> bool:
    if loose:
        return torch.allclose(tensor_a, tensor_b, atol=1e-2, rtol=1e-3)
    return torch.allclose(tensor_a, tensor_b)


def check_grads(model, zero_model, loose=False):
    for p, zero_p in zip(model.parameters(), zero_model.parameters()):
        zero_grad = zero_p.grad.clone().to(p.device)
        grad = p.grad.float()
        assert grad.dtype == zero_grad.dtype
        assert allclose(grad, zero_grad, loose=loose)


def check_params(model, zero_model, loose=False):
    for p, zero_p in zip(model.parameters(), zero_model.parameters()):
        zero_p = zero_p.clone().to(p.device)
        # assert p.dtype == zero_p.dtype
        assert allclose(p.float(), zero_p.float(), loose=loose), f"diff {p.float() - zero_p.float()}"


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


def check_params_padding(model, zero_model, loose=False):
    rank = dist.get_rank()
    for p, zero_p in zip(model.parameters(), zero_model.parameters()):
        zero_p = zero_p.clone().to(p.device)
        chunks = torch.flatten(p).chunk(dist.get_world_size())
        if rank >= len(chunks):
            continue
        p = chunks[rank]
        if zero_p.size(0) > p.size(0):
            zero_p = zero_p[:p.size(0)]
        assert p.dtype == zero_p.dtype
        assert allclose(p, zero_p, loose=loose)


def check_sharded_model_params(model, zero_model, loose=False, reuse_fp16_shard=False):
    rank = dist.get_rank()
    for (name, p), (zero_name, zero_p) in zip(model.named_parameters(), zero_model.named_parameters()):
        if zero_p.colo_attr.param_is_sharded:
            zero_p = zero_p.colo_attr.data_payload.to(p.device).float()
            chunks = torch.flatten(p).chunk(dist.get_world_size())
            if rank >= len(chunks):
                continue
            p = chunks[rank].float()
            if zero_p.size(0) > p.size(0):
                zero_p = zero_p[:p.size(0)]
        else:
            zero_p = zero_p.colo_attr.data_payload.to(p.device)

        assert p.dtype == zero_p.dtype, "Parameter `{}`:\n{} vs {}".format(name, p.dtype, zero_p.dtype)
        assert allclose(p, zero_p, loose=loose), f'{p} vs {zero_p}'
