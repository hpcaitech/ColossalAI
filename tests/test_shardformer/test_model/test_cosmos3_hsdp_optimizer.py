"""Actual four-rank Gloo HSDP AdamW, accumulation, clipping, offload and optimizer resume.

CPU offload here validates CPU master/state paths, not device-to-host transfers. Async all-gather uses real
Gloo handles; communication-stream overlap cannot be validated on CPU (Tensor.record_stream needs a device
stream). A CPU accelerator shim supplies only the stream API needed when overlap_communication=False.
"""

import copy
import io
from contextlib import nullcontext
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from colossalai.accelerator import set_accelerator
from colossalai.zero.low_level import LowLevelZeroOptimizer
from colossalai.zero.low_level.zero_hook import wait_all_gather_handle
from tests.test_shardformer.test_model.test_hsdp_and_context_parallel import _CpuZeROAccelerator, _hsdp_groups

WORLD = 4
MICROBATCHES = 3
MAX_NORM = 0.2


def _model():
    torch.manual_seed(314)
    # Odd parameter sizes exercise shard padding and unpadding during optimizer state gathering/restoring.
    return torch.nn.Sequential(torch.nn.Linear(7, 11), torch.nn.Tanh(), torch.nn.Linear(11, 5))


def _optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=0.003, betas=(0.8, 0.95), eps=1e-6, weight_decay=0.07)


def _zero(model, shard_group, replica_group, partition_grad, offload, overlap_allgather):
    return LowLevelZeroOptimizer(
        _optimizer(model),
        dp_process_group=shard_group,
        replica_group=replica_group,
        partition_grad=partition_grad,
        cpu_offload=offload,
        overlap_allgather=overlap_allgather,
        overlap_communication=False,
        clip_grad_norm=MAX_NORM,
        reduce_bucket_size=128,
    )


def _compare_state(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=2e-5, atol=2e-6)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _compare_state(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for got, want in zip(actual, expected):
            _compare_state(got, want)
    else:
        assert actual == expected


def _run_config(shard_group, replica_group, partition_grad, offload, overlap_allgather):
    rank = dist.get_rank()
    model, reference = _model(), _model()
    optimizer = _zero(model, shard_group, replica_group, partition_grad, offload, overlap_allgather)
    reference_optimizer = _optimizer(reference)
    for step in range(4):
        reference_optimizer.zero_grad(set_to_none=True)
        for micro in range(MICROBATCHES):
            generator = torch.Generator().manual_seed(12345 + rank * 100 + step * MICROBATCHES + micro)
            inputs = torch.randn(6, 7, generator=generator)
            targets = torch.randn(6, 5, generator=generator)
            # ZeRO1 accumulates local gradients under no_sync; ZeRO2 accumulates reduced shards each backward.
            context = optimizer.no_sync() if not partition_grad and micro < MICROBATCHES - 1 else nullcontext()
            with context:
                optimizer.backward(8 * (model(inputs) - targets).square().mean() / MICROBATCHES)
            (8 * (reference(inputs) - targets).square().mean() / MICROBATCHES).backward()
        # Independent plain DP reference: average dense full gradients, then clip the one global vector.
        for parameter in reference.parameters():
            dist.all_reduce(parameter.grad, group=dist.group.WORLD)
            parameter.grad.div_(WORLD)
        norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), MAX_NORM)
        assert norm > MAX_NORM, "the regression must actually activate clipping"
        reference_optimizer.step()
        optimizer.step()
        # Real async handles are waited before any read/next forward, matching ZeroOpHook's contract.
        for parameter in model.parameters():
            if overlap_allgather:
                assert hasattr(parameter, "_all_gather_handle")
            wait_all_gather_handle(parameter)
        torch.testing.assert_close(torch.tensor(optimizer.get_grad_norm()), norm, rtol=2e-5, atol=2e-6)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        optimizer.zero_grad()
        # Every replica's gathered optimizer moments/step must agree with the dense AdamW baseline.
        state = optimizer.state_dict()
        _compare_state(state, reference_optimizer.state_dict())
        assert all(
            parameter.device.type == "cpu" for group in optimizer.optim.param_groups for parameter in group["params"]
        )
        if step == 1:
            # Serialize an actual model+optimizer checkpoint, rebuild both objects and continue two more updates.
            checkpoint = io.BytesIO()
            torch.save({"model": model.state_dict(), "optimizer": state}, checkpoint)
            checkpoint.seek(0)
            restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model = _model()
            model.load_state_dict(restored["model"])
            optimizer = _zero(model, shard_group, replica_group, partition_grad, offload, overlap_allgather)
            optimizer.load_state_dict(copy.deepcopy(restored["optimizer"]))
            _compare_state(optimizer.state_dict(), state)


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=WORLD, timeout=timedelta(seconds=90)
    )
    set_accelerator(_CpuZeROAccelerator())
    try:
        shard_group, replica_group = _hsdp_groups(2)
        for partition_grad in (False, True):
            for offload in (False, True):
                for overlap_allgather in (False, True):
                    _run_config(shard_group, replica_group, partition_grad, offload, overlap_allgather)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires real Gloo collectives")
def test_hsdp_adamw_accumulation_clipping_offload_and_checkpoint(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "hsdp-adamw"),), nprocs=WORLD, join=True)
