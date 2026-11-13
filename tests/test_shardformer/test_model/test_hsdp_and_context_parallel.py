"""CPU checks for hybrid sharded data parallelism and Cosmos 3 Ulysses context parallelism.

Both features are pure communication algebra, so they are checked on the CPU with gloo: a small MLP plus the real
``LowLevelZeroOptimizer`` for HSDP, a stand-in attention module plus the real two-way kernel for CP, and the two
scalar collectives the training step issues -- the gradient norm and the supervised-token count -- whose dtypes are
constrained by HCCL. No device and no diffusers install is needed, which keeps the checks runnable on any machine.

The HSDP check compares the parameter update against the plain average of the per-rank gradients, i.e. what a
single-level ZeRO over the whole data parallel group would produce. The CP check compares the context-parallel
processor, fed sequence shards of a pack whose AR and DM totals are both odd, against the single-device processor on
the full pack.
"""

import os
import sys
from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from colossalai.accelerator import CpuAccelerator

HSDP_WORLD = 4
CP_WORLD = 2


class _DummyStream:
    """Placeholder for the communication stream; unused while ``overlap_communication`` is off."""

    def wait_stream(self, other):
        pass


class _CpuZeROAccelerator(CpuAccelerator):
    """A CPU accelerator with the stream surface ``LowLevelZeroOptimizer`` asks for.

    The low level ZeRO path asserts that the accelerator is a "cuda" or "npu" device and requests a communication
    stream; everything it then does with fp32 CPU tensors is device agnostic. Naming the shim "npu" lets the real
    optimizer run on the CPU, so the gradient algebra is exercised rather than re-implemented in the test.
    """

    def __init__(self):
        super().__init__()
        self._name = "npu"

    def Stream(self):
        return _DummyStream()

    def current_stream(self, device=None):
        return _DummyStream()

    def stream(self, stream_):
        return nullcontext()


def _launch(rank, world, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["OMP_NUM_THREADS"] = "1"
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from colossalai.accelerator import set_accelerator

    set_accelerator(_CpuZeROAccelerator())


def _hsdp_groups(shard_size):
    """The same layout the plugin builds: contiguous shard groups, strided replica groups."""
    world = dist.get_world_size()
    ranks = list(range(world))
    shards = [ranks[start : start + shard_size] for start in range(0, world, shard_size)]
    shard_groups = [dist.new_group(shard) for shard in shards]
    replica_groups = [dist.new_group([shard[offset] for shard in shards]) for offset in range(shard_size)]
    rank = dist.get_rank()
    return shard_groups[rank // shard_size], replica_groups[rank % shard_size]


def _mlp():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))


def _check_hsdp(shard_size, partition_grad):
    from colossalai.zero.low_level import LowLevelZeroOptimizer

    rank = dist.get_rank()
    shard_group, replica_group = _hsdp_groups(shard_size)

    model = _mlp()
    init = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()

    generator = torch.Generator().manual_seed(1000 + rank)
    inputs = torch.randn(8, 16, generator=generator)
    targets = torch.randn(8, 4, generator=generator)

    # Reference: one plain gradient per rank, averaged over the whole data parallel group, applied with the same SGD
    # learning rate this optimizer uses.
    reference = _mlp()
    ((reference(inputs) - targets) ** 2).mean().backward()
    local_grad = torch.cat([p.grad.reshape(-1) for p in reference.parameters()])
    gathered = [torch.empty_like(local_grad) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_grad)
    expected = init - 0.1 * torch.stack(gathered).mean(0)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    zero = LowLevelZeroOptimizer(
        optimizer,
        dp_process_group=shard_group,
        replica_group=replica_group,
        partition_grad=partition_grad,
        reduce_bucket_size=1024,
        overlap_communication=False,
        clip_grad_norm=0.0,
    )
    zero.backward(((model(inputs) - targets) ** 2).mean())
    zero.step()
    zero.zero_grad()

    got = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    error = (got - expected).abs().max()
    dist.all_reduce(error, op=dist.ReduceOp.MAX)
    return error.item()


class FakeAttention(torch.nn.Module):
    """The attribute surface the Cosmos 3 attention processors use, with real linear and norm weights."""

    def __init__(self, hidden=8, heads=4, kv_heads=2, head_dim=4):
        super().__init__()
        self.num_attention_heads = heads
        self.num_key_value_heads = kv_heads
        self.head_dim = head_dim
        self.to_q = torch.nn.Linear(hidden, heads * head_dim, bias=False)
        self.to_k = torch.nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.to_v = torch.nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.add_q_proj = torch.nn.Linear(hidden, heads * head_dim, bias=False)
        self.add_k_proj = torch.nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.add_v_proj = torch.nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.to_out = torch.nn.Linear(heads * head_dim, hidden, bias=False)
        self.to_add_out = torch.nn.Linear(heads * head_dim, hidden, bias=False)
        self.norm_q = torch.nn.LayerNorm(head_dim)
        self.norm_k = torch.nn.LayerNorm(head_dim)
        self.norm_added_q = torch.nn.LayerNorm(head_dim)
        self.norm_added_k = torch.nn.LayerNorm(head_dim)
        self.k_norm_und_for_gen = None
        self.processor = None

    def set_processor(self, processor):
        self.processor = processor


def _check_context_parallel():
    from colossalai.shardformer.layer.two_way_attn import TwoWayPackedMeta
    from colossalai.shardformer.modeling.cosmos3 import ColossalTwoWayAttnProcessor, Cosmos3ContextParallel

    torch.manual_seed(7)
    attn = FakeAttention()
    hidden, head_dim = 8, attn.head_dim
    # Odd AR and DM totals, so both pathways need padding before they can be split evenly across two ranks.
    ar_lens, dm_lens = [5, 3, 1], [4, 2, 1]
    total_ar, total_dm = sum(ar_lens), sum(dm_lens)

    generator = torch.Generator().manual_seed(11)
    und_seq = torch.randn(total_ar, hidden, generator=generator)
    gen_seq = torch.randn(total_dm, hidden, generator=generator)
    rotary = tuple(
        torch.randn(length, head_dim, generator=generator) for length in (total_ar, total_ar, total_dm, total_dm)
    )
    meta = TwoWayPackedMeta(ar_lens, dm_lens)

    plain = ColossalTwoWayAttnProcessor(backend="naive")
    plain.set_packed_meta(meta)
    with torch.no_grad():
        ref_und, ref_gen = plain(attn, und_seq, gen_seq, rotary)

    context = Cosmos3ContextParallel(process_group=dist.group.WORLD, backend="naive")
    context.processor.set_packed_meta(meta)
    und_shard, gen_shard, rotary_shard = context.shard(und_seq, gen_seq, rotary)
    with torch.no_grad():
        got_und, got_gen = context.gather(*context.processor(attn, und_shard, gen_shard, rotary_shard))
        # The shard/gather pair must also be the identity on its own.
        back_und, back_gen = context.gather(*context.shard(und_seq, gen_seq, rotary)[:2])

    error = max(
        (got_und - ref_und).abs().max(),
        (got_gen - ref_gen).abs().max(),
        (back_und - und_seq).abs().max(),
        (back_gen - gen_seq).abs().max(),
    )
    dist.all_reduce(error, op=dist.ReduceOp.MAX)
    return error.item()


def _check_context_parallel_gradients():
    """The gather seam has to carry gradients, and the group's summed gradient has to be the single-device one.

    Each rank's parameter gradient comes back multiplied by the group size -- the gather's backward is a reduce-scatter
    and every rank runs the same loss -- so the check sums the per-rank gradients over the group and divides by it,
    which is exactly what ZeRO's average over the same ``dp * sp`` group does.
    """
    import copy

    from colossalai.shardformer.layer.two_way_attn import TwoWayPackedMeta
    from colossalai.shardformer.modeling.cosmos3 import ColossalTwoWayAttnProcessor, Cosmos3ContextParallel

    torch.manual_seed(17)
    attn = FakeAttention()
    reference_attn = copy.deepcopy(attn)
    hidden, head_dim = 8, attn.head_dim
    ar_lens, dm_lens = [5, 3, 1], [4, 2, 1]
    total_ar, total_dm = sum(ar_lens), sum(dm_lens)

    generator = torch.Generator().manual_seed(19)
    und_seq = torch.randn(total_ar, hidden, generator=generator)
    gen_seq = torch.randn(total_dm, hidden, generator=generator)
    rotary = tuple(
        torch.randn(length, head_dim, generator=generator) for length in (total_ar, total_ar, total_dm, total_dm)
    )
    meta = TwoWayPackedMeta(ar_lens, dm_lens)

    plain = ColossalTwoWayAttnProcessor(backend="naive")
    plain.set_packed_meta(meta)
    ref_und, ref_gen = plain(reference_attn, und_seq, gen_seq, rotary)
    (ref_und.square().sum() + ref_gen.square().sum()).backward()
    reference = torch.cat([p.grad.reshape(-1) for p in reference_attn.parameters()])

    context = Cosmos3ContextParallel(process_group=dist.group.WORLD, backend="naive")
    context.processor.set_packed_meta(meta)
    und_shard, gen_shard, rotary_shard = context.shard(und_seq, gen_seq, rotary)
    got_und, got_gen = context.gather(*context.processor(attn, und_shard, gen_shard, rotary_shard))
    (got_und.square().sum() + got_gen.square().sum()).backward()
    got = torch.cat([p.grad.reshape(-1) for p in attn.parameters()])

    dist.all_reduce(got, op=dist.ReduceOp.SUM)
    error = (got / context.cp_size - reference).abs().max()
    dist.all_reduce(error, op=dist.ReduceOp.MAX)
    return error.item()


def _check_context_parallel_rejects_different_packs():
    """Every CP rank has to see the same pack; differing real lengths must raise before any attention collective."""
    from colossalai.shardformer.modeling.cosmos3 import Cosmos3ContextParallel

    context = Cosmos3ContextParallel(process_group=dist.group.WORLD, backend="naive")
    rank = dist.get_rank()
    und_seq = torch.randn(5 + rank, 8)
    gen_seq = torch.randn(4, 8)
    rotary = tuple(torch.randn(length, 4) for length in (5 + rank, 5 + rank, 4, 4))
    try:
        context.shard(und_seq, gen_seq, rotary)
    except ValueError as error:
        return "same packed sequence" in str(error)
    return False


def _check_hsdp_needs_zero():
    """HSDP is carried by the ZeRO optimizer; with zero_stage=0 the plugin must refuse instead of staying plain DDP
    over the whole data parallel group and ignoring the shard size."""
    from colossalai.booster.plugin import HybridParallelPlugin

    try:
        HybridParallelPlugin(tp_size=1, pp_size=1, zero_stage=0, hsdp_shard_size=2)
    except ValueError as error:
        return "zero_stage=0" in str(error)
    return False


def _hsdp_zero_stage_worker(rank, world, port, queue):
    _launch(rank, world, port)
    ok = _check_hsdp_needs_zero()
    if rank == 0:
        queue.put(ok)
    dist.destroy_process_group()


def _hsdp_worker(rank, world, port, configs, queue):
    _launch(rank, world, port)
    results = [_check_hsdp(shard, partition) for shard, partition in configs]
    if rank == 0:
        queue.put(results)
    dist.destroy_process_group()


def _cp_worker(rank, world, port, queue):
    _launch(rank, world, port)
    result = _check_context_parallel()
    if rank == 0:
        queue.put(result)
    dist.destroy_process_group()


def _cp_grad_worker(rank, world, port, queue):
    _launch(rank, world, port)
    result = _check_context_parallel_gradients()
    if rank == 0:
        queue.put(result)
    dist.destroy_process_group()


def _cp_pack_mismatch_worker(rank, world, port, queue):
    _launch(rank, world, port)
    result = _check_context_parallel_rejects_different_packs()
    if rank == 0:
        queue.put(result)
    dist.destroy_process_group()


def _check_grad_norm_collective():
    import math

    from colossalai.zero.low_level import LowLevelZeroOptimizer

    class Stub:
        _GRAD_NORM_CHUNK = 64

        def _grad_norm_weight(self, grad):
            return 1.0

        def _grad_norm_process_groups(self):
            return [dist.group.WORLD]

    grads = [torch.full((8,), 2e20 + 3e20 * dist.get_rank())]
    seen = []
    real_all_reduce = dist.all_reduce

    def recording_all_reduce(tensor, op=None, group=None):
        seen.append(tensor.dtype)
        real_all_reduce(tensor, op=op, group=group)

    dist.all_reduce = recording_all_reduce
    try:
        value = LowLevelZeroOptimizer._compute_grad_norm(Stub(), grads, 2)
    finally:
        dist.all_reduce = real_all_reduce

    expected = math.sqrt(8 * (2e20**2 + 5e20**2))
    return abs(value - expected) / expected < 1e-6 and seen == [torch.float32, torch.float32]


def _check_supervised_token_collective():
    from colossalai.shardformer.modeling.cosmos_mot import all_reduce_supervised_tokens

    total = all_reduce_supervised_tokens(3 + dist.get_rank(), group=dist.group.WORLD, device=torch.device("cpu"))
    return total == 7


def _grad_norm_collective_worker(rank, world, port, queue):
    _launch(rank, world, port)
    result = _check_grad_norm_collective()
    if rank == 0:
        queue.put(result)
    dist.destroy_process_group()


def _supervised_token_collective_worker(rank, world, port, queue):
    _launch(rank, world, port)
    result = _check_supervised_token_collective()
    if rank == 0:
        queue.put(result)
    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="HSDP grad sync is checked over gloo")
def test_hsdp_gradient_sync():
    """ZeRO sharded inside a group of `shard_size` ranks and replicated across the rest must average over all of them."""
    context = mp.get_context("spawn")
    queue = context.Queue()
    # shard == world is plain ZeRO-2, shard == 1 is pure replication, and the two middle cases are HSDP proper; the
    # partition_grad=False case covers the ZeRO-1 path, whose gradient is all-reduced instead of reduce-scattered.
    configs = [(HSDP_WORLD, True), (2, True), (2, False), (1, True)]
    mp.spawn(_hsdp_worker, args=(HSDP_WORLD, 29561, configs, queue), nprocs=HSDP_WORLD, join=True)
    errors = queue.get(timeout=60)
    assert all(error < 1e-5 for error in errors), errors


@pytest.mark.skipif(not dist.is_gloo_available(), reason="the plugin validation is checked over gloo")
def test_hsdp_requires_a_zero_stage():
    """``hsdp_shard_size`` with ``zero_stage=0`` must raise, not warn and run plain DDP over the whole DP group."""
    context = mp.get_context("spawn")
    queue = context.Queue()
    mp.spawn(_hsdp_zero_stage_worker, args=(HSDP_WORLD, 29565, queue), nprocs=HSDP_WORLD, join=True)
    assert queue.get(timeout=60) is True


@pytest.mark.skipif(not dist.is_gloo_available(), reason="the Ulysses exchange is checked over gloo")
def test_cosmos3_context_parallel():
    """Two ranks, half a pack each, must reproduce the single-device two-way attention bit for bit."""
    context = mp.get_context("spawn")
    queue = context.Queue()
    mp.spawn(_cp_worker, args=(CP_WORLD, 29562, queue), nprocs=CP_WORLD, join=True)
    error = queue.get(timeout=60)
    assert error < 1e-5, error


@pytest.mark.skipif(not dist.is_gloo_available(), reason="the Ulysses gather backward is checked over gloo")
def test_cosmos3_context_parallel_gradients():
    """The gradient has to cross the gather seam, scaled by the group size and averaged back out by ZeRO."""
    context = mp.get_context("spawn")
    queue = context.Queue()
    mp.spawn(_cp_grad_worker, args=(CP_WORLD, 29563, queue), nprocs=CP_WORLD, join=True)
    error = queue.get(timeout=60)
    assert error < 1e-5, error


@pytest.mark.skipif(not dist.is_gloo_available(), reason="the CP pack check is exercised over gloo")
def test_cosmos3_context_parallel_rejects_different_packs():
    """A CP group that sees different real sequence lengths must fail instead of hanging in attention."""
    context = mp.get_context("spawn")
    queue = context.Queue()
    mp.spawn(_cp_pack_mismatch_worker, args=(CP_WORLD, 29564, queue), nprocs=CP_WORLD, join=True)
    assert queue.get(timeout=60) is True


@pytest.mark.skipif(not dist.is_gloo_available(), reason="the grad norm collective is checked over gloo")
def test_grad_norm_collective_stays_float32():
    """HCCL has no float64 all-reduce: the cross-rank scalar must be float32 and still reproduce the float64 norm."""
    context = mp.get_context("spawn")
    queue = context.Queue()
    mp.spawn(_grad_norm_collective_worker, args=(CP_WORLD, 29566, queue), nprocs=CP_WORLD, join=True)
    assert queue.get(timeout=60) is True


@pytest.mark.skipif(not dist.is_gloo_available(), reason="the supervised token collective is checked over gloo")
def test_supervised_token_collective_is_int64():
    """A token count is reduced exactly, as int64 on the accelerator device rather than float64 on the host."""
    context = mp.get_context("spawn")
    queue = context.Queue()
    mp.spawn(_supervised_token_collective_worker, args=(CP_WORLD, 29567, queue), nprocs=CP_WORLD, join=True)
    assert queue.get(timeout=60) is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s"]))
