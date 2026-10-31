import os
import tempfile
from contextlib import nullcontext
from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
from torch.optim import Adam
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralModel

import colossalai
from colossalai.booster.booster import Booster
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.testing.random import seed_all
from tests.test_moe.moe_utils import check_model_equal, ep_local_slice

# Relative L2 error allowed between ZeRO grads (bf16) and an fp32 single-GPU reference. bf16 noise stays below 2e-2
# here, while without the sp reduction the expert grads are off by 0.8-1.0.
GRAD_RTOL = 3e-2

EP_SIZE, SP_SIZE = 2, 2
# Sizes chosen so that expert params (5 x 6 = 30 elements) are not divisible by the dp size (4), which exercises the
# ZeRO padding when optimizer states are re-sharded on loading. head_dim is set separately because RoPE needs it even.
HIDDEN_SIZE, NUM_HEADS, HEAD_DIM, INTERMEDIATE_SIZE = 6, 2, 4, 5
BATCH_SIZE, SEQ_LEN = 2, 8


def build_plugin(zero_stage: int) -> MoeHybridParallelPlugin:
    return MoeHybridParallelPlugin(
        pp_size=1,
        tp_size=1,
        ep_size=EP_SIZE,
        sp_size=SP_SIZE,
        enable_sequence_parallelism=True,
        sequence_parallelism_mode="all_to_all",
        zero_stage=zero_stage,
        precision="bf16",
        overlap_communication=False,
    )


def build_model() -> MixtralModel:
    seed_all(10086)
    config = MixtralConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=2,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        num_local_experts=4,
        num_experts_per_tok=2,
        attn_implementation="sdpa",
        # transformers v5 defaults to grouped_mm experts, which need 16-byte aligned strides that these tiny
        # sizes do not have; v4 ignores this argument
        experts_implementation="eager",
    )
    return MixtralModel(config).to(torch.bfloat16).cuda()


def replica_batch(plugin: MoeHybridParallelPlugin, index: int = None) -> torch.Tensor:
    """The batch of a data replica. The ranks of one sp group share it; sp splits its sequence."""
    if index is None:
        index = dist.get_rank(plugin.dp_group)
    generator = torch.Generator(device="cuda").manual_seed(1453 + index)
    return torch.rand(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, generator=generator, device="cuda")


@parameterize("zero_stage", [1, 2])
def run_grad_sync(zero_stage: int):
    """Expert grads must be reduced over the sp group too, like the dense ones.

    Each sp rank runs the MoE layers on its own part of the sequence. The reference is a single-GPU fp32 copy of the
    model run on every data replica's batch, whose loss is the global mean loss.
    """
    plugin = build_plugin(zero_stage)
    booster = Booster(plugin=plugin)
    torch_model = build_model()
    single_model = deepcopy(torch_model).float()
    zero_model = deepcopy(torch_model)
    zero_optimizer = torch.optim.SGD(zero_model.parameters(), lr=1)
    zero_model, zero_optimizer, _, _, _ = booster.boost(zero_model, zero_optimizer)

    zero_output = zero_model(inputs_embeds=replica_batch(plugin).bfloat16()).last_hidden_state.mean()
    zero_optimizer.backward(zero_output)
    num_replicas = dist.get_world_size(plugin.dp_group)
    single_loss = sum(
        single_model(inputs_embeds=replica_batch(plugin, i)).last_hidden_state.mean() for i in range(num_replicas)
    )
    (single_loss / num_replicas).backward()

    # get_param_grad is collective: finish the loop on every rank before asserting
    errors = []
    single_params = dict(single_model.named_parameters())
    for name, param in zero_model.named_parameters():
        name = name.removeprefix("module.")
        grad = zero_optimizer.get_param_grad(param)
        grad = torch.zeros_like(param, dtype=torch.float) if grad is None else grad.float()
        ref = single_params[name].grad
        ref = torch.zeros_like(grad) if ref is None else ep_local_slice(ref.float(), param)
        err, ref_norm = (grad - ref).norm().item(), ref.norm().item()
        if err > GRAD_RTOL * ref_norm + 1e-6:
            errors.append(f"{name}: |zero - single| / |single| = {err / max(ref_norm, 1e-12):.3f}")
        peers = [torch.empty_like(grad) for _ in range(SP_SIZE)]
        dist.all_gather(peers, grad, group=plugin.sp_group)
        if not all(torch.equal(peer, peers[0]) for peer in peers[1:]):
            errors.append(f"{name}: grads differ across the sp group")
    assert not errors, f"rank {dist.get_rank()} zero_stage={zero_stage}: " + "; ".join(errors)


def run_checkpoint():
    """Sharded model and optimizer checkpoints must round-trip when expert states are sharded over moe_dp * sp."""
    plugin = build_plugin(zero_stage=1)
    booster = Booster(plugin=plugin)
    model = build_model()
    optimizer = Adam(model.parameters(), lr=1e-3)
    model, optimizer, _, _, _ = booster.boost(model, optimizer)
    optimizer.backward(model(inputs_embeds=replica_batch(plugin).bfloat16()).last_hidden_state.mean())
    optimizer.step()

    context = tempfile.TemporaryDirectory() if dist.get_rank() == 0 else nullcontext()
    with context as tmpdir:
        tmpdir = [tmpdir]
        dist.broadcast_object_list(tmpdir, src=0)
        model_dir, optim_dir = os.path.join(tmpdir[0], "model"), os.path.join(tmpdir[0], "optim")
        booster.save_model(model, model_dir, shard=True)
        booster.save_optimizer(optimizer, optim_dir, shard=True)
        dist.barrier()

        new_model = build_model()
        new_optimizer = Adam(new_model.parameters(), lr=1e-3)
        new_model, new_optimizer, _, _, _ = booster.boost(new_model, new_optimizer)
        booster.load_model(new_model, model_dir)
        check_model_equal(model, new_model, dtype=torch.bfloat16)

        snapshot = {id(p): {k: v.clone() for k, v in s.items()} for p, s in optimizer.unwrap().state.items()}
        for state in optimizer.unwrap().state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    value.zero_()
        booster.load_optimizer(optimizer, optim_dir)
        for param, state in optimizer.unwrap().state.items():
            for key, value in state.items():
                assert torch.equal(value, snapshot[id(param)][key]), f"rank {dist.get_rank()}: optimizer state differs"
        dist.barrier()


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    torch.cuda.set_device(rank)
    run_grad_sync()
    run_checkpoint()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [4])
@rerun_if_address_is_in_use()
def test_moe_ep_sp_grad_sync(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_moe_ep_sp_grad_sync(world_size=4)
