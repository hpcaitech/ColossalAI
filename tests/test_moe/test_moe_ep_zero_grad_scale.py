from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralModel

import colossalai
from colossalai.booster.booster import Booster
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.testing.random import seed_all
from tests.test_moe.moe_utils import assert_loose_close, ep_local_slice

# Relative L2 error allowed against the reference grads. bf16 noise stays below 1e-2 here, while the old moe_dp
# scaling was off by 0.33x-3x on the experts. The atol of assert_loose_close (4e-3) exceeds these tiny grads.
GRAD_RTOL = 2e-2

EP_SIZE = 2
HIDDEN_SIZE = 8


def run_forced_routing():
    """Expert grads must match the references when an expert is only used by some moe_dp replicas.

    The gates send tokens with hidden[0] > 0 to expert 0 and the others to expert 1. moe_dp replica 0 feeds
    hidden[0] = +5 and the other replicas feed -5, so each of the two experts gets tokens from only part of the
    moe_dp group. With random inputs every expert is hit on every replica and this case is never exercised.

    Two references: DDP over the whole dp group, and a single-GPU fp32 copy run on the concatenation of every
    rank's batch. The latter has no parallel logic at all and its loss is the global mean loss by definition, so it
    also checks that DDP (which averages in the zero grads of replicas that skip an expert) is the right target.
    """
    dtype = torch.bfloat16
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    plugin = MoeHybridParallelPlugin(
        pp_size=1,
        tp_size=1,
        ep_size=EP_SIZE,
        zero_stage=1,
        precision="bf16",
        overlap_communication=False,
    )
    booster = Booster(plugin=plugin)

    seed_all(10086)
    config = MixtralConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=HIDDEN_SIZE * 2,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=1,
    )
    torch_model = MixtralModel(config).to(dtype).cuda()
    with torch.no_grad():
        for layer in torch_model.layers:
            # transformers v5 renamed `block_sparse_moe` to `mlp`
            moe_block = layer.block_sparse_moe if hasattr(layer, "block_sparse_moe") else layer.mlp
            gate = moe_block.gate.weight
            gate.zero_()
            gate[0, 0], gate[1, 0] = 10.0, -10.0

    single_model = deepcopy(torch_model).float()
    zero_model = deepcopy(torch_model)
    zero_optimizer = torch.optim.SGD(zero_model.parameters(), lr=1)
    zero_model, zero_optimizer, _, _, _ = booster.boost(zero_model, zero_optimizer)
    ddp_model = DDP(torch_model, process_group=plugin.dp_group, find_unused_parameters=True)

    seed_all(1453 + rank)
    input_data = torch.rand(4, 7, HIDDEN_SIZE).cuda().to(dtype)
    input_data[..., 0] = 5.0 if rank // EP_SIZE == 0 else -5.0

    zero_output = zero_model(inputs_embeds=input_data).last_hidden_state.mean()
    zero_optimizer.backward(zero_output)
    ddp_output = ddp_model(inputs_embeds=input_data).last_hidden_state.mean()
    ddp_output.backward()
    assert_loose_close(zero_output, ddp_output, dtype=dtype)

    # every rank has the same batch size, so the mean over the concatenation equals the mean of per-rank losses
    all_inputs = [torch.empty_like(input_data) for _ in range(dist.get_world_size())]
    dist.all_gather(all_inputs, input_data)
    single_model(inputs_embeds=torch.cat(all_inputs).float()).last_hidden_state.mean().backward()

    # get_param_grad is collective: finish the loop on every rank before asserting, otherwise a mismatch on one rank
    # leaves the other ranks waiting in all-gather until the NCCL timeout.
    errors = []
    ddp_params = dict(ddp_model.module.named_parameters())
    single_params = dict(single_model.named_parameters())
    for n, p in zero_model.named_parameters():
        n = n.removeprefix("module.")
        grads = {
            "zero": zero_optimizer.get_param_grad(p),
            "ddp": ddp_params[n].grad,
            "single": single_params[n].grad,
        }
        grads = {k: torch.zeros_like(p, dtype=torch.float) if g is None else g.float() for k, g in grads.items()}
        grads = {k: ep_local_slice(g, p) for k, g in grads.items()}
        for name, ref_name in (("zero", "ddp"), ("zero", "single"), ("ddp", "single")):
            grad, ref_grad = grads[name], grads[ref_name]
            err, ref = (grad - ref_grad).norm().item(), ref_grad.norm().item()
            if err > GRAD_RTOL * ref + 1e-6:
                errors.append(
                    f"{n}: |{name} - {ref_name}| / |{ref_name}| = {err / max(ref, 1e-12):.3f}, "
                    f"|{name}| / |{ref_name}| = {grad.norm().item() / max(ref, 1e-12):.3f}"
                )
    assert not errors, f"rank {rank}: " + "; ".join(errors)


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_forced_routing()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [4, 8])
@rerun_if_address_is_in_use()
def test_moe_ep_zero_grad_scale(world_size):
    if world_size > torch.cuda.device_count():
        pytest.skip(f"needs {world_size} GPUs")
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_moe_ep_zero_grad_scale(world_size=4)
