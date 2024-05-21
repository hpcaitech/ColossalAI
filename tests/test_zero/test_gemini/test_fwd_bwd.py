import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing import assert_close

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.legacy.amp import convert_to_apex_amp
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils import set_seed
from colossalai.zero import GeminiDDP, GeminiOptimizer
from colossalai.zero.gemini.chunk import search_chunk_configuration
from tests.kit.model_zoo import model_zoo, run_fwd_bwd

PLACEMENT_CONFIGS = [
    {"placement_policy": "static", "shard_param_frac": 0.0},  # zero2
    {"placement_policy": "static", "shard_param_frac": 1.0},  # zero3
    {"placement_policy": "static", "shard_param_frac": 0.5},  # zero3-half
    {"placement_policy": "auto"},
]


def check_grad(model: GeminiDDP, torch_model: torch.nn.Module):
    chunk_manager = model.chunk_manager
    param_list = [p for p in model.parameters()]
    chunk_list = chunk_manager.get_chunks(param_list)
    if not model.chunk_manager.reuse_fp16_chunk:
        chunk_list = [chunk.grad_chunk for chunk in chunk_list]
    for chunk in chunk_list:
        chunk_manager.access_chunk(chunk)

    for p0, p1 in zip(model.parameters(), torch_model.parameters()):
        assert_close(p0, p1.grad, rtol=1e-3, atol=5e-5)


@parameterize("placement_config", PLACEMENT_CONFIGS)
@parameterize("keep_gather", [False, True])
@parameterize("model_name", ["transformers_gpt_lm"])
@parameterize("use_grad_checkpoint", [False, True])
@parameterize("master_weights", [False, True])
@parameterize("max_prefetch", [0, 1, 4])
def exam_gpt_fwd_bwd(
    placement_config,
    keep_gather,
    model_name: str,
    use_grad_checkpoint: bool = False,
    master_weights: bool = True,
    max_prefetch: int = 0,
):
    init_device = get_accelerator().get_current_device()
    model_builder, data_gen_fn, output_transform_fn, loss_fn, *_ = next(
        iter(model_zoo.get_sub_registry(model_name).values())
    )

    set_seed(42)
    model = model_builder()

    set_seed(42)
    torch_model = model_builder().cuda()
    for torch_p, p in zip(torch_model.parameters(), model.parameters()):
        torch_p.data.copy_(p.data)

    if use_grad_checkpoint:
        model.gradient_checkpointing_enable()
        torch_model.gradient_checkpointing_enable()

    world_size = torch.distributed.get_world_size()
    config_dict, *_ = search_chunk_configuration(model, search_range_m=1, search_interval=100)
    config_dict[world_size]["chunk_size"] = 5000
    config_dict[world_size]["keep_gathered"] = keep_gather
    model = GeminiDDP(
        model,
        config_dict,
        init_device,
        pin_memory=True,
        **placement_config,
        master_weights=master_weights,
        max_prefetch=max_prefetch,
    )
    optimizer = HybridAdam(model.parameters(), lr=1e-3)
    zero_optim = GeminiOptimizer(optimizer, model, initial_scale=1)

    rank = dist.get_rank()
    amp_config = dict(opt_level="O2", keep_batchnorm_fp32=False, loss_scale=1, master_weights=master_weights)
    torch_optim = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
    torch_model, torch_optim = convert_to_apex_amp(torch_model, torch_optim, amp_config)
    torch_model = DDP(torch_model, device_ids=[rank])

    set_seed(rank)

    data = data_gen_fn()
    data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}

    torch_optim.zero_grad()
    zero_optim.zero_grad()

    # set random seed is same as torch_model.eval()
    set_seed(42)
    torch_loss = run_fwd_bwd(torch_model, data, output_transform_fn, loss_fn, optimizer=torch_optim)
    set_seed(42)
    loss = run_fwd_bwd(model, data, output_transform_fn, loss_fn, optimizer=zero_optim)

    assert_close(torch_loss.float(), loss.float())

    check_grad(model, torch_model)


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    exam_gpt_fwd_bwd()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 4])
@rerun_if_address_is_in_use()
def test_gpt(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_gpt(1)
