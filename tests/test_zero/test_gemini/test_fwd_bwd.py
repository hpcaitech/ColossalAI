import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing import assert_close

import colossalai
from colossalai.legacy.amp import convert_to_apex_amp
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils import set_seed
from colossalai.utils.cuda import get_current_device
from colossalai.zero import GeminiDDP, GeminiOptimizer
from colossalai.zero.gemini.chunk import search_chunk_configuration
from tests.components_to_test import run_fwd_bwd
from tests.components_to_test.registry import non_distributed_component_funcs

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
    if not model.reuse_fp16_chunk:
        chunk_list = [chunk.grad_chunk for chunk in chunk_list]
    for chunk in chunk_list:
        chunk_manager.access_chunk(chunk)

    for p0, p1 in zip(model.parameters(), torch_model.parameters()):
        assert_close(p0, p1.grad, rtol=1e-3, atol=5e-5)


@parameterize("placement_config", PLACEMENT_CONFIGS)
@parameterize("keep_gather", [False, True])
@parameterize("model_name", ["gpt2", "bert"])
@parameterize("use_grad_checkpoint", [False, True])
@parameterize("master_weights", [False, True])
def exam_gpt_fwd_bwd(
    placement_config,
    keep_gather,
    model_name: str,
    use_grad_checkpoint: bool = False,
    master_weights: bool = True,
):
    init_device = get_current_device()
    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    set_seed(42)
    model = model_builder(use_grad_checkpoint)

    set_seed(42)
    torch_model = model_builder(use_grad_checkpoint).cuda()
    for torch_p, p in zip(torch_model.parameters(), model.parameters()):
        torch_p.data.copy_(p.data)

    world_size = torch.distributed.get_world_size()
    config_dict, *_ = search_chunk_configuration(model, search_range_m=1, search_interval=100)
    config_dict[world_size]["chunk_size"] = 5000
    config_dict[world_size]["keep_gathered"] = keep_gather
    model = GeminiDDP(
        model, config_dict, init_device, pin_memory=True, **placement_config, master_weights=master_weights
    )
    optimizer = HybridAdam(model.parameters(), lr=1e-3)
    zero_optim = GeminiOptimizer(optimizer, model, initial_scale=1)

    rank = dist.get_rank()
    amp_config = dict(opt_level="O2", keep_batchnorm_fp32=False, loss_scale=1, master_weights=master_weights)
    torch_optim = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
    torch_model, torch_optim = convert_to_apex_amp(torch_model, torch_optim, amp_config)
    torch_model = DDP(torch_model, device_ids=[rank])

    set_seed(rank)
    for i, (input_ids, label) in enumerate(train_dataloader):
        # you can only test a single fwd + bwd.
        # after bwd param is grad for Gemini, due to the chunk reuse optimization.
        if i > 0:
            break
        input_ids, label = input_ids.cuda(), label.cuda()

        torch_optim.zero_grad()
        zero_optim.zero_grad()

        # set random seed is same as torch_model.eval()
        set_seed(42)
        torch_loss = run_fwd_bwd(torch_model, input_ids, label, criterion, torch_optim)
        set_seed(42)
        loss = run_fwd_bwd(model, input_ids, label, criterion, zero_optim)

        assert torch.equal(torch_loss, loss)

        check_grad(model, torch_model)


def run_dist(rank, world_size, port):
    config = {}
    colossalai.launch(config=config, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    exam_gpt_fwd_bwd()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 4])
@rerun_if_address_is_in_use()
def test_gpt(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_gpt(1)
