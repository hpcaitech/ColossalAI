import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing import assert_close

import colossalai
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils import set_seed
from colossalai.utils.cuda import get_current_device
from colossalai.zero import GeminiDDP, GeminiOptimizer
from colossalai.zero.gemini.chunk import search_chunk_configuration
from tests.components_to_test import run_fwd
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
    for chunk in chunk_list:
        chunk_manager.access_chunk(chunk)

    for (n, p0), (_, p1) in zip(model.named_parameters(), torch_model.named_parameters()):
        # after backward, gradient is placed at parameters chunks
        assert_close(
            p0, p1.grad.to(p0.dtype), rtol=1e-3, atol=1e-3, msg=f"{n}, gemini_grad: {p0}, torch_grad: {p1.grad}"
        )


@parameterize("placement_config", PLACEMENT_CONFIGS)
@parameterize("keep_gathered", [False, True])
@parameterize("model_name", ["gpt2"])
@parameterize("use_grad_checkpoint", [False, True])
def exam_gemini_grad_acc(placement_config, keep_gathered: bool, use_grad_checkpoint: bool, model_name: str):
    init_device = get_current_device()
    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    set_seed(42)
    gemini_model = model_builder(use_grad_checkpoint)

    set_seed(42)
    torch_model = model_builder(use_grad_checkpoint).cuda()
    for torch_p, p in zip(torch_model.parameters(), gemini_model.parameters()):
        torch_p.data.copy_(p.data)

    world_size = torch.distributed.get_world_size()
    config_dict, *_ = search_chunk_configuration(gemini_model, search_range_m=1, search_interval=100)
    config_dict[world_size]["chunk_size"] = 5000
    config_dict[world_size]["keep_gathered"] = keep_gathered
    gemini_model = GeminiDDP(
        gemini_model, config_dict, init_device, pin_memory=True, enable_gradient_accumulation=True, **placement_config
    )
    optimizer = HybridAdam(gemini_model.parameters(), lr=1e-3)
    gemini_optim = GeminiOptimizer(optimizer, gemini_model, initial_scale=1)

    rank = dist.get_rank()
    torch_optim = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
    torch_model = DDP(torch_model)

    set_seed(rank)
    accum_iter = 4
    for i, (input_ids, label) in enumerate(train_dataloader):
        input_ids, label = input_ids.cuda(), label.cuda()

        torch_model.train()
        gemini_model.train()

        set_seed(42 + rank)
        torch_loss = run_fwd(torch_model, input_ids, label, criterion)
        torch_loss = torch_loss / accum_iter
        torch_loss.backward()

        set_seed(42 + rank)
        gemini_loss = run_fwd(gemini_model, input_ids, label, criterion)
        gemini_loss = gemini_loss / accum_iter
        gemini_optim.backward(gemini_loss)

        print(i, torch_loss, gemini_loss)
        assert torch.allclose(torch_loss, gemini_loss, rtol=1e-3, atol=1e-5)

        if (i + 1) % accum_iter == 0:
            torch_optim.step()
            gemini_optim.step()
            continue

        check_grad(gemini_model, torch_model)

        if i == accum_iter:
            break

    # check updated param
    torch_dict = torch_model.state_dict()
    gemini_dict = gemini_model.state_dict(only_rank_0=False)

    for key, value in gemini_dict.items():
        torch_key = "module." + key
        torch_value = torch_dict[torch_key].to(device=value.device, dtype=value.dtype)
        assert_close(value, torch_value, rtol=1e-3, atol=2e-3)


def run_dist(rank, world_size, port):
    config = {}
    colossalai.launch(config=config, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    exam_gemini_grad_acc()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_grad_accumulation():
    spawn(run_dist, 2)


if __name__ == "__main__":
    test_grad_accumulation()
