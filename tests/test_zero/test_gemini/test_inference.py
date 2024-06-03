from typing import Callable

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing import assert_close

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.legacy.amp import convert_to_apex_amp
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import DummyDataloader, parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils import set_seed
from colossalai.zero import GeminiDDP, GeminiOptimizer
from colossalai.zero.gemini.chunk import search_chunk_configuration
from tests.kit.model_zoo import model_zoo, run_fwd, run_fwd_bwd

PLACEMENT_CONFIGS = [
    {"placement_policy": "static", "shard_param_frac": 0.0},  # zero2
    {"placement_policy": "static", "shard_param_frac": 1.0},  # zero3
    {"placement_policy": "static", "shard_param_frac": 0.5},  # zero3-half
    {"placement_policy": "auto"},
]


def check_param(model: GeminiDDP, torch_model: torch.nn.Module):
    zero_dict = model.state_dict(only_rank_0=False)
    torch_dict = torch_model.state_dict()

    for key, value in torch_dict.items():
        # key is 'module.model.PARAMETER', so we truncate it
        key = key[7:]
        assert key in zero_dict, "{} not in ZeRO dictionary.".format(key)
        temp_zero_value = zero_dict[key].to(device=value.device, dtype=value.dtype)
        # debug_print([0], "max range: ", key, torch.max(torch.abs(value - temp_zero_value)))
        assert_close(value, temp_zero_value, rtol=1e-3, atol=4e-3)


def multi_chunk_init(model: torch.nn.Module, placement_config: dict):
    world_size = dist.get_world_size()
    config_dict, *_ = search_chunk_configuration(model, search_range_m=1, search_interval=100)
    config_dict[world_size]["chunk_size"] = 5000
    config_dict[world_size]["keep_gathered"] = False
    model = GeminiDDP(model, config_dict, pin_memory=True, **placement_config)
    return model


def single_chunk_init(model: torch.nn.Module, placement_config: dict):
    model = GeminiDDP(
        model, chunk_init_device=get_accelerator().get_current_device(), pin_memory=True, **placement_config
    )
    return model


@parameterize("placement_config", PLACEMENT_CONFIGS)
@parameterize("model_name", ["transformers_gpt_lm"])
@parameterize("model_init_func", [single_chunk_init, multi_chunk_init])
def exam_inference(placement_config: dict, model_name: str, model_init_func: Callable):
    set_seed(19360226)
    model_builder, data_gen_fn, output_transform_fn, *_ = next(iter(model_zoo.get_sub_registry(model_name).values()))

    torch_model = model_builder().cuda()
    amp_config = dict(opt_level="O2", keep_batchnorm_fp32=False, loss_scale=128)
    torch_optim = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
    torch_model, torch_optim = convert_to_apex_amp(torch_model, torch_optim, amp_config)
    torch_model = DDP(torch_model, device_ids=[dist.get_rank()])
    init_dev = get_accelerator().get_current_device()
    model = model_builder().to(init_dev)

    for torch_p, p in zip(torch_model.parameters(), model.parameters()):
        p.data.copy_(torch_p.data)

    model = model_init_func(model, placement_config)
    optimizer = HybridAdam(model.parameters(), lr=1e-3)
    zero_optim = GeminiOptimizer(optimizer, model, initial_scale=128)

    model.eval()
    torch_model.eval()

    set_seed(dist.get_rank() * 3 + 128)
    train_dataloader = iter(DummyDataloader(data_gen_fn))

    def train_iter():
        data = next(train_dataloader)
        data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        zero_optim.zero_grad()
        torch_optim.zero_grad()
        torch_loss = run_fwd_bwd(torch_model, data, output_transform_fn, optimizer=torch_optim)
        loss = run_fwd_bwd(model, data, output_transform_fn, optimizer=zero_optim)
        assert_close(torch_loss.float(), loss.float(), rtol=1e-5, atol=1e-5)
        zero_optim.step()
        torch_optim.step()
        check_param(model, torch_model)

    def inference_iter():
        data = next(train_dataloader)
        data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        with torch.no_grad():
            torch_loss = run_fwd(torch_model, data, output_transform_fn)
            zero_loss = run_fwd(model, data, output_transform_fn)
        assert_close(torch_loss.float(), zero_loss.float(), rtol=1e-5, atol=1e-5)

    train_iter()
    inference_iter()
    train_iter()


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    exam_inference()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 4])
@rerun_if_address_is_in_use()
def test_inference(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_inference(1)
