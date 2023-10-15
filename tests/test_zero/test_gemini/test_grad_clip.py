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
from colossalai.zero import GeminiDDP, GeminiOptimizer
from colossalai.zero.gemini.chunk import search_chunk_configuration
from tests.components_to_test import run_fwd_bwd
from tests.components_to_test.registry import non_distributed_component_funcs

PLACEMENT_CONFIGS = [
    {
        "placement_policy": "static",
        "shard_param_frac": 0.0,
        "offload_optim_frac": 0.0,
        "offload_param_frac": 0.0,
    },  # zero2
    {
        "placement_policy": "static",
        "shard_param_frac": 0.0,
        "offload_optim_frac": 1.0,
        "offload_param_frac": 0.0,
    },  # zero2-offload
    {
        "placement_policy": "static",
        "shard_param_frac": 0.0,
        "offload_optim_frac": 0.5,
        "offload_param_frac": 0.0,
    },  # zero2-offload-half
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


@parameterize("placement_config", PLACEMENT_CONFIGS)
@parameterize("model_name", ["gpt2"])
def exam_grad_clipping(placement_config, model_name: str):
    set_seed(1912)
    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    torch_model = model_builder().cuda()
    amp_config = dict(opt_level="O2", keep_batchnorm_fp32=False, loss_scale=32)
    torch_optim = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
    torch_model, torch_optim = convert_to_apex_amp(torch_model, torch_optim, amp_config)
    torch_model = DDP(torch_model, device_ids=[dist.get_rank()])

    model = model_builder()

    for torch_p, p in zip(torch_model.parameters(), model.parameters()):
        p.data.copy_(torch_p.data)

    world_size = torch.distributed.get_world_size()
    config_dict, *_ = search_chunk_configuration(model, search_range_m=1, search_interval=100)
    config_dict[world_size]["chunk_size"] = 5000
    config_dict[world_size]["keep_gathered"] = False
    if placement_config["placement_policy"] != "cuda":
        init_device = torch.device("cpu")
    else:
        init_device = None

    model = GeminiDDP(
        model,
        chunk_config_dict=config_dict,
        chunk_init_device=init_device,
        pin_memory=True,
        **placement_config,
    )

    optimizer = HybridAdam(model.parameters(), lr=1e-3)
    zero_optim = GeminiOptimizer(optimizer, model, initial_scale=32, clipping_norm=1.0)

    model.train()
    torch_model.train()

    set_seed(dist.get_rank() * 3 + 128)
    for i, (data, label) in enumerate(train_dataloader):
        if i > 2:
            break
        data = data.cuda()
        label = label.cuda()

        zero_optim.zero_grad()
        torch_optim.zero_grad()

        torch_loss = run_fwd_bwd(torch_model, data, label, criterion, torch_optim)
        loss = run_fwd_bwd(model, data, label, criterion, zero_optim)
        assert_close(torch_loss, loss)

        import apex.amp as apex_amp

        torch.nn.utils.clip_grad_norm_(apex_amp.master_params(torch_optim), 1.0)
        torch_optim.step()
        zero_optim.step()

        check_param(model, torch_model)


def run_dist(rank, world_size, port):
    config = {}
    colossalai.launch(config=config, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    exam_grad_clipping()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 2])
@rerun_if_address_is_in_use()
def test_grad_clip(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_grad_clip(2)
