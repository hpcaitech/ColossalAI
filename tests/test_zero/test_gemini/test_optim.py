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
from tests.kit.model_zoo import model_zoo, run_fwd_bwd

PLACEMENT_CONFIGS = [
    {"placement_policy": "static", "shard_param_frac": 0.3, "offload_param_frac": 0.3, "offload_optim_frac": 0.3},
    {"placement_policy": "auto"},
]

# this model is large enough to slice to chunks
TEST_MODELS = ["transformers_gpt_lm"]
# these models are too small, all parameters in these models are compacted into one chunk
EXAMPLE_MODELS = [
    "transformers_bert_for_sequence_classification",
    "custom_hanging_param_model",
    "custom_nested_model",
    "custom_repeated_computed_layers",
]

# bfloat16 cannot represent them exactly
BF16_IGNORED_KEYS = [
    "masked_bias",
]


def check_param(model: GeminiDDP, torch_model: torch.nn.Module, dtype: torch.dtype):
    zero_dict = model.state_dict(only_rank_0=False)
    torch_dict = torch_model.state_dict()

    for key, value in torch_dict.items():
        # key is 'module.model.PARAMETER', so we truncate it
        key = key[7:]
        assert key in zero_dict, "{} not in ZeRO dictionary.".format(key)
        temp_zero_value = zero_dict[key].to(device=value.device)
        if dtype is torch.bfloat16 and any(k in key for k in BF16_IGNORED_KEYS):
            continue
        rtol, atol = 2e-3, 6e-3
        if dtype is torch.bfloat16:
            rtol, atol = 4e-3, 8e-3
        # debug_print([0], "max range: ", key, torch.max(torch.abs(value - temp_zero_value)))
        assert_close(
            value.float(),
            temp_zero_value.float(),
            rtol=rtol,
            atol=atol,
            msg=lambda s: s + f"\n{key}\n{temp_zero_value.dtype}",
        )


@parameterize("placement_config", PLACEMENT_CONFIGS)
@parameterize("model_name", TEST_MODELS)
@parameterize("mixed_precision", [torch.half, torch.bfloat16])
@parameterize("master_weights", [True, False])
@parameterize("enable_async_reduce", [True])
def exam_model_step(
    placement_config, model_name: str, mixed_precision: torch.dtype, master_weights: bool, enable_async_reduce=True
):
    set_seed(42)
    model_builder, data_gen_fn, output_transform_fn, loss_fn, *_ = next(
        iter(model_zoo.get_sub_registry(model_name).values())
    )

    torch_model = model_builder().cuda()
    # apex no master weights leads to nan, so we don't use it
    amp_config = dict(opt_level="O2", keep_batchnorm_fp32=False, loss_scale=128)
    torch_optim = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
    torch_model, torch_optim = convert_to_apex_amp(torch_model, torch_optim, amp_config)
    torch_model = DDP(torch_model, device_ids=[dist.get_rank()])

    model = model_builder().cuda()

    for torch_p, p in zip(torch_model.parameters(), model.parameters()):
        p.data.copy_(torch_p.data)

    world_size = torch.distributed.get_world_size()
    config_dict, *_ = search_chunk_configuration(model, search_range_m=1, search_interval=100)
    config_dict[world_size]["chunk_size"] = 5000
    config_dict[world_size]["keep_gathered"] = False
    model = GeminiDDP(
        model,
        config_dict,
        **placement_config,
        mixed_precision=mixed_precision,
        master_weights=master_weights,
        enable_async_reduce=enable_async_reduce,
    )

    optimizer = HybridAdam(model.parameters(), lr=1e-3)
    zero_optim = GeminiOptimizer(optimizer, model, initial_scale=128)

    model.eval()
    torch_model.eval()

    set_seed(dist.get_rank() * 3 + 128)
    rtol, atol = 4e-2, 4e-2
    train_dataloader = iter(DummyDataloader(data_gen_fn))
    for i, data in enumerate(train_dataloader):
        if i > 2:
            break
        data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        zero_optim.zero_grad()
        torch_optim.zero_grad()

        torch_loss = run_fwd_bwd(torch_model, data, output_transform_fn, loss_fn, optimizer=torch_optim)
        loss = run_fwd_bwd(model, data, output_transform_fn, loss_fn, optimizer=zero_optim)
        # as no master weights leads to error accumulation, we don't check the loss
        if master_weights:
            assert_close(torch_loss.float(), loss.float(), rtol=rtol, atol=atol)

        zero_optim.step()
        torch_optim.step()

        if master_weights:
            check_param(model, torch_model, mixed_precision)


@parameterize("placement_config", [{"placement_policy": "static", "shard_param_frac": 1.0}])
@parameterize("model_name", EXAMPLE_MODELS)
@parameterize("mixed_precision", [torch.half])
def exam_tiny_example(placement_config, model_name: str, mixed_precision: torch.dtype):
    set_seed(2008)
    model_builder, data_gen_fn, output_transform_fn, loss_fn, *_ = next(
        iter(model_zoo.get_sub_registry(model_name).values())
    )

    torch_model = model_builder().cuda()
    amp_config = dict(opt_level="O2", keep_batchnorm_fp32=False, loss_scale=2)
    torch_optim = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
    torch_model, torch_optim = convert_to_apex_amp(torch_model, torch_optim, amp_config)
    torch_model = DDP(torch_model, device_ids=[dist.get_rank()])

    model = model_builder().cuda()

    for torch_p, p in zip(torch_model.parameters(), model.parameters()):
        p.data.copy_(torch_p.data)

    model = GeminiDDP(
        model,
        chunk_init_device=get_accelerator().get_current_device(),
        search_range_m=1,
        pin_memory=True,
        mixed_precision=mixed_precision,
        **placement_config,
    )
    optimizer = HybridAdam(model.parameters(), lr=1e-3)
    zero_optim = GeminiOptimizer(optimizer, model, initial_scale=2)

    model.eval()
    torch_model.eval()

    set_seed(dist.get_rank() * 3 + 128)

    train_dataloader = DummyDataloader(data_gen_fn)
    for i, data in enumerate(train_dataloader):
        if i > 2:
            break

        data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}

        zero_optim.zero_grad()
        torch_optim.zero_grad()

        run_fwd_bwd(torch_model, data, output_transform_fn, loss_fn, optimizer=torch_optim)
        run_fwd_bwd(model, data, output_transform_fn, loss_fn, optimizer=zero_optim)
        zero_optim.step()
        torch_optim.step()

        check_param(model, torch_model, mixed_precision)


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    exam_model_step()
    exam_tiny_example()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [4])
@rerun_if_address_is_in_use()
def test_optim(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_optim(1)
