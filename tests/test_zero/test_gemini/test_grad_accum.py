import pytest
import torch
import torch.distributed as dist
from apex import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing import assert_close

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import DummyDataloader, parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils import set_seed
from colossalai.zero import GeminiDDP, GeminiOptimizer
from colossalai.zero.gemini.chunk import search_chunk_configuration
from tests.kit.model_zoo import model_zoo, run_fwd

PLACEMENT_CONFIGS = [
    {"placement_policy": "static", "shard_param_frac": 0.75},
    {"placement_policy": "auto"},
]


def check_grad(model: GeminiDDP, torch_model: torch.nn.Module):
    chunk_manager = model.chunk_manager
    grad_chunk_list = []
    device_list = []

    # Access gradient chunks.
    for p in model.parameters():
        grad_chunk = chunk_manager.get_chunk(p).grad_chunk
        if grad_chunk not in grad_chunk_list:
            chunk_manager.access_chunk(grad_chunk)
            grad_chunk_list.append(grad_chunk)
            device_list.append(model.grads_device[p])

    # Compare gradients.
    for p0, p1 in zip(model.parameters(), torch_model.parameters()):
        assert_close(p0, p1.grad, rtol=2e-3, atol=2e-2)

    # Release gradient chunks and move them to gradient device.
    for grad_chunk, device in zip(grad_chunk_list, device_list):
        chunk_manager.release_chunk(grad_chunk)
        chunk_manager.move_chunk(grad_chunk, device, force_copy=True)


@parameterize("placement_config", PLACEMENT_CONFIGS)
@parameterize("keep_gathered", [False, True])
@parameterize("model_name", ["transformers_gpt_lm"])
@parameterize("master_weights", [False, True])
@parameterize("use_grad_checkpoint", [False, True])
@parameterize("max_prefetch", [0, 4])
@parameterize("enable_async_reduce", [False, True])
def exam_gemini_grad_acc(
    placement_config,
    keep_gathered: bool,
    model_name: str,
    master_weights: bool,
    use_grad_checkpoint: bool,
    max_prefetch: int,
    enable_async_reduce: bool,
):
    init_device = get_accelerator().get_current_device()
    model_builder, data_gen_fn, output_transform_fn, loss_fn, *_ = next(
        iter(model_zoo.get_sub_registry(model_name).values())
    )

    set_seed(42)
    gemini_model = model_builder()

    set_seed(42)
    torch_model = model_builder().cuda()
    for torch_p, p in zip(torch_model.parameters(), gemini_model.parameters()):
        torch_p.data.copy_(p.data)

    if use_grad_checkpoint:
        gemini_model.gradient_checkpointing_enable()
        torch_model.gradient_checkpointing_enable()

    world_size = torch.distributed.get_world_size()
    config_dict, *_ = search_chunk_configuration(gemini_model, search_range_m=1, search_interval=100)
    config_dict[world_size]["chunk_size"] = 5000
    config_dict[world_size]["keep_gathered"] = keep_gathered
    gemini_model = GeminiDDP(
        gemini_model,
        config_dict,
        init_device,
        pin_memory=True,
        enable_gradient_accumulation=True,
        master_weights=master_weights,
        max_prefetch=max_prefetch,
        enable_async_reduce=enable_async_reduce,
        **placement_config,
    )
    optimizer = HybridAdam(gemini_model.parameters(), lr=1e-3)
    gemini_optim = GeminiOptimizer(
        optimizer, gemini_model, initial_scale=1, max_norm=1.0, enable_async_reduce=enable_async_reduce
    )

    rank = dist.get_rank()

    # setting master_weights to False will cause overflow after optimizer.step()
    amp_config = dict(
        opt_level="O2", keep_batchnorm_fp32=False, loss_scale=1, min_loss_scale=1, max_loss_scale=1, master_weights=True
    )
    torch_optim = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
    torch_model, torch_optim = amp.initialize(torch_model, torch_optim, **amp_config)
    torch_model = DDP(torch_model, device_ids=[rank])

    set_seed(rank)
    accum_iter = 2
    train_dataloader = DummyDataloader(data_gen_fn)
    for i, data in enumerate(train_dataloader):
        delay_unscale = False if (i + 1) % accum_iter == 0 else True
        data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}

        set_seed(42 + rank)
        torch_loss = run_fwd(torch_model, data, output_transform_fn, loss_fn)
        torch_loss = torch_loss / accum_iter
        with amp.scale_loss(torch_loss, torch_optim, delay_unscale=delay_unscale) as scaled_loss:
            scaled_loss.backward()

        set_seed(42 + rank)
        gemini_loss = run_fwd(gemini_model, data, output_transform_fn, loss_fn)
        gemini_loss = gemini_loss / accum_iter
        gemini_optim.backward(gemini_loss)

        assert torch.allclose(torch_loss.float(), gemini_loss.float(), rtol=1e-3, atol=1e-5)

        check_grad(gemini_model, torch_model)

        if (i + 1) % accum_iter == 0:
            torch.nn.utils.clip_grad_norm_(amp.master_params(torch_optim), 1.0)
            torch_optim.step()
            gemini_optim.step()
            torch_optim.zero_grad()

            # check updated param
            torch_dict = torch_model.state_dict()
            gemini_dict = gemini_model.state_dict(only_rank_0=False)

            for key, value in gemini_dict.items():
                torch_key = "module." + key
                torch_value = torch_dict[torch_key].to(value.device).to(value.dtype)
                assert_close(value, torch_value, rtol=1e-3, atol=2e-3)

        if i == accum_iter:
            break


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    exam_gemini_grad_acc()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_grad_accumulation():
    spawn(run_dist, 2)


if __name__ == "__main__":
    test_grad_accumulation()
