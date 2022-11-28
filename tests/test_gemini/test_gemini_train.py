from functools import partial

import pytest
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import colossalai
from colossalai.amp import convert_to_apex_amp
from colossalai.logging import disable_existing_loggers
from colossalai.nn.parallel import ZeroDDP
from colossalai.tensor import ProcessGroup
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port, get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext
from tests.components_to_test import run_fwd_bwd
from tests.components_to_test.registry import non_distributed_component_funcs


def check_grad(model: ZeroDDP, torch_model: torch.nn.Module):
    chunk_manager = model.chunk_manager
    param_list = [p for p in model.parameters()]
    chunk_list = chunk_manager.get_chunks(param_list)
    for chunk in chunk_list:
        chunk_manager.access_chunk(chunk)

    for (p0, p1) in zip(model.parameters(), torch_model.parameters()):
        assert torch.allclose(p0, p1.grad, atol=1e-3, rtol=1e-5), "{}".format(torch.max(torch.abs(p0 - p1.grad)).item())


def run_gemini_fwd_bwd(rank, world_size, port, model_name: str, placement_policy_str: str = 'auto', iter_num=2):
    placement_policy_str = 'auto'
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, train_dataloader, _, _, criterion = get_components_func()

    # build torch model
    torch_model = model_builder(checkpoint=False).cuda()
    # build torch
    pg = ProcessGroup()
    amp_config = dict(opt_level='O2', keep_batchnorm_fp32=False, loss_scale=1)
    torch_optim = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
    torch_model, torch_optim = convert_to_apex_amp(torch_model, torch_optim, amp_config)
    torch_model = DDP(torch_model, device_ids=[pg.rank()], process_group=pg.dp_process_group())
    torch_model.train()

    for i, (data, label) in enumerate(train_dataloader):
        if i >= iter_num:
            break
        run_fwd_bwd(torch_model, data.cuda(), label.cuda(), criterion, use_init_ctx=False)

    # build CAI model
    with ColoInitContext(device=get_current_device()):
        model = model_builder(checkpoint=False)

    from colossalai.gemini import ChunkManager, GeminiManager, search_chunk_configuration
    config_dict, _ = search_chunk_configuration(model, search_range_mb=1, search_interval_byte=100)
    chunk_manager = ChunkManager(config_dict, init_device=GeminiManager.get_default_device(placement_policy_str))
    gemini_manager = GeminiManager(placement_policy_str, chunk_manager)
    model = ZeroDDP(model, gemini_manager)
    model.train()

    for i, (data, label) in enumerate(train_dataloader):
        if i >= iter_num:
            break
        run_fwd_bwd(model, data.cuda(), label.cuda(), criterion, use_init_ctx=True)

    # check_grad(model, torch_model)
    # for p1, p2 in zip(model.parameters(), torch_model.parameters()):
    #     torch.allclose(p1.to(torch.float), p2.to(torch.float))


@pytest.mark.parametrize("model_name", ["inline_op_model", "bert", "simple_net", "gpt2", "resnet18"])
@rerun_if_address_is_in_use()
def test_gemini_train(model_name, iter_num=4):
    run_func = partial(run_gemini_fwd_bwd, world_size=1, port=free_port(), model_name=model_name, iter_num=iter_num)
    mp.spawn(run_func, nprocs=1)


if __name__ == '__main__':
    # for model_name in ["bert", "resnet18", "inline_op_model"]:
    # bert, gpt, inline_op_model, nested_model, no_leaf_module,
    # repeated_computed_layer, resnet, simple_net
    for model_name in ["resnet18"]:
        test_gemini_train(model_name=model_name, iter_num=4)
