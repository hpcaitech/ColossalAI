from functools import partial

import pytest
import torch
import torch.multiprocessing as mp

import colossalai
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.parallel import ZeroDDP
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port, get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext
from tests.components_to_test import run_fwd_bwd
from tests.components_to_test.registry import non_distributed_component_funcs


def run_gemini_fwd_bwd(rank, world_size, port, model_name: str, iter_num=2):
    PLACEMENT_POLICY = 'auto'
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, train_dataloader, _, _, criterion = get_components_func()

    # build torch model
    model_torch = model_builder(checkpoint=False).cuda()

    for i, (data, label) in enumerate(train_dataloader):
        if i >= iter_num:
            break
        run_fwd_bwd(model_torch, data.cuda(), label.cuda(), criterion, False, use_init_ctx=False)

    # build CAI model
    with ColoInitContext(device=get_current_device()):
        model = model_builder(checkpoint=False)

    from colossalai.gemini import ChunkManager, GeminiManager, search_chunk_configuration
    config_dict, _ = search_chunk_configuration(model, search_range_mb=1, search_interval_byte=100)
    chunk_manager = ChunkManager(config_dict, init_device=GeminiManager.get_default_device(PLACEMENT_POLICY))
    gemini_manager = GeminiManager(PLACEMENT_POLICY, chunk_manager)
    model = ZeroDDP(model, gemini_manager)

    model.train()

    for i, (data, label) in enumerate(train_dataloader):
        if i >= iter_num:
            break
        run_fwd_bwd(model, data.cuda(), label.cuda(), criterion, False, use_init_ctx=True)

    for p1, p2 in zip(model.parameters(), model_torch.parameters()):
        torch.allclose(p1.to(torch.float), p2.to(torch.float))
    print(f'pass test {model_name}')


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
