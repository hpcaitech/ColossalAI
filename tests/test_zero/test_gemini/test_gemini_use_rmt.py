import pytest
import torch

import colossalai
from colossalai.tensor import ProcessGroup
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.zero import ColoInitContext, ZeroDDP
from colossalai.zero.gemini.chunk import ChunkManager, search_chunk_configuration
from colossalai.zero.gemini.gemini_mgr import GeminiManager
from colossalai.zero.gemini.memory_tracer.runtime_mem_tracer import RuntimeMemTracer
from tests.components_to_test import run_fwd_bwd
from tests.components_to_test.registry import non_distributed_component_funcs
from tests.test_tensor.common_utils import set_seed

# run gemini use the runtime memory tracer


@parameterize('placement_policy', ['auto'])
@parameterize('keep_gather', [False])
@parameterize('model_name', ['repeated_computed_layers', 'bert', 'albert', 'gpt2'])
@parameterize('use_grad_checkpoint', [False, True])
def run_gemini_use_rmt(placement_policy, keep_gather, model_name: str, use_grad_checkpoint: bool = False):
    set_seed(42)
    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    with ColoInitContext(device='cpu'):
        model = model_builder(use_grad_checkpoint)

    print(f'model_name {model_name}')
    runtime_mem_tracer = RuntimeMemTracer(model)
    for i, (input_ids, label) in enumerate(train_dataloader):
        if i > 0:
            break
        input_ids, label = input_ids.cuda(), label.cuda()

        # mem tracing
        if i == 0:
            run_fwd_bwd(runtime_mem_tracer, input_ids, label, criterion, runtime_mem_tracer)
    memstats = runtime_mem_tracer.memstats()
    runtime_tracer_non_model_data = runtime_mem_tracer._memstats._non_model_data_cuda_list
    print('runtime tracer non model data points: ', len(runtime_tracer_non_model_data))
    print('runtime tracer: ', runtime_tracer_non_model_data)
    print([memstats.param_used_step(p) for p in model.parameters()])

    if model_name == 'repeated_computed_layers':
        for idx, p in enumerate(model.parameters()):
            step_list = memstats.param_used_step(p)
            if idx < 4:
                assert len(step_list) == 4

    if model_name == 'repeated_computed_layers':
        for idx, p in enumerate(model.parameters()):
            step_list = memstats.param_used_step(p)
            if idx < 4:
                assert len(step_list) == 4

    world_size = torch.distributed.get_world_size()
    config_dict, *_ = search_chunk_configuration(model, search_range_m=1, search_interval=100)
    config_dict[world_size]['chunk_size'] = 5000
    config_dict[world_size]['keep_gathered'] = keep_gather
    chunk_manager = ChunkManager(config_dict)
    gemini_manager = GeminiManager(placement_policy, chunk_manager, memstats)
    model = ZeroDDP(model, gemini_manager, pin_memory=True)

    pg = ProcessGroup()
    set_seed(pg.dp_local_rank())
    for i, (input_ids, label) in enumerate(train_dataloader):
        # you can only test a single fwd + bwd.
        # after bwd param is grad for Gemini, due to the chunk reuse optimization.
        # print(f'iteration {i}')
        if i > 4:
            break
        input_ids, label = input_ids.cuda(), label.cuda()

        set_seed(42)
        loss = run_fwd_bwd(model, input_ids, label, criterion, model)

    gemini_non_model_data = gemini_manager._mem_stats_collector._memstats.non_model_data_list('cuda')

    # print('gemini non model data:', gemini_non_model_data)

    assert len(gemini_non_model_data) == len(runtime_tracer_non_model_data), \
        f'model_name {model_name} {len(gemini_non_model_data)} vs {len(runtime_tracer_non_model_data)}'


def run_dist(rank, world_size, port):
    config = {}
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_gemini_use_rmt()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_gemini_use_rmt(world_size):
    spawn(run_dist, world_size)


if __name__ == '__main__':
    test_gemini_use_rmt(1)
