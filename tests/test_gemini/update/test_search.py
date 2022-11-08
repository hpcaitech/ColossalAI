from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import colossalai
from colossalai.gemini.chunk import search_chunk_configuration
from colossalai.tensor import ComputePattern, ComputeSpec, ProcessGroup, ShardSpec
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port, get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext
from tests.components_to_test.registry import non_distributed_component_funcs


def init_1d_row_spec(model, pg: ProcessGroup):
    tensor_spec = (ShardSpec([0], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    for n, p in model.named_parameters():
        if 'weight' in n and 'ln' not in n:
            p.set_process_group(pg)
            p.set_tensor_spec(*tensor_spec)


def exam_search_chunk_size():

    world_size = torch.distributed.get_world_size()
    pg_tp = ProcessGroup(tp_degree=world_size)

    get_components_func = non_distributed_component_funcs.get_callable('gpt2')
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    # make sure torch_model and model has the same parameter values
    with ColoInitContext(device=get_current_device()):
        model = model_builder()
    init_1d_row_spec(model, pg_tp)
    config_dict, _ = search_chunk_configuration(model,
                                                search_range_mb=1,
                                                search_interval_byte=16,
                                                min_chunk_size_mb=0,
                                                filter_exlarge_params=True)

    for key in config_dict:
        chunk_size = config_dict[key]['chunk_size']
        if world_size == 1:
            assert chunk_size == 31616
        else:
            assert chunk_size == 1024


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    exam_search_chunk_size()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_search(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_search(4)
