import colossalai
import pytest
import torch.multiprocessing as mp
from colossalai.utils import free_port
from colossalai.testing import rerun_if_address_is_in_use
from functools import partial

from colossalai.gemini.memory_tracer.static_memstats_collector import StaticMemStatsCollector
from tests.components_to_test.registry import non_distributed_component_funcs


def run_mem_collector_testing():

    model_input_name_dict = {'gpt2': ['input_ids', 'attention_mask'],
                                'resnet18': ['x'],
                                'simple_net': ['x']}

    for model_name, input_names in model_input_name_dict.items():

        get_components_func = non_distributed_component_funcs.get_callable(model_name)
        model_builder, train_dataloader, _, _, criterion = get_components_func()

        model = model_builder(checkpoint=False)
        mem_collector = StaticMemStatsCollector(model)

        for i, inputs in enumerate(train_dataloader):

            meta_args = {}

            for idx, inp_name in enumerate(input_names):
                meta_args[inp_name] = inputs[idx].to(device='meta')
            mem_collector.init_mem_stats(**meta_args)

            break

        del model, mem_collector


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_mem_collector_testing()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_mem_collector(world_size=1):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)

if __name__ == '__main__':
    test_mem_collector()
