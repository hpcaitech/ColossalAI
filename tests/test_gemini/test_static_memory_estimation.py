import numpy as np
import torch
import colossalai
import pytest
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from colossalai.utils.cuda import get_current_device
from colossalai.utils import free_port
from colossalai.testing import rerun_if_address_is_in_use
from functools import partial

from colossalai.gemini.memory_tracer.memstats_collector import MemStatsCollectorStatic

from model_repo import SimpleNet, NestedNet, NoLeafModule, NetWithRepeatedlyComputedLayers


GPT_BATCH_SIZE = 8
TM_BATCH_SIZE = 64

def run_mem_collector_testing():

    model = NetWithRepeatedlyComputedLayers(checkpoint=False)

    data = torch.rand(int(TM_BATCH_SIZE), 1024, device='meta')
    # data = torch.randint(low=0, high=2048, size=(TM_BATCH_SIZE, 16), device='meta')

    mem_collector = MemStatsCollectorStatic(model)
    mem_collector.init_mem_stats(x=data)


    cuda_non_model_data_list = np.array(mem_collector._non_model_data_cuda_list) / 1024 ** 2
    print("_non_model_data_cuda_list", len(cuda_non_model_data_list))
    print(cuda_non_model_data_list)


    s_res_file = open("res_static.txt", "w", encoding="utf-8")
    for ddd in cuda_non_model_data_list:
        s_res_file.write(str(ddd/2) + "\n")
    s_res_file.close()


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
