import torch
import colossalai
import pytest
import torch.multiprocessing as mp
from colossalai.utils.cuda import get_current_device
from colossalai.gemini.memory_tracer import MemStatsCollector
from colossalai.gemini.memory_tracer import GLOBAL_MODEL_DATA_TRACER
from colossalai.utils.memory import colo_set_process_memory_fraction
from colossalai.gemini import StatefulTensorMgr
from colossalai.zero.sharded_param.sharded_param import ShardedParamV2
from colossalai.zero.sharded_param.tensorful_state import TensorState
from colossalai.utils import free_port
from colossalai.testing import rerun_if_address_is_in_use
from torch.nn.parameter import Parameter
from typing import List
from functools import partial

from colossalai.gemini import StatefulTensorMgr
from colossalai.gemini.tensor_placement_policy import AutoTensorPlacementPolicy


class Net(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # each parameter is 128 MB
        self.p0 = Parameter(torch.empty(1024, 1024, 32))
        self.p1 = Parameter(torch.empty(1024, 1024, 32))
        self.p2 = Parameter(torch.empty(1024, 1024, 32))


def limit_cuda_memory(memory_in_g: float):
    cuda_capacity = torch.cuda.get_device_properties(get_current_device()).total_memory
    fraction = (memory_in_g * 1024**3) / cuda_capacity
    colo_set_process_memory_fraction(fraction)


def run_stm():
    # warmup phase use 20% CUDA memory to store params
    # only 2 params can be on CUDA
    limit_cuda_memory(1.26)
    model = Net()
    for p in model.parameters():
        p.colo_attr = ShardedParamV2(p, set_data_none=True)
    GLOBAL_MODEL_DATA_TRACER.register_model(model)
    mem_collector = MemStatsCollector()
    tensor_placement_policy = AutoTensorPlacementPolicy(mem_stats_collector=mem_collector)
    stateful_tensor_mgr = StatefulTensorMgr(tensor_placement_policy)
    for p in model.parameters():
        stateful_tensor_mgr.register_stateful_param(p.colo_attr)

    mem_collector.start_collection()
    # Compute order: 0 1 2 0 1
    # warmup
    # use naive eviction strategy
    apply_adjust(model, model.p0, [model.p0], stateful_tensor_mgr)
    mem_collector.sample_model_data()
    mem_collector.sample_overall_data()
    apply_adjust(model, model.p1, [model.p0, model.p1], stateful_tensor_mgr)
    mem_collector.sample_model_data()
    mem_collector.sample_overall_data()
    apply_adjust(model, model.p2, [model.p1, model.p2], stateful_tensor_mgr)
    mem_collector.sample_model_data()
    mem_collector.sample_overall_data()
    apply_adjust(model, model.p0, [model.p0, model.p2], stateful_tensor_mgr)
    mem_collector.sample_model_data()
    mem_collector.sample_overall_data()
    apply_adjust(model, model.p1, [model.p1, model.p2], stateful_tensor_mgr)
    mem_collector.sample_model_data()
    mem_collector.finish_collection()
    stateful_tensor_mgr.reset()

    # warmup done
    # only 2 params can be on CUDA
    limit_cuda_memory(0.26 / tensor_placement_policy._steady_cuda_cap_ratio)
    # use OPT-like eviction strategy
    apply_adjust(model, model.p0, [model.p0, model.p1], stateful_tensor_mgr)
    apply_adjust(model, model.p1, [model.p0, model.p1], stateful_tensor_mgr)
    apply_adjust(model, model.p2, [model.p0, model.p2], stateful_tensor_mgr)
    apply_adjust(model, model.p0, [model.p0, model.p2], stateful_tensor_mgr)
    apply_adjust(model, model.p1, [model.p1, model.p2], stateful_tensor_mgr)


def apply_adjust(model: torch.nn.Module, compute_param: Parameter, cuda_param_after_adjust: List[Parameter],
                 stateful_tensor_mgr: StatefulTensorMgr):
    compute_param.colo_attr._sharded_data_tensor.trans_state(TensorState.COMPUTE)
    for p in model.parameters():
        if p is not compute_param and p.colo_attr._sharded_data_tensor.state != TensorState.HOLD:
            p.colo_attr._sharded_data_tensor.trans_state(TensorState.HOLD)
    stateful_tensor_mgr.adjust_layout()
    print_stats(model)
    device = torch.device(torch.cuda.current_device())
    cuda_param_after_adjust = [hash(p) for p in cuda_param_after_adjust]
    for n, p in model.named_parameters():
        if hash(p) in cuda_param_after_adjust:
            assert p.colo_attr._sharded_data_tensor.device == device, f'{n} {p.colo_attr._sharded_data_tensor.device} vs {device}'
        else:
            assert p.colo_attr._sharded_data_tensor.device == torch.device('cpu')


def print_stats(model: torch.nn.Module):
    msgs = []
    for n, p in model.named_parameters():
        msgs.append(f'{n}: {p.colo_attr._sharded_data_tensor.state}({p.colo_attr._sharded_data_tensor.device})')
    print(f'[ {", ".join(msgs)} ]')


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_stm()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_stateful_tensor_manager(world_size=1):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    # this unit test can pass if available CUDA memory >= 1.5G
    test_stateful_tensor_manager()
