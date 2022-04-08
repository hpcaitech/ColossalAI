import torch
import colossalai
import pytest
from colossalai.utils.memory_tracer import MemStatsCollector
from colossalai.utils.memory_tracer.model_data_memtracer import GLOBAL_MODEL_DATA_TRACER
from colossalai.utils.memory_utils.utils import colo_cuda_memory_capacity, colo_set_process_memory_fraction
from colossalai.zero.shard_utils.tensor_utils import colo_model_data_tensor_move_inline
from colossalai.zero.shard_utils import StatefulTensorMgr
from colossalai.zero.sharded_param.sharded_param import ShardedParamV2
from colossalai.zero.sharded_param.tensorful_state import StatefulTensor, TensorState
from torch.nn.parameter import Parameter
from typing import List


class Net(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # each parameter is 512 MB
        self.p0 = Parameter(torch.empty(1024, 1024, 128))
        self.p1 = Parameter(torch.empty(1024, 1024, 128))
        self.p2 = Parameter(torch.empty(1024, 1024, 128))


@pytest.mark.skip
def test():
    colossalai.launch(config={}, rank=0, world_size=1, host='localhost', port=29444, local_rank=0)
    cuda_capacity = colo_cuda_memory_capacity()
    fraction = (1.4 * 1024**3) / cuda_capacity
    # limit max memory to 1.4GB
    # which means only 2 parameters can be on CUDA
    colo_set_process_memory_fraction(fraction)
    model = Net()
    for p in model.parameters():
        p.colo_attr = ShardedParamV2(p, rm_torch_payload=True)
    GLOBAL_MODEL_DATA_TRACER.register_model(model)
    mem_collector = MemStatsCollector()
    stateful_tensor_mgr = StatefulTensorMgr(mem_collector)
    for p in model.parameters():
        stateful_tensor_mgr.register_stateful_param(p.colo_attr)

    mem_collector.start_collection()
    # Compute order: 0 1 2 0 1
    mem_collector.sample_memstats()
    test_adjust(model, model.p0, [model.p0], stateful_tensor_mgr)
    mem_collector.sample_memstats()
    test_adjust(model, model.p1, [model.p0, model.p1], stateful_tensor_mgr)
    mem_collector.sample_memstats()
    test_adjust(model, model.p2, [model.p1, model.p2], stateful_tensor_mgr)
    mem_collector.sample_memstats()
    test_adjust(model, model.p0, [model.p0, model.p2], stateful_tensor_mgr)
    mem_collector.sample_memstats()
    test_adjust(model, model.p1, [model.p1, model.p2], stateful_tensor_mgr)
    mem_collector.finish_collection()
    mem_collector.reset_sampling_cnter()
    print('warmup done')
    stateful_tensor_mgr.reset()
    mem_collector.sample_memstats()
    test_adjust(model, model.p0, [model.p0, model.p1], stateful_tensor_mgr)
    mem_collector.sample_memstats()
    test_adjust(model, model.p1, [model.p0, model.p1], stateful_tensor_mgr)
    mem_collector.sample_memstats()
    test_adjust(model, model.p2, [model.p0, model.p2], stateful_tensor_mgr)
    mem_collector.sample_memstats()
    test_adjust(model, model.p0, [model.p0, model.p2], stateful_tensor_mgr)
    mem_collector.sample_memstats()
    test_adjust(model, model.p1, [model.p1, model.p2], stateful_tensor_mgr)


def test_adjust(model: torch.nn.Module, compute_param: Parameter, cuda_param_after_adjust: List[Parameter],
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


if __name__ == '__main__':
    test()
