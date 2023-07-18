import time

import pytest
import torch
from torch.utils._pytree import tree_map

import colossalai
from colossalai.auto_parallel.offload.amp_optimizer import AMPOptimizer
from colossalai.auto_parallel.offload.mem_optimize import memory_optimize
from colossalai.auto_parallel.offload.solver import NOT_NVML
from colossalai.fx.profiler import parameter_size
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils import get_current_device
from colossalai.zero import ColoInitContext, zero_model_wrapper, zero_optim_wrapper
from tests.test_auto_parallel.test_offload.model_utils import *
from tests.test_tensor.common_utils import set_seed


@parameterize('model_name', ['gpt2_'])
@parameterize('memory_budget', [5000])
@parameterize('solver_name', ['asyn'])
def exam_fwd_bwd(model_name: str, memory_budget: float, solver_name: str):

    # build model
    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, data_gen = get_components_func()
    label = torch.randint(low=0, high=128, size=(
        64,
        8,
    ), device=get_current_device())
    criterion = LMLoss()

    set_seed(42)
    start_time = time.time()
    model = model_builder()
    model.train()
    param_size = parameter_size(model) / 1024**2 / 2
    init_time = time.time() - start_time
    print(f"init_param_size={param_size:.3f} MB | init_model_time={init_time:.3f} s")

    data_args = data_gen(device="cpu")
    wrap_fn = lambda x: x.to(dtype=torch.half) if isinstance(x, torch.Tensor) and torch.is_floating_point(x) else x
    data_args = tree_map(wrap_fn, data_args)
    start_time = time.time()
    model = memory_optimize(model, data_args, memory_budget * 1024 * 1024, solver_name)
    solver_time = time.time() - start_time
    print(f"solver_time={solver_time:.3f} s")

    hybrid_optimizer = HybridAdam(model.model.parameters(), lr=1e-3)
    optim = AMPOptimizer(hybrid_optimizer, model)

    with ColoInitContext(device=torch.device('cpu')):
        gemini_model = model_builder()
    gemini_model.train()

    hybrid_optimizer = HybridAdam(gemini_model.parameters(), lr=1e-3)
    gemini_config = dict(strict_ddp_mode=False,
                         device=torch.device('cpu'),
                         placement_policy='cpu',
                         pin_memory=True,
                         hidden_dim=8192,
                         search_range_m=128)
    gemini_model = zero_model_wrapper(gemini_model, 3, gemini_config)
    optim_config = dict(reduce_bucket_size=12 * 1024 * 1024, overlap_communication=True, verbose=True)
    gemini_optim = zero_optim_wrapper(gemini_model, hybrid_optimizer, optim_config=optim_config)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # test gemini
    time_list = []
    set_seed(42)
    data_args = data_gen(device="cuda")
    for step in range(10):
        gemini_optim.zero_grad()
        torch.cuda.synchronize()
        start_time = time.time()
        gemini_out = gemini_model(**data_args)
        gemini_loss = criterion(gemini_out, label)
        gemini_optim.backward(gemini_loss)
        torch.cuda.synchronize()
        time_list.append(time.time() - start_time)
        gemini_optim.step()

    torch.cuda.synchronize()

    exec_time = sum(sorted(time_list)[:5]) / 5
    runtime_peak_mem_alc = torch.cuda.max_memory_allocated() / 1024**2
    runtime_peak_mem_res = torch.cuda.max_memory_reserved() / 1024**2
    print(f'gemini | model_name: {model_name}')
    print(f'| exec_time={exec_time:.3f} s | param_size={param_size:.3f} MB '
          f'| runtime_peak_mem_alc={runtime_peak_mem_alc:.3f} MB| runtime_peak_mem_res={runtime_peak_mem_res:.3f} MB|')
    print(time_list)

    del data_args
    del gemini_model
    del gemini_optim
    del gemini_out
    del gemini_loss

    # test asyn offload
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    time_list = []
    set_seed(42)
    data_args = data_gen(device="cuda")
    data_args = tree_map(wrap_fn, data_args)
    for step in range(10):
        optim.zero_grad()
        torch.cuda.synchronize()
        start_time = time.time()
        loss = criterion(model(**data_args), label)
        optim.backward(loss)
        torch.cuda.synchronize()
        time_list.append(time.time() - start_time)
        optim.step()

    torch.cuda.synchronize()

    exec_time = sum(sorted(time_list)[:5]) / 5
    runtime_peak_mem_alc = torch.cuda.max_memory_allocated() / 1024**2
    runtime_peak_mem_res = torch.cuda.max_memory_reserved() / 1024**2
    print(f'solver_name: {solver_name} | model_name: {model_name}')
    print(f'| exec_time={exec_time:.3f} s | param_size={param_size:.3f} MB '
          f'| runtime_peak_mem_alc={runtime_peak_mem_alc:.3f} MB| runtime_peak_mem_res={runtime_peak_mem_res:.3f} MB|')
    print(time_list)


def run_dist(rank, world_size, port):
    config = {}
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    exam_fwd_bwd()


@pytest.mark.skip("this test failed")
@pytest.mark.skipif(NOT_NVML, reason='pynvml is not installed')
@rerun_if_address_is_in_use()
def test_perf():
    spawn(run_dist, 1)


if __name__ == '__main__':
    test_perf()
