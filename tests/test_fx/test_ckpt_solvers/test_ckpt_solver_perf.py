import os
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tm
from torch.fx import symbolic_trace
import torch.fx
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import colossalai
from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from colossalai.fx import ColoGraphModule, ColoTracer
from colossalai.fx.passes.algorithms import solver_rotor
from colossalai.utils import free_port
import pytest
import time
import numpy as np

try:
    from colossalai.fx.codegen import ActivationCheckpointCodeGen
except:
    from colossalai.fx.codegen import python_code_with_activation_checkpoint


def _data_gen(batch_size: int, shape: Tuple[int, int, int], device='cuda'):
    data = torch.rand(batch_size, *shape, device=device)
    label = torch.empty(batch_size, dtype=torch.long, device=device).random_(1000)
    return data, label


def _test_forward(gm: torch.fx.GraphModule, bs: int = 32, num_steps: int = 5):
    gm.train()
    gm.cuda()
    criterion = CrossEntropyLoss()
    step_time = float('inf')
    for _ in range(num_steps):
        data, label = _data_gen(bs, (3, 224, 224))
        torch.cuda.synchronize(device="cuda")
        time0 = time.time()
        output = gm(data)
        loss = criterion(output, label)
        loss.backward()
        torch.cuda.synchronize(device="cuda")
        time1 = time.time()
        step_time = min(step_time, time1 - time0)
        for child in gm.children():
            for param in child.parameters():
                param.grad = None
        del data, label

    gm.to("cpu")

    return torch.cuda.max_memory_allocated(device="cuda") / 1024**2, step_time * 1.0e3


@pytest.mark.skip(reason="no needed in pytest")
def test_meta_info_prop(bs: int = 32, rank=0):
    colossalai.launch(config={}, rank=rank, world_size=1, host='localhost', port=free_port(), backend='nccl')

    for M in [
            tm.resnet18, tm.resnet34, tm.resnet50, tm.resnet101, tm.resnet152, tm.densenet121, tm.densenet161,
            tm.densenet169, tm.densenet201
    ]:

        torch.cuda.reset_peak_memory_stats()
        print(f"Testing {M.__name__}")
        model = M()
        data = torch.rand(bs, 3, 224, 224, device='meta')
        mem, step_time = _test_forward(model, bs=bs, num_steps=10)
        if os.path.exists(f"./{M.__name__}.log"):
            os.remove(f"./{M.__name__}.log")
        with open(f"./{M.__name__}.log", "a") as f:
            f.write(
                f'|{M}|mem_limit: None|real memory consumption: {mem:.3f} MB|train step time: {step_time:.3f} MS|\n')
            f.write(f"=============================================\n")
        mem_limits = np.linspace(int(mem / 3) // 100 * 100, int(mem * 1.2) // 100 * 100, 11)

        for mem_limit in mem_limits:
            try:
                del model
                if 'gm' in locals():
                    del gm
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                model = M()
                tracer = ColoTracer()
                graph = tracer.trace(model, meta_args={"x": data})
                graph.set_codegen(ActivationCheckpointCodeGen())
                gm = ColoGraphModule(model, graph, model.__class__.__name__)
                gm = solver_rotor(gm, data, mem_limit * 1024 * 1024)
                gm.recompile()
                mem, step_time = _test_forward(gm, bs=bs, num_steps=10)
            except:
                mem = mem_limit
                step_time = float('inf')
            with open(f"./{M.__name__}.log", "a") as f:
                if hasattr(gm, "__sequence__"):
                    f.write(str(gm.__sequence__) + "\n")
                f.write(
                    f'|{M}|mem_limit: {mem_limit} MB|real memory consumption: {mem:.3f} MB|train step time: {step_time:.3f} MS|\n'
                )
                f.write(f"=============================================\n")


if __name__ == '__main__':
    test_meta_info_prop(bs=128)
