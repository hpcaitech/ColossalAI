from tests.components_to_test.registry import non_distributed_component_funcs
import pytest
import torch
import torch.multiprocessing as mp
from functools import partial
import random
import os
import numpy as np

import colossalai
from colossalai.testing import parameterize, rerun_if_address_is_in_use
from colossalai.utils.cuda import get_current_device
from colossalai.utils import free_port, ColoInitContext
from colossalai.tensor import named_params_with_colotensor, TensorSpec, ComputePattern, ParallelAction, ColoTensor, ColoOptimizer
from colossalai.context import ParallelMode

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def run_bert_1d():
    get_components_func = non_distributed_component_funcs.get_callable('bert')
    model_builder, train_dataloader, _, optimizer_class, criterion = get_components_func()
    device = get_current_device()
    
    set_seed(1)
    with ColoInitContext(device=device):
        model = model_builder(checkpoint=True)
    
    parallel_action_list_row = [
        ParallelAction(priority=1, compute_pattern=ComputePattern.TP1DRow_Linear, parallel_mode=ParallelMode.PARALLEL_1D)
    ]
    spec_row = TensorSpec(parallel_action_list_row)

    parallel_action_list_col = [
        ParallelAction(priority=1, compute_pattern=ComputePattern.TP1DCol_Linear, parallel_mode=ParallelMode.PARALLEL_1D)
    ]
    spec_col = TensorSpec(parallel_action_list_col)

    parallel_action_list_embedding_col = [
        ParallelAction(priority=1, compute_pattern=ComputePattern.TP1DCol_Embedding, parallel_mode=ParallelMode.PARALLEL_1D)
    ]
    spec_embedding_col = TensorSpec(parallel_action_list_embedding_col)

    for name, p in named_params_with_colotensor(model):
        if not isinstance(p, ColoTensor):
            continue
        print(name)
        if 'classifier' in name and ('weight' in name or 'bias' in name):
            p.set_spec(spec_col)
        if '_embeddings' in name and 'weight' in name:
            p.set_spec(spec_embedding_col)
    for name, p in named_params_with_colotensor(model):
        if not isinstance(p, ColoTensor):
            continue
        print(f"{name}: is_gathered {p.is_gathered()}")

    model = model.cuda()

    for i, (data, label) in enumerate(train_dataloader):
        if i > 5:
            break
        data = data.to(device)
        label = label.to(device)

        model.train()
        if criterion:
            output = model(data)
            loss = criterion(output, label)
        else:
            output = model(data, label)
            loss = output

        loss.backward()
        print(loss.torch_tensor())

def run_dist(rank, world_size, port):
    config = dict(parallel=dict(tensor=dict(mode="1d", size=world_size),))
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_bert_1d()

@pytest.mark.skip
@pytest.mark.dist
@parameterize('world_size', [1])
@rerun_if_address_is_in_use()
def test_bert(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)

if __name__ == '__main__':
    test_bert()