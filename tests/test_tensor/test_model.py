from tests.components_to_test.registry import non_distributed_component_funcs

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.testing import parameterize, rerun_if_address_is_in_use
from colossalai.utils.cuda import get_current_device
from colossalai.utils import free_port
from colossalai.utils import ColoInitContext
from colossalai.tensor import named_params_with_colotensor, TensorSpec, ComputePattern, ParallelAction, ColoTensor, ColoOptimizer
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc

from functools import partial
import random
import os
import numpy as np

# Hack huggingface Bert ModelOutput
# Make it available to our ColoTensor
from transformers.file_utils import ModelOutput
from dataclasses import fields


def _post_init_colo(self):
    class_fields = fields(self)
    # Safety and consistency checks
    if not len(class_fields):
        raise ValueError(f"{self.__class__.__name__} has no fields.")
    if not all(field.default is None for field in class_fields[1:]):
        raise ValueError(f"{self.__class__.__name__} should not have more than one required field.")

    first_field = getattr(self, class_fields[0].name)
    other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

    def is_tensor_with_colo(x):
        """
        Tests if `x` is a `ColoTensor` or `torch.Tensor`.
        """
        if isinstance(x, torch.Tensor):
            return True

        return isinstance(x, ColoTensor)

    if other_fields_are_none and not is_tensor_with_colo(first_field):
        if isinstance(first_field, dict):
            iterator = first_field.items()
            first_field_iterator = True
        else:
            try:
                iterator = iter(first_field)
                first_field_iterator = True
            except TypeError:
                first_field_iterator = False

        # if we provided an iterator as first field and the iterator is a (key, value) iterator
        # set the associated fields
        if first_field_iterator:
            for element in iterator:
                if (not isinstance(element, (list, tuple)) or not len(element) == 2 or not isinstance(element[0], str)):
                    break
                setattr(self, element[0], element[1])
                if element[1] is not None:
                    self[element[0]] = element[1]
        elif first_field is not None:
            self[class_fields[0].name] = first_field
    else:
        for field in class_fields:
            v = getattr(self, field.name)
            if v is not None:
                self[field.name] = v


ModelOutput.__post_init__ = _post_init_colo
# complete the hack


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def run_1d_hybrid_tp(model_name):
    # A simple net with two stacked nn.Linear
    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()
    rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    set_seed(1)
    with ColoInitContext(device=get_current_device()):
        model = model_builder(checkpoint=True)

    if 'bert' == model_name:
        parallel_action_list_row = [
            ParallelAction(priority=1,
                           compute_pattern=ComputePattern.TP1DRow_Linear,
                           parallel_mode=ParallelMode.PARALLEL_1D)
        ]
        spec_linear_row = TensorSpec(parallel_action_list_row)

        parallel_action_list_embedding_col = [
            ParallelAction(priority=1,
                           compute_pattern=ComputePattern.TP1DCol_Embedding,
                           parallel_mode=ParallelMode.PARALLEL_1D)
        ]
        spec_embedding_col = TensorSpec(parallel_action_list_embedding_col)

        parallel_action_list_embedding_row = [
            ParallelAction(priority=1,
                           compute_pattern=ComputePattern.TP1DRow_Embedding,
                           parallel_mode=ParallelMode.PARALLEL_1D)
        ]
        spec_embedding_row = TensorSpec(parallel_action_list_embedding_row)

        for name, p in model.colo_named_parameters():
            if not isinstance(p, ColoTensor):
                continue
            #print(name)
            # num_class = type_vocab_size = 2 | (8, 2)
            if 'classifier' in name and 'weight' in name:
                p.set_spec(spec_linear_row)
            # num_class = vocab_size = 30524 | (30524, 8)
            if 'word_embeddings' in name and 'weight' in name:
                p.set_spec(spec_embedding_row)
            # num_class = seq_len = 512 | (512, 8)
            if 'position_embeddings' in name and 'weight' in name:
                p.set_spec(spec_embedding_row)
            # num_class = type_vocab_size = 2 | (2, 8)
            if 'token_type_embeddings' in name and 'weight' in name:
                p.set_spec(spec_embedding_col)
    elif "simple_net" == model_name:
        parallel_action_list_row = [
            ParallelAction(priority=1,
                           compute_pattern=ComputePattern.TP1DRow_Linear,
                           parallel_mode=ParallelMode.PARALLEL_1D)
        ]
        spec_row = TensorSpec(parallel_action_list_row)

        parallel_action_list_col = [
            ParallelAction(priority=1,
                           compute_pattern=ComputePattern.TP1DCol_Linear,
                           parallel_mode=ParallelMode.PARALLEL_1D),
        ]
        spec_col = TensorSpec(parallel_action_list_col)

        parallel_action_list_classifier_col = [
            ParallelAction(priority=1,
                           compute_pattern=ComputePattern.TP1DCol_Linear,
                           parallel_mode=ParallelMode.PARALLEL_1D,
                           gather_out=False),
        ]
        spec_classifier_col = TensorSpec(parallel_action_list_classifier_col)

        parallel_action_list_embedding_col = [
            ParallelAction(priority=1,
                           compute_pattern=ComputePattern.TP1DCol_Embedding,
                           parallel_mode=ParallelMode.PARALLEL_1D)
        ]
        spec_embedding_col = TensorSpec(parallel_action_list_embedding_col)
        # A naive way to set spec for all weights in Linear
        for name, p in model.colo_named_parameters():
            if not isinstance(p, ColoTensor):
                continue
            if 'embed' in name and 'weight' in name:
                p.set_spec(spec_embedding_col)
            if 'proj1' in name and ('weight' in name or 'bias' in name):
                p.set_spec(spec_col)
            if 'proj2' in name and 'weight' in name:
                p.set_spec(spec_row)
            if 'classifier' in name and ('weight' in name or 'bias' in name):
                p.set_spec(spec_classifier_col)

    set_seed(1)
    if rank == 0:
        model_torch = model_builder(checkpoint=True)
        model_torch = model_torch.cuda()

    model = model.cuda()

    for i, (data, label) in enumerate(train_dataloader):
        data = data.to(get_current_device())
        label = label.to(get_current_device())

        torch.distributed.broadcast(data, 0, group=gpc.get_group(ParallelMode.PARALLEL_1D))
        torch.distributed.broadcast(label, 0, group=gpc.get_group(ParallelMode.PARALLEL_1D))

        # Bcast rank0 data to all processes
        if criterion:
            output = model(data)
            loss = criterion(output, label)
        else:
            output = model(data, label)
            loss = output

        # For reference
        if rank == 0:
            if criterion:
                output_torch = model_torch(data)
                loss_torch = criterion(output_torch, label)
            else:
                output_torch = model_torch(data, label)
                loss_torch = output_torch

        if rank == 0:
            # print(loss.torch_tensor().item())
            # print('loss torch', loss_torch.item())
            assert torch.allclose(loss.torch_tensor(), loss_torch, rtol=1e-2)

        loss.backward()

        if rank == 0:
            loss_torch.backward()
        if i > 5:
            break


# Test the overrided parameters() and named_parameters() member functions
def test_model_parameters():
    # build a module with 2 Linear, 4 parameters in total.
    class Net(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.fcs = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 2))
            self.extra_param = torch.nn.Parameter(torch.randn(2))

    with ColoInitContext(device=get_current_device()):
        model = Net()

    param_cnt = 0
    for name, p in model.named_parameters():
        param_cnt += 1
    assert param_cnt == 5

    for name, colo_p in model.colo_named_parameters():
        assert colo_p.is_model_data()

    param_cnt = 0
    for name, p in model.named_parameters(recurse=False):
        param_cnt += 1
    assert param_cnt == 1

    param_cnt = 0
    for p in model.fcs[0].parameters(recurse=False):
        param_cnt += 1
    assert param_cnt == 2


def test_colo_optimizer():
    get_components_func = non_distributed_component_funcs.get_callable('simple_net')
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()
    set_seed(1)
    with ColoInitContext(lazy_memory_allocate=False, device=get_current_device()):
        model = model_builder(checkpoint=True)

    colo_optimizer = ColoOptimizer(dict(model.named_parameters()), torch.optim.SGD, lr=0.1)
    for i, (data, label) in enumerate(train_dataloader):
        colo_optimizer.zero_grad()
        data = data.to(get_current_device())
        label = label.to(get_current_device())

        # Bcast rank0 data to all processes
        if criterion:
            output = model(data)
            loss = criterion(output, label)
        else:
            output = model(data, label)
            loss = output

        loss.backward()
        colo_optimizer.step()

        if i > 5:
            break


def run_1d_row_tp(model_name: str):
    # A simple net with two stacked nn.Linear
    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()
    rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    set_seed(1)
    with ColoInitContext(device=get_current_device()):
        model = model_builder(checkpoint=True)

    set_seed(1)
    if rank == 0:
        model_torch = model_builder(checkpoint=True)
        model_torch = model_torch.cuda()

    parallel_action_list = [
        ParallelAction(priority=1,
                       compute_pattern=ComputePattern.TP1DRow_Linear,
                       parallel_mode=ParallelMode.PARALLEL_1D)
    ]
    spec = TensorSpec(parallel_action_list)

    parallel_action_list_embedding_row = [
        ParallelAction(priority=1,
                       compute_pattern=ComputePattern.TP1DRow_Embedding,
                       parallel_mode=ParallelMode.PARALLEL_1D)
    ]
    spec_embedding_row = TensorSpec(parallel_action_list_embedding_row)

    # A naive way to set spec for all weights in Linear
    for name, p in model.colo_named_parameters():
        if not isinstance(p, ColoTensor):
            continue
        if 'weight' in name and 'LayerNorm' not in name and 'ln' not in name and 'embed' not in name:
            p.set_spec(spec)
        if 'embed' in name and 'weight' in name:
            p.set_spec(spec_embedding_row)

    model = model.cuda()

    for i, (data, label) in enumerate(train_dataloader):
        data = data.to(get_current_device())
        label = label.to(get_current_device())

        torch.distributed.broadcast(data, 0, group=gpc.get_group(ParallelMode.PARALLEL_1D))
        torch.distributed.broadcast(label, 0, group=gpc.get_group(ParallelMode.PARALLEL_1D))

        # Bcast rank0 data to all processes
        if criterion:
            output = model(data)
            loss = criterion(output, label)
        else:
            output = model(data, label)
            loss = output

        # For reference
        if rank == 0:
            if criterion:
                output_torch = model_torch(data)
                loss_torch = criterion(output_torch, label)
            else:
                output_torch = model_torch(data, label)
                loss_torch = output_torch

        if rank == 0:
            # print(loss.torch_tensor().item())
            # print('loss torch', loss_torch.item())
            assert torch.allclose(loss.torch_tensor(), loss_torch, rtol=1e-2)

        loss.backward()

        if rank == 0:
            loss_torch.backward()
        if i > 5:
            break


def run_dist(rank, world_size, port):
    config = dict(parallel=dict(tensor=dict(mode="1d", size=world_size),))
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    for name in ['simple_net']:
        run_1d_row_tp(name)
    for name in ['bert', 'simple_net']:    
        run_1d_hybrid_tp(name)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
#@parameterize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_model(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    # test_model_parameters()
    # test_colo_optimizer()
    test_model()
