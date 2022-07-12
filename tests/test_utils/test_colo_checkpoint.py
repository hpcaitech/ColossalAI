from abc import ABC, abstractmethod
import os, shutil
import torch
import torch.nn as nn
import pytest
from functools import partial

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import MultiplicativeLR

import colossalai
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils.cuda import get_current_device
from colossalai.utils import free_port
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.tensor import ComputePattern, ComputeSpec, DistSpecManager, ShardSpec, ProcessGroup
from colossalai.nn.parallel.data_parallel import ColoDDP
from colossalai.utils.checkpoint import save_checkpoint, load_checkpoint
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR


class DummyDataGenerator(ABC):

    def __init__(self, length=10):
        self.length = length

    @abstractmethod
    def generate(self):
        pass

    def __iter__(self):
        self.step = 0
        return self

    def __next__(self):
        if self.step < self.length:
            self.step += 1
            return self.generate()
        else:
            raise StopIteration

    def __len__(self):
        return self.length


class DummyDataLoader(DummyDataGenerator):

    def __init__(self, batch_size, category, feature_size, length=10):
        super().__init__(length)
        self.batch_size = batch_size
        self.category = category
        self.feature_size = feature_size

    def generate(self):
        image_dict = {}
        image_dict['pixel_values'] = torch.rand(self.batch_size, self.feature_size, device=get_current_device()) * 2 - 1
        image_dict['label'] = torch.randint(self.category, (self.batch_size,),
                                            dtype=torch.int64,
                                            device=get_current_device())
        return image_dict


class MLP(nn.Module):

    def __init__(self, in_features, out_features, hidden_features=None):
        super().__init__()
        if hidden_features is None:
            hidden_features = out_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


def init_1d_row_for_linear_weight_spec(model, pg: ProcessGroup):
    spec = (ShardSpec([-1], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    with DistSpecManager.no_grad():
        for n, p in model.named_parameters():
            if 'weight' in n:
                p.set_process_group(pg)
                p.set_tensor_spec(*spec)


def check_param_equal(model, torch_model):
    for p, torch_p in zip(model.parameters(), torch_model.parameters()):
        assert torch.allclose(torch_p, p, rtol=1e-3, atol=1e-1)


def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
    else:
        raise ValueError("file {} is not a file or dir.".format(path))


def run_checkpoint(init_spec_func, use_ddp, use_mp_reload, test_scheduler, pg):
    num_epoch = 5
    warmup_epoch = 2

    batch = 3
    feature = 32
    category = 16

    with ColoInitContext(device=get_current_device()):
        model = MLP(feature, category)

    with ColoInitContext(device=get_current_device()):
        model_reload = MLP(feature, category)

    model = model.cuda()
    model_reload = model_reload.cuda()
    if use_ddp:
        model = ColoDDP(model, pg)
        model_reload = ColoDDP(model_reload, pg)

    init_spec_func(model, pg)
    if use_mp_reload:
        init_spec_func(model_reload, pg)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    optimizer_reload = torch.optim.Adam(model_reload.parameters(),
                                        lr=0.001,
                                        betas=(0.9, 0.999),
                                        eps=1e-08,
                                        weight_decay=0)

    lr_scheduler = None
    if test_scheduler == 'colossalai_cosine_warmup':
        lr_scheduler = CosineAnnealingWarmupLR(optimizer=optimizer, total_steps=num_epoch, warmup_steps=warmup_epoch)
        lr_scheduler_reload = CosineAnnealingWarmupLR(optimizer=optimizer_reload,
                                                      total_steps=num_epoch,
                                                      warmup_steps=warmup_epoch)
    elif test_scheduler == 'torch_cosine':
        lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=num_epoch)
        lr_scheduler_reload = CosineAnnealingLR(optimizer=optimizer_reload, T_max=num_epoch)
    elif test_scheduler == 'torch_lambda':
        lr_lambda = lambda epoch: 0.95
        lr_scheduler = MultiplicativeLR(optimizer=optimizer, lr_lambda=lr_lambda)
        lr_scheduler_reload = MultiplicativeLR(optimizer=optimizer_reload, lr_lambda=lr_lambda)
    else:
        raise TypeError(f"{test_scheduler} is invalid")

    save_checkpoint('./checkpoint', 0, model, optimizer, lr_scheduler)
    dist.barrier()
    load_checkpoint('./checkpoint', 0, model_reload, optimizer_reload, lr_scheduler_reload)

    # Since model is sharded, we merge them before param checking.
    for p in model.parameters():
        p.to_replicate_()

    for p in model_reload.parameters():
        p.to_replicate_()

    check_param_equal(model, model_reload)


def run_dist(rank, world_size, port, use_ddp, use_mp_reload, test_scheduler):
    if use_ddp and world_size == 1:
        return
    tp_world_size = world_size // 2 if use_ddp else world_size
    config = dict(parallel=dict(tensor=dict(mode="1d", size=tp_world_size),))
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    pg = ProcessGroup(tp_degree=world_size)
    run_checkpoint(init_1d_row_for_linear_weight_spec, use_ddp, use_mp_reload, test_scheduler=test_scheduler, pg=pg)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 2])
@pytest.mark.parametrize('use_ddp', [True, False])
@pytest.mark.parametrize('use_mp_reload', [True, False])
@pytest.mark.parametrize('test_scheduler', ['colossalai_cosine_warmup', 'torch_cosine', 'torch_lambda'])
@rerun_if_address_is_in_use()
def test_checkpoint(world_size, use_ddp, use_mp_reload, test_scheduler):
    if not os.path.isdir('./checkpoint'):
        os.mkdir('./checkpoint')
    run_func = partial(run_dist,
                       world_size=world_size,
                       port=free_port(),
                       use_ddp=use_ddp,
                       use_mp_reload=use_mp_reload,
                       test_scheduler=test_scheduler)
    mp.spawn(run_func, nprocs=world_size)
    remove('./checkpoint')


if __name__ == '__main__':
    test_checkpoint(2, True, False, "torch_cosine")
