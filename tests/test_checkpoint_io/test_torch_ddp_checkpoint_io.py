import tempfile

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD
from torchvision.models import resnet18

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.booster.plugin.torch_ddp_plugin import TorchDDPCheckpointIO
from colossalai.interface import OptimizerWrapper
from colossalai.testing import clear_cache_before_run, rerun_if_address_is_in_use, spawn


def check_torch_ddp_checkpointIO():
    plugin = TorchDDPPlugin()
    booster = Booster(plugin=plugin)
    model = resnet18()
    criterion = lambda x: x.mean()
    optimizer = SGD((model.parameters()), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion, lr_scheduler=scheduler)

    assert isinstance(model.module, DDP)
    assert isinstance(optimizer, OptimizerWrapper)

    x = torch.randn(4, 3, 224, 224)
    x = x.to('cuda')
    output = model(x)
    loss = criterion(output)
    booster.backward(loss, optimizer)
    optimizer.clip_grad_by_norm(1.0)
    optimizer.step()
    scheduler.step()

    optimizer_ckpt_tempfile = tempfile.NamedTemporaryFile()
    lr_scheduler_ckpt_tempfile = tempfile.NamedTemporaryFile()
    ckpt_io = TorchDDPCheckpointIO()
    ckpt_io.save_optimizer(optimizer, optimizer_ckpt_tempfile.name)
    ckpt_io.save_lr_scheduler(scheduler, lr_scheduler_ckpt_tempfile.name)

    if ckpt_io.coordinator.is_master():
        new_model = resnet18()
        new_optimizer = SGD((new_model.parameters()), lr=0.001)
        new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=1, gamma=0.1)
        _, new_optimizer, _, _, new_scheduler = booster.boost(new_model, new_optimizer, lr_scheduler=new_scheduler)

        ckpt_io.load_optimizer(new_optimizer, optimizer_ckpt_tempfile.name)
        recursive_check(optimizer.state_dict(), new_optimizer.state_dict())

        ckpt_io.load_lr_scheduler(new_scheduler, lr_scheduler_ckpt_tempfile.name)
        recursive_check(scheduler.state_dict(), new_scheduler.state_dict())


def run_dist(rank, world_size, port):
    colossalai.launch(config=(dict()), rank=rank, world_size=world_size, port=port, host='localhost')
    check_torch_ddp_checkpointIO()


@rerun_if_address_is_in_use()
def test_torch_ddp_checkpointIO():
    spawn(run_dist, 2)


def recursive_check(d1, d2):
    for k, v in d1.items():
        if isinstance(v, dict):
            recursive_check(v, d2[k])
        elif isinstance(v, list):
            for i in range(len(v)):
                if isinstance(v[i], torch.Tensor):
                    v[i] = v[i].to('cpu')
                    d2[k][i] = d2[k][i].to('cpu')
                    if not torch.equal(v[i], d2[k][i]):
                        raise AssertionError
                elif v[i] != d2[k][i]:
                    assert v[i] == d2[k][i]

        elif isinstance(v, torch.Tensor):
            v = v.to('cpu')
            d2[k] = d2[k].to('cpu')
            if not torch.equal(v, d2[k]):
                raise AssertionError
        elif not v == d2[k]:
            raise AssertionError
