import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD
from torchvision.models import resnet18
from utils import shared_tempdir

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.interface import OptimizerWrapper
from colossalai.testing import check_state_dict_equal, parameterize, rerun_if_address_is_in_use, spawn


@parameterize('shard', [True, False])
@parameterize('size_per_shard', [16, 128])
def check_torch_ddp_checkpointIO(shard: bool, size_per_shard: int):
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

    with shared_tempdir() as tempdir:
        model_ckpt_path = f"{tempdir}/model"
        optimizer_ckpt_path = f"{tempdir}/optimizer"
        lr_scheduler_ckpt_path = f"{tempdir}/lr_scheduler"
        booster.save_model(model, model_ckpt_path, shard=shard, size_per_shard=size_per_shard)
        booster.save_optimizer(optimizer, optimizer_ckpt_path, shard=shard, size_per_shard=size_per_shard)
        booster.save_lr_scheduler(scheduler, lr_scheduler_ckpt_path)
        dist.barrier()

        new_model = resnet18()
        new_optimizer = SGD((new_model.parameters()), lr=0.001)
        new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=1, gamma=0.1)
        new_model, new_optimizer, _, _, new_scheduler = booster.boost(new_model,
                                                                      new_optimizer,
                                                                      lr_scheduler=new_scheduler)

        booster.load_model(new_model, model_ckpt_path)
        check_state_dict_equal(model.state_dict(), new_model.state_dict(), False)

        booster.load_optimizer(new_optimizer, optimizer_ckpt_path)
        check_state_dict_equal(optimizer.state_dict(), new_optimizer.state_dict(), False)
        booster.load_lr_scheduler(new_scheduler, lr_scheduler_ckpt_path)
        check_state_dict_equal(scheduler.state_dict(), new_scheduler.state_dict(), False)


def run_dist(rank, world_size, port):
    colossalai.launch(config=(dict()), rank=rank, world_size=world_size, port=port, host='localhost')
    check_torch_ddp_checkpointIO()


@rerun_if_address_is_in_use()
def test_torch_ddp_checkpointIO():
    spawn(run_dist, 2)
