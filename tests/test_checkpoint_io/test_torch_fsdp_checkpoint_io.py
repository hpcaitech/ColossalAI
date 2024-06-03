import pytest
import torch
from packaging import version
from torch.optim import SGD
from torchvision.models import resnet18
from utils import shared_tempdir

import colossalai
from colossalai.booster import Booster

if version.parse(torch.__version__) >= version.parse("1.12.0"):
    from colossalai.booster.plugin import TorchFSDPPlugin
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from colossalai.testing import rerun_if_address_is_in_use, spawn


def compare_nested_dict(dict1, dict2):
    for key in dict1:
        if key in dict2:
            if type(dict1[key]) is dict:
                assert type(dict2[key]) is dict
                diff = compare_nested_dict(dict1[key], dict2[key])
                if not diff:
                    return diff
            elif type(dict1[key]) is list:
                assert type(dict2[key]) is list
                for i, val in enumerate(dict1[key]):
                    if isinstance(val, torch.Tensor):
                        if not torch.equal(dict1[key][i], dict2[key][i]):
                            return False
                    elif val != dict2[key][i]:
                        return False
            elif type(dict1[key]) is torch.Tensor:
                assert type(dict2[key]) is torch.Tensor
                if not torch.equal(dict1[key], dict2[key]):
                    return False
            else:
                if dict1[key] != dict2[key]:
                    return False
        else:
            return False
    return True


def check_torch_fsdp_ckpt():
    model = resnet18()
    plugin = TorchFSDPPlugin()
    booster = Booster(plugin=plugin)
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = lambda x: x.mean()
    fsdp_model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)

    inputs = torch.randn(4, 3, 224, 224)
    outputs = None

    def run_model():
        nonlocal outputs
        outputs = fsdp_model(inputs)
        optimizer.zero_grad()
        criterion(outputs).backward()
        optimizer.step()

    with shared_tempdir() as tempdir:
        model_ckpt_path = f"{tempdir}/model"
        optim_ckpt_path = f"{tempdir}/optimizer"

        run_model()

        booster.save_model(fsdp_model, model_ckpt_path, shard=False)
        booster.save_optimizer(optimizer, optim_ckpt_path, shard=False)

        full_msd = fsdp_model.state_dict()
        # full_osd = FSDP.full_optim_state_dict(fsdp_model, optimizer)
        sharded_osd = optimizer.state_dict()
        import copy

        sharded_osd = copy.deepcopy(sharded_osd)

        run_model()

        full_msd_updated = fsdp_model.state_dict()
        # full_osd_updated = FSDP.full_optim_state_dict(fsdp_model, optimizer, rank0_only=True)
        sharded_osd_updated = optimizer.state_dict()

        assert not compare_nested_dict(sharded_osd, sharded_osd_updated)
        assert not compare_nested_dict(full_msd_updated, full_msd)
        outputs_first = fsdp_model(inputs)
        assert criterion(outputs_first) != criterion(outputs)

        booster.load_model(fsdp_model, model_ckpt_path)
        booster.load_optimizer(optimizer, optim_ckpt_path)

        full_msd_restore = fsdp_model.state_dict()
        # full_osd_restore = FSDP.full_optim_state_dict(fsdp_model, optimizer, rank0_only=True)
        sharded_osd_restore = optimizer.state_dict()

        assert compare_nested_dict(sharded_osd, sharded_osd_restore)
        assert compare_nested_dict(full_msd_restore, full_msd)
        outputs_sec = fsdp_model(inputs)
        assert criterion(outputs_sec) == criterion(outputs)

    with shared_tempdir() as tempdir:
        model_ckpt_path = f"{tempdir}/model"
        optim_ckpt_path = f"{tempdir}/optimizer"

        run_model()

        booster.save_model(fsdp_model, model_ckpt_path, shard=True)
        booster.save_optimizer(optimizer, optim_ckpt_path, shard=True)

        full_msd = fsdp_model.unwrap().state_dict()
        full_osd = FSDP.full_optim_state_dict(optimizer.unwrap_model().unwrap(), optim=optimizer)

        import copy

        sharded_osd = copy.deepcopy(full_osd)

        run_model()

        full_msd_updated = fsdp_model.unwrap().state_dict()
        full_osd_updated = FSDP.full_optim_state_dict(optimizer.unwrap_model().unwrap(), optim=optimizer)

        # cost much time led to timeout
        # assert not compare_nested_dict(full_osd_updated, sharded_osd)
        # assert not compare_nested_dict(full_msd_updated, full_msd)
        outputs_first = fsdp_model(inputs)
        assert criterion(outputs_first) != criterion(outputs)

        booster.load_model(fsdp_model, model_ckpt_path)
        booster.load_optimizer(optimizer, optim_ckpt_path)

        full_msd_restore = fsdp_model.unwrap().state_dict()
        sharded_osd_restore = FSDP.full_optim_state_dict(optimizer.unwrap_model().unwrap(), optim=optimizer)

        assert compare_nested_dict(sharded_osd, sharded_osd_restore)
        assert compare_nested_dict(full_msd_restore, full_msd)
        outputs_sec = fsdp_model(inputs)
        assert criterion(outputs_sec) == criterion(outputs)


def run_dist(rank, world_size, port):
    # init dist env
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")
    check_torch_fsdp_ckpt()


@pytest.mark.skipif(version.parse(torch.__version__) < version.parse("1.12.0"), reason="requires torch1.12 or higher")
@rerun_if_address_is_in_use()
def test_torch_fsdp_ckpt():
    spawn(run_dist, 2)
