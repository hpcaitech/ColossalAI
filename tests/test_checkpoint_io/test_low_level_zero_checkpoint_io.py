import tempfile

import pytest
import torch
from torchvision.models import resnet18

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.booster.plugin.low_level_zero_plugin import LowLevelZeroCheckpointIO
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import (
    check_state_dict_equal,
    clear_cache_before_run,
    parameterize,
    rerun_if_address_is_in_use,
    spawn,
)


@clear_cache_before_run()
@parameterize('stage', [2])
def check_low_level_zero_checkpointIO(stage: int):
    plugin = LowLevelZeroPlugin(stage=stage, max_norm=1.0, initial_scale=32)
    booster = Booster(plugin=plugin)
    model = resnet18()
    criterion = lambda x: x.mean()
    optimizer = HybridAdam((model.parameters()), lr=0.001)
    model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)

    x = torch.randn(4, 3, 224, 224)
    x = x.to('cuda')
    output = model(x)
    loss = criterion(output)
    booster.backward(loss, optimizer)
    optimizer.step()

    optimizer_ckpt_tempfile = tempfile.NamedTemporaryFile()
    ckpt_io = LowLevelZeroCheckpointIO()
    ckpt_io.save_optimizer(optimizer, optimizer_ckpt_tempfile.name)

    if ckpt_io.coordinator.is_master():
        new_model = resnet18()
        new_optimizer = HybridAdam((new_model.parameters()), lr=0.001)
        _, new_optimizer, _, _, _ = booster.boost(new_model, new_optimizer)
        ckpt_io.load_optimizer(new_optimizer, optimizer_ckpt_tempfile.name)
        check_state_dict_equal(optimizer.state_dict(), new_optimizer.state_dict(), False)


def run_dist(rank, world_size, port):
    colossalai.launch(config=(dict()), rank=rank, world_size=world_size, port=port, host='localhost')
    check_low_level_zero_checkpointIO()


@rerun_if_address_is_in_use()
def test_low_level_zero_checkpointIO():
    spawn(run_dist, 2)
