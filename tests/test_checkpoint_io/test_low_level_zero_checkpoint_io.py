import torch
import torch.distributed as dist
from torchvision.models import resnet18
from utils import shared_tempdir

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
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
@parameterize('shard', [True, False])
def check_low_level_zero_checkpointIO(stage: int, shard: bool):
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
    with shared_tempdir() as tempdir:
        model_ckpt_path = f"{tempdir}/model"
        optimizer_ckpt_path = f"{tempdir}/optimizer"
        # lr scheduler is tested in test_torch_ddp_checkpoint_io.py and low level zero does not change it, we can skip it here
        booster.save_model(model, model_ckpt_path, shard=shard)
        if not shard:
            # TODO(ver217): optimizer checkpointing is not supported for sharded checkpoint
            booster.save_optimizer(optimizer, optimizer_ckpt_path)
        dist.barrier()

        new_model = resnet18()
        new_optimizer = HybridAdam((new_model.parameters()), lr=0.001)
        new_model, new_optimizer, _, _, _ = booster.boost(new_model, new_optimizer)

        booster.load_model(new_model, model_ckpt_path)
        check_state_dict_equal(model.state_dict(), new_model.state_dict(), False)
        if not shard:
            booster.load_optimizer(new_optimizer, optimizer_ckpt_path)
            check_state_dict_equal(optimizer.state_dict(), new_optimizer.state_dict(), False)


def run_dist(rank, world_size, port):
    colossalai.launch(config=(dict()), rank=rank, world_size=world_size, port=port, host='localhost')
    check_low_level_zero_checkpointIO()


@rerun_if_address_is_in_use()
def test_low_level_zero_checkpointIO():
    spawn(run_dist, 2)
