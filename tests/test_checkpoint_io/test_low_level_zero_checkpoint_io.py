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
from colossalai.zero import LowLevelZeroOptimizer


# stage 1 and 2 process the optimizer/mode the same way
# only test 2 is fine
@clear_cache_before_run()
@parameterize("stage", [2])
@parameterize("shard", [True, False])
@parameterize("offload", [False, True])
def check_low_level_zero_checkpointIO(stage: int, shard: bool, offload: bool):
    plugin = LowLevelZeroPlugin(stage=stage, max_norm=1.0, initial_scale=32, cpu_offload=offload)
    booster = Booster(plugin=plugin)
    model = resnet18()
    criterion = lambda x: x.mean()
    optimizer = HybridAdam((model.parameters()), lr=0.001)
    model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)

    x = torch.randn(1, 3, 224, 224, device="cuda")
    output = model(x)
    loss = criterion(output)
    booster.backward(loss, optimizer)
    optimizer.step()
    with shared_tempdir() as tempdir:
        model_ckpt_path = f"{tempdir}/model"
        optimizer_ckpt_path = f"{tempdir}/optimizer"
        # lr scheduler is tested in test_torch_ddp_checkpoint_io.py and low level zero does not change it, we can skip it here
        booster.save_model(model, model_ckpt_path, shard=shard)
        booster.save_optimizer(optimizer, optimizer_ckpt_path, shard=shard)

        dist.barrier()

        new_model = resnet18()
        new_optimizer = HybridAdam((new_model.parameters()), lr=0.001)
        new_model, new_optimizer, _, _, _ = booster.boost(new_model, new_optimizer)

        booster.load_model(new_model, model_ckpt_path)
        check_state_dict_equal(model.state_dict(), new_model.state_dict(), False)
        # check master weight
        assert isinstance(new_optimizer, LowLevelZeroOptimizer)
        working_param_id_set = set(id(p) for p in new_model.parameters())
        for p_id, master_param in new_optimizer._param_store.working_to_master_param.items():
            assert p_id in working_param_id_set
            working_param = new_optimizer._param_store.master_to_working_param[id(master_param)]
            padding = new_optimizer._param_store.get_param_padding_size(working_param)
            padded_param = torch.nn.functional.pad(working_param.data.view(-1), (0, padding))
            working_shard = padded_param.chunk(dist.get_world_size())[dist.get_rank()]
            assert torch.equal(
                working_shard, master_param.data.view(-1).to(dtype=padded_param.dtype, device=padded_param.device)
            )

        booster.load_optimizer(new_optimizer, optimizer_ckpt_path)
        check_state_dict_equal(optimizer.optim.state_dict(), new_optimizer.optim.state_dict(), False)
    torch.cuda.empty_cache()


def run_dist(rank, world_size, port):
    colossalai.launch(config=(dict()), rank=rank, world_size=world_size, port=port, host="localhost")
    check_low_level_zero_checkpointIO()
    torch.cuda.empty_cache()


@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_low_level_zero_checkpointIO():
    spawn(run_dist, 2)


if __name__ == "__main__":
    test_low_level_zero_checkpointIO()
