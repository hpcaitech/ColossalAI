import pytest
import torch.nn as nn

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.context import MOE_CONTEXT
from colossalai.nn import CheckpointModule
from colossalai.nn.layer import MoeModule
from colossalai.testing import rerun_if_address_is_in_use, spawn

CONFIG = dict(fp16=dict(mode=None,),
              zero=dict(level=2),
              parallel=dict(pipeline=dict(size=1), tensor=dict(size=1, mode=None)))


class MoeModel(nn.Module):

    def __init__(self, checkpoint: bool = False):

        class TestSubModule(CheckpointModule):

            def __init__(self):
                super().__init__(checkpoint)
                expert_cls = nn.Linear
                expert_args_dict = dict(in_features=16, out_features=16)
                self.moe = MoeModule(dim_model=16,
                                     num_experts=8,
                                     use_residual=True,
                                     expert_cls=expert_cls,
                                     **expert_args_dict)
                self.proj = nn.Linear(16, 4)

            def _forward(self, x):
                x, y = self.moe(x)
                x = self.proj(x)
                return x, y

        super().__init__()
        self.test_embed = nn.Linear(4, 16)
        self.test_transform = TestSubModule()

    def forward(self, x):
        MOE_CONTEXT.reset_loss()

        x = self.test_embed(x)
        x, y = self.test_transform(x)

        MOE_CONTEXT.add_loss(y)
        return x


def run_moe_zero_init():
    model = MoeModel(checkpoint=True)
    plugin = LowLevelZeroPlugin(initial_scale=2**5)
    booster = Booster(plugin=plugin)
    model = booster.boost(model)[0]

    # assert local expert number
    assert len(model.module.test_transform.moe.moe_layer.experts.experts) == 8 // MOE_CONTEXT.world_size

    # for name, param in model.named_parameters():
    #     print(name, param)


def _run_dist(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    MOE_CONTEXT.setup(seed=42)
    run_moe_zero_init()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [2, 4])
@rerun_if_address_is_in_use()
def test_moe_zero_init(world_size):
    spawn(_run_dist, world_size)


if __name__ == '__main__':
    test_moe_zero_init(world_size=2)
