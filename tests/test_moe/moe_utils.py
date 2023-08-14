import torch.nn as nn

from colossalai.context import MOE_CONTEXT
from colossalai.nn import CheckpointModule
from colossalai.nn.layer import MoeModule


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
