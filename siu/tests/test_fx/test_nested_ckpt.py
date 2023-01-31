import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from siu.fx import symbolic_trace


class MyModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.a = nn.Linear(10, 10)
        self.b = nn.Linear(10, 10)
        self.c = nn.Linear(10, 10)
        self.d = nn.Linear(10, 10)
        self.e = nn.Linear(10, 10)

    def checkpoint_0(self, x):
        return checkpoint(self.checkpoint_0_0, x) + checkpoint(self.checkpoint_0_1, x) + self.e(x)

    def checkpoint_0_0(self, x):
        return checkpoint(self.checkpoint_0_0_0, x) + checkpoint(self.checkpoint_0_0_1, x)

    def checkpoint_0_0_0(self, x):
        return self.a(x) + checkpoint(self.checkpoint_0_0_0_0, x, use_reentrant=False)

    def checkpoint_0_0_0_0(self, x):
        return self.b(x)

    def checkpoint_0_0_1(self, x):
        return self.b(x) + self.c(x)

    def checkpoint_0_1(self, x):
        return self.d(x)

    def forward(self, x):
        return checkpoint(self.checkpoint_0, x)


def test_nested_ckpt():
    model = MyModule()
    x = torch.rand(10, 10)
    gm = symbolic_trace(model, meta_args={'x': x}, trace_act_ckpt=True)
    assert torch.allclose(gm(x), model(x)), "The traced model should generate the same output as the original model."
    for ckpt_def in filter(lambda s: s.startswith('checkpoint'), dir(model)):
        assert ckpt_def in gm.code, f"Checkpoint {ckpt_def} should be in the traced code.\n Traced code = {gm.code}"


if __name__ == "__main__":
    test_nested_ckpt()
