from torch.utils.checkpoint import checkpoint
import torch
from colossalai.utils import ColoInitContext


def test_checkpoint():
    class ModelwithCheckPoint(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(32, 32)
            self.proj_ckpt = torch.nn.Linear(32, 2)

        def forward(self, x):
            x = self.proj(x)
            x = checkpoint(self.proj_ckpt, x)
            assert x.requires_grad
            return x

    with ColoInitContext():
        model = ModelwithCheckPoint()

    x = torch.rand(2,32)
    out = model(x)
    loss = torch.sum(out)
    loss.backward()
    assert loss.requires_grad


if __name__ == '__main__':
    test_checkpoint()