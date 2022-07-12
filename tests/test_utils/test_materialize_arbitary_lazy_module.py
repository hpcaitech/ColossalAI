import torch
from colossalai.utils.model.lazy_init_context import LazyInitContext
from torchvision.models import resnet34
import random
import numpy as np

MANUAL_SEED = 0
random.seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)


class MLP(torch.nn.Module):

    def __init__(self, dim: int = 4):
        super().__init__()
        intermediate_dim = dim * 4
        self.dense_1 = torch.nn.Linear(dim, intermediate_dim)
        self.activation = torch.nn.GELU()
        self.dense_2 = torch.nn.Linear(intermediate_dim, dim)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


def test_lazy_init():
    cpu_rng_state = torch.get_rng_state()
    origin_model = MLP()
    origin_param_dict = dict(origin_model.named_parameters())
    torch.set_rng_state(cpu_rng_state)
    ctx = LazyInitContext()
    with ctx:
        model = MLP()
    for param in model.parameters():
        assert param.is_meta
    for buffer in model.buffers():
        assert buffer.is_meta
    for module in model.children():
        ctx.lazy_init_parameters(module)
        for param in module.parameters():
            assert not param.is_meta
        for buffer in module.buffers():
            assert not buffer.is_meta
    param_dict = dict(model.named_parameters())
    for key in origin_param_dict.keys():
        assert origin_param_dict[key].data.equal(param_dict[key].data)


if __name__ == '__main__':
    test_lazy_init()
