import torch
from colossalai.utils.model.lazy_init_context import LazyInitContext
from torchvision.models import resnet34
import random
import numpy as np

MANUAL_SEED = 0
random.seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)


def test_lazy_init():
    cpu_rng_state = torch.get_rng_state()
    origin_model = resnet34(num_classes=10)
    origin_param_dict = dict(origin_model.named_parameters())
    torch.set_rng_state(cpu_rng_state)
    ctx = LazyInitContext()
    with ctx:
        model = resnet34(num_classes=10)
    for param in model.parameters():
        assert param.is_meta
    for buffer in model.buffers():
        assert buffer.is_meta
    ctx.lazy_init_parameters(model)
    for param in model.parameters():
        assert not param.is_meta
    for buffer in model.buffers():
        assert not buffer.is_meta
    param_dict = dict(model.named_parameters())
    for key in origin_param_dict.keys():
        assert origin_param_dict[key].data.equal(param_dict[key].data)


if __name__ == '__main__':
    test_lazy_init()
