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
    origin_model = resnet34(num_classes=10)
    origin_param_dict = dict(origin_model.named_parameters())
    ctx = LazyInitContext()
    with ctx:
        model = resnet34(num_classes=10)
    for param in model.parameters():
        assert param.is_meta
    for buffer in model.buffers():
        assert buffer.is_meta
    ctx.lazy_init_parameters(model)

    for name, param in model.named_parameters():
        assert not param.is_meta, name

    # for buffer in model.buffers():
    #     assert not buffer.is_meta


if __name__ == '__main__':
    test_lazy_init()
