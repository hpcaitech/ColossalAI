import torch
from colossalai.utils.model.lazy_init_context import LazyInitContext
from torchvision.models import resnet34
import random
import numpy as np

MANUAL_SEED = 0
random.seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)


def test_lazy_init_with_meta():
    ctx = LazyInitContext(to_meta=True)
    with ctx:
        model = resnet34(num_classes=10)

    for param in model.parameters():
        assert param.is_meta
    for buffer in model.buffers():
        assert buffer.is_meta

    ctx.lazy_init_parameters(model)

    for name, param in model.named_parameters():
        assert not param.is_meta, name

    for buffer in model.buffers():
        assert not buffer.is_meta


def test_lazy_init_without_meta():
    ctx = LazyInitContext(to_meta=False)
    with ctx:
        model = resnet34(num_classes=10)

    for param in model.parameters():
        assert not param.is_meta
    for buffer in model.buffers():
        assert not buffer.is_meta

    conv1_weight_before_init = model.conv1.weight.clone()
    ctx.lazy_init_parameters(model)
    conv1_weight_after_init = model.conv1.weight.clone()

    assert not torch.allclose(conv1_weight_after_init, conv1_weight_before_init)


if __name__ == '__main__':
    test_lazy_init_with_meta()
    test_lazy_init_without_meta()
