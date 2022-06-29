import torch
from colossalai.utils.model.lazy_init_context import LazyInitContext
from torchvision.models import resnet34


def test_lazy_init():
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


if __name__ == '__main__':
    test_lazy_init()
