import torchvision.models as tm
import timm.models as tmm
import torch
from colossalai.fx.profiler import MetaTensor

import pytest

try:
    meta_lib = torch.library.Library("aten", "IMPL", "Meta")
    incompatible = False  # version > 1.12.0
except:
    incompatible = True


tm_models = [
    tm.vgg11, 
    tm.resnet18, 
    tm.densenet121, 
    tm.mobilenet_v3_small, 
    tm.resnext50_32x4d, 
    tm.wide_resnet50_2,
    tm.regnet_x_16gf, 
    tm.mnasnet0_5, 
    tm.efficientnet_b0,
]


tmm_models = [
    tmm.resnest.resnest50d,
    tmm.beit.beit_base_patch16_224,
    tmm.cait.cait_s24_224,
    tmm.efficientnet.efficientnetv2_m,
    tmm.resmlp_12_224,
    tmm.vision_transformer.vit_base_patch16_224,
    tmm.deit_base_distilled_patch16_224,
    tmm.convnext.convnext_base, 
    tmm.vgg.vgg11, 
    tmm.dpn.dpn68, 
    tmm.densenet.densenet121, 
    tmm.rexnet.rexnet_100,
    tmm.swin_transformer.swin_base_patch4_window7_224
]


@pytest.mark.skipif(incompatible, reason='torch version is lower than 1.12.0')
def test_torchvision_models():
    for m in tm_models:
        model = m().to('meta')
        data = torch.rand(1000, 3, 224, 224, device='meta')
        model(MetaTensor(data)).sum().backward()


@pytest.mark.skipif(incompatible, reason='torch version is lower than 1.12.0')
def test_timm_models():
    for m in tmm_models:
        model = m().to('meta')
        data = torch.rand(1000, 3, 224, 224, device='meta')
        model(MetaTensor(data)).sum().backward()


if __name__ == '__main__':
    test_torchvision_models()
    test_timm_models()
