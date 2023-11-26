import pytest
import timm.models as tmm
import torch
import torchvision.models as tm

from colossalai.fx._compatibility import is_compatible_with_meta

if is_compatible_with_meta():
    from colossalai.fx import meta_trace

from colossalai.testing import clear_cache_before_run

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
    tmm.swin_transformer.swin_base_patch4_window7_224,
]


@pytest.mark.skipif(not is_compatible_with_meta(), reason="torch version is lower than 1.12.0")
@clear_cache_before_run()
def test_torchvision_models_trace():
    for m in tm_models:
        model = m()
        data = torch.rand(1000, 3, 224, 224, device="meta")
        meta_trace(model, torch.device("cpu"), data)


@pytest.mark.skipif(not is_compatible_with_meta(), reason="torch version is lower than 1.12.0")
@clear_cache_before_run()
def test_timm_models_trace():
    for m in tmm_models:
        model = m()
        data = torch.rand(1000, 3, 224, 224, device="meta")
        meta_trace(model, torch.device("cpu"), data)


if __name__ == "__main__":
    test_torchvision_models_trace()
    test_timm_models_trace()
