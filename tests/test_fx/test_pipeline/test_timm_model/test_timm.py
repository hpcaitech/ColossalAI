import pytest
import timm.models as tm
import torch
from timm_utils import split_model_and_compare_output


@pytest.mark.skip("balance split v2 is not ready")
def test_timm_models_without_control_flow():
    MODEL_LIST = [
        tm.resnest.resnest50d,
        tm.beit.beit_base_patch16_224,
        tm.cait.cait_s24_224,
        tm.convmixer.convmixer_768_32,
        tm.efficientnet.efficientnetv2_m,
        tm.resmlp_12_224,
        tm.vision_transformer.vit_base_patch16_224,
        tm.deit_base_distilled_patch16_224,
    ]

    data = torch.rand(2, 3, 224, 224)

    for model_cls in MODEL_LIST:
        model = model_cls()
        split_model_and_compare_output(model, data)


@pytest.mark.skip("balance split v2 is not ready")
def test_timm_models_with_control_flow():
    torch.backends.cudnn.deterministic = True

    MODEL_LIST_WITH_CONTROL_FLOW = [
        tm.convnext.convnext_base,
        tm.vgg.vgg11,
        tm.dpn.dpn68,
        tm.densenet.densenet121,
        tm.rexnet.rexnet_100,
        tm.swin_transformer.swin_base_patch4_window7_224,
    ]

    data = torch.rand(2, 3, 224, 224)

    meta_args = {"x": data.to("meta")}

    for model_cls in MODEL_LIST_WITH_CONTROL_FLOW:
        model = model_cls()
        split_model_and_compare_output(model, data, meta_args)


if __name__ == "__main__":
    test_timm_models_without_control_flow()
    test_timm_models_with_control_flow()
