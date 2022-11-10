import pytest
import timm.models as tm
import torch

from colossalai.fx import symbolic_trace


def trace_and_compare(model_cls, data, meta_args=None):
    # trace
    model = model_cls()

    # convert to eval for inference
    # it is important to set it to eval mode before tracing
    # without this statement, the torch.nn.functional.batch_norm will always be in training mode
    model.eval()

    gm = symbolic_trace(model, meta_args=meta_args)

    # run forward
    with torch.no_grad():
        fx_out = gm(data)
        non_fx_out = model(data)

    # compare output
    if isinstance(fx_out, tuple):
        # some models produce tuple as output
        for v1, v2 in zip(fx_out, non_fx_out):
            assert torch.allclose(v1, v2), f'{model.__class__.__name__} has inconsistent outputs, {v1} vs {v2}'
    else:
        assert torch.allclose(
            fx_out, non_fx_out,
            atol=1e-5), f'{model.__class__.__name__} has inconsistent outputs, {fx_out} vs {non_fx_out}'


def test_timm_models_without_control_flow():
    torch.backends.cudnn.deterministic = True

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
        trace_and_compare(model_cls, data)


def test_timm_models_with_control_flow():
    torch.backends.cudnn.deterministic = True

    MODEL_LIST_WITH_CONTROL_FLOW = [
        tm.convnext.convnext_base, tm.vgg.vgg11, tm.dpn.dpn68, tm.densenet.densenet121, tm.rexnet.rexnet_100,
        tm.swin_transformer.swin_base_patch4_window7_224
    ]

    data = torch.rand(2, 3, 224, 224)

    meta_args = {'x': data.to('meta')}

    for model_cls in MODEL_LIST_WITH_CONTROL_FLOW:
        trace_and_compare(model_cls, data, meta_args)


if __name__ == '__main__':
    test_timm_models_with_control_flow()
    test_timm_models_without_control_flow()
