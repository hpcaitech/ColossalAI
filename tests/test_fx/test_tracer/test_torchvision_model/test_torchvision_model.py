import torch
import torchvision
import torchvision.models as tm
from packaging import version

from colossalai.fx import symbolic_trace


def test_torchvision_models():
    MODEL_LIST = [
        tm.vgg11, tm.resnet18, tm.densenet121, tm.mobilenet_v3_small, tm.resnext50_32x4d, tm.wide_resnet50_2,
        tm.regnet_x_16gf, tm.mnasnet0_5, tm.efficientnet_b0
    ]

    RANDOMIZED_MODELS = [tm.efficientnet_b0]

    if version.parse(torchvision.__version__) >= version.parse('0.12.0'):
        MODEL_LIST.extend([tm.vit_b_16, tm.convnext_small])
        RANDOMIZED_MODELS.append(tm.convnext_small)

    torch.backends.cudnn.deterministic = True

    data = torch.rand(2, 3, 224, 224)

    for model_cls in MODEL_LIST:
        if model_cls in RANDOMIZED_MODELS:
            # remove the impact of randomicity
            model = model_cls(stochastic_depth_prob=0)
        else:
            model = model_cls()

        gm = symbolic_trace(model)

        model.eval()
        gm.eval()

        with torch.no_grad():
            fx_out = gm(data)
            non_fx_out = model(data)
        assert torch.allclose(
            fx_out, non_fx_out), f'{model.__class__.__name__} has inconsistent outputs, {fx_out} vs {non_fx_out}'


if __name__ == '__main__':
    test_torchvision_models()
