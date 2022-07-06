import torch
import pytest
try:
    import torchvision.models as tm
except:
    pass
from colossalai.fx import ColoTracer
from torch.fx import GraphModule


@pytest.mark.skip('skip as torchvision is required')
def test_torchvision_models():
    MODEL_LIST = [
        tm.vgg11, tm.resnet18, tm.densenet121, tm.mobilenet_v3_small, tm.resnext50_32x4d, tm.wide_resnet50_2,
        tm.regnet_x_16gf, tm.vit_b_16, tm.convnext_small, tm.mnasnet0_5, tm.efficientnet_b0
    ]

    torch.backends.cudnn.deterministic = True

    tracer = ColoTracer()
    data = torch.rand(2, 3, 224, 224)

    for model_cls in MODEL_LIST:
        if model_cls in [tm.convnext_small, tm.efficientnet_b0]:
            # remove the impact of randomicity
            model = model_cls(stochastic_depth_prob=0)
        else:
            model = model_cls()

        graph = tracer.trace(root=model)

        gm = GraphModule(model, graph, model.__class__.__name__)
        gm.recompile()

        model.eval()
        gm.eval()

        with torch.no_grad():
            fx_out = gm(data)
            non_fx_out = model(data)
        assert torch.allclose(
            fx_out, non_fx_out), f'{model.__class__.__name__} has inconsistent outputs, {fx_out} vs {non_fx_out}'


if __name__ == '__main__':
    test_torchvision_models()
