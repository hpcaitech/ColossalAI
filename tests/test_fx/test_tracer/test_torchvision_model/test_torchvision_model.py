import torch

from colossalai.fx import symbolic_trace
from tests.kit.model_zoo import model_zoo


def test_torchvision_models():
    torch.backends.cudnn.deterministic = True
    tv_sub_registry = model_zoo.get_sub_registry('torchvision')

    for name, (model_fn, data_gen_fn, model_attribute) in tv_sub_registry.items():
        data = data_gen_fn()

        if model_attribute is not None and model_attribute.has_stochastic_depth_prob:
            model = model_fn(stochastic_depth_prob=0)
        else:
            model = model_fn()

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
