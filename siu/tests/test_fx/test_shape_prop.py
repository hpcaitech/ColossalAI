import pytest
import timm.models as tmm
import torch
import torchvision.models as tm
from zoo import tm_models, tmm_models

from siu._subclasses import MetaTensorMode
from siu.fx import symbolic_trace
from siu.fx.passes.shape_prop import shape_prop_pass
from siu.fx.symbolic_profile import register_shape_impl


def _check_gm_validity(gm: torch.fx.GraphModule):
    for node in gm.graph.nodes:
        assert node.meta['info'].outputs, f'In {gm.__class__.__name__}, {node} has no output shape.'
        if node.op in [
        # 'call_module',    # can apply to params
        # 'call_function',  # can apply to params
        # 'call_method',    # can apply to params
        ]:
            assert node.meta['info'].inputs, f'In {gm.__class__.__name__}, {node} has no input shape.'


@register_shape_impl(torch.nn.functional.linear)
def linear_impl(*args, **kwargs):
    assert True
    return torch.nn.functional.linear(*args, **kwargs)


@pytest.mark.parametrize('m', tm_models)
def test_torchvision_shape_prop(m):
    with MetaTensorMode():
        model = m()
        data = torch.rand(100, 3, 224, 224)
    meta_args = {
        "x": data,
    }
    gm = symbolic_trace(model, meta_args=meta_args)
    shape_prop_pass(gm, data)
    _check_gm_validity(gm)


@pytest.mark.parametrize('m', tmm_models)
def test_timm_shape_prop(m):
    with MetaTensorMode():
        model = m()
        data = torch.rand(100, 3, 224, 224)
    meta_args = {
        "x": data,
    }
    gm = symbolic_trace(model, meta_args=meta_args)
    shape_prop_pass(gm, data)
    _check_gm_validity(gm)


if __name__ == "__main__":
    test_torchvision_shape_prop(tm.resnet18)
    test_timm_shape_prop(tmm.vgg11)
