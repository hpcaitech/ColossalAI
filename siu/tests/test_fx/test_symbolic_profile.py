import pytest
import timm.models as tmm
import torch
import torchvision.models as tm
from zoo import tm_models, tmm_models

from siu._subclasses import MetaTensorMode
from siu.fx import symbolic_profile, symbolic_trace


def _check_gm_validity(gm: torch.fx.GraphModule):
    for node in gm.graph.nodes:
        assert len(node.meta['info'].global_ctx), f'In {gm.__class__.__name__}, {node} has empty global context.'


@pytest.mark.parametrize('m', tm_models)
def test_torchvision_profile(m):
    with MetaTensorMode():
        model = m()
        data = torch.rand(100, 3, 224, 224)
    meta_args = {
        "x": data,
    }
    gm = symbolic_trace(model, meta_args=meta_args)
    symbolic_profile(gm, data, verbose=True)
    _check_gm_validity(gm)


@pytest.mark.parametrize('m', tmm_models)
def test_timm_profile(m):
    with MetaTensorMode():
        model = m()
        data = torch.rand(100, 3, 224, 224)
    meta_args = {
        "x": data,
    }
    gm = symbolic_trace(model, meta_args=meta_args)
    symbolic_profile(gm, data, verbose=True)
    _check_gm_validity(gm)


if __name__ == "__main__":
    test_torchvision_profile(tm.mobilenet_v2)
    # test_timm_profile(tmm.dm_nfnet_f0)
