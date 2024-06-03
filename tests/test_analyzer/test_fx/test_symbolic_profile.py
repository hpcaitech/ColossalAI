import pytest
import torch
from packaging import version

from colossalai.testing.utils import clear_cache_before_run, parameterize
from tests.test_analyzer.test_fx.zoo import tm_models, tmm_models

try:
    from colossalai._analyzer._subclasses import MetaTensorMode
    from colossalai._analyzer.fx import symbolic_profile, symbolic_trace
except:
    pass


def _check_gm_validity(gm: torch.fx.GraphModule):
    for node in gm.graph.nodes:
        assert len(node.meta["info"].global_ctx), f"In {gm.__class__.__name__}, {node} has empty global context."


@pytest.mark.skipif(version.parse(torch.__version__) < version.parse("1.12.0"), reason="torch version < 12")
@clear_cache_before_run()
@parameterize("m", tm_models)
def test_torchvision_profile(m, verbose=False, bias_addition_split=False):
    with MetaTensorMode():
        model = m()
        data = torch.rand(8, 3, 224, 224)
    meta_args = {
        "x": data,
    }
    gm = symbolic_trace(model, meta_args=meta_args, bias_addition_split=bias_addition_split)
    symbolic_profile(gm, data, verbose=verbose)
    _check_gm_validity(gm)


@pytest.mark.skipif(version.parse(torch.__version__) < version.parse("1.12.0"), reason="torch version < 12")
@clear_cache_before_run()
@parameterize("m", tmm_models)
def test_timm_profile(m, verbose=False, bias_addition_split=False):
    with MetaTensorMode():
        model = m()
        data = torch.rand(8, 3, 224, 224)
    meta_args = {
        "x": data,
    }
    gm = symbolic_trace(model, meta_args=meta_args, bias_addition_split=bias_addition_split)
    symbolic_profile(gm, data, verbose=verbose)
    _check_gm_validity(gm)


if __name__ == "__main__":
    test_torchvision_profile()
    test_timm_profile()
