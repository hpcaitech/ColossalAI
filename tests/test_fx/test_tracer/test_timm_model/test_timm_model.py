import pytest
import timm.models as tm
import torch

from colossalai.fx import symbolic_trace
from tests.kit.model_zoo import model_zoo


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
        fx_out = gm(**data)
        non_fx_out = model(**data)

    # compare output
    if isinstance(fx_out, tuple):
        # some models produce tuple as output
        for v1, v2 in zip(fx_out, non_fx_out):
            assert torch.allclose(v1, v2), f'{model.__class__.__name__} has inconsistent outputs, {v1} vs {v2}'
    else:
        assert torch.allclose(
            fx_out, non_fx_out,
            atol=1e-5), f'{model.__class__.__name__} has inconsistent outputs, {fx_out} vs {non_fx_out}'


def test_timm_models():
    torch.backends.cudnn.deterministic = True

    sub_model_zoo = model_zoo.get_sub_registry('timm')

    for name, (model_fn, data_gen_fn, attribute) in sub_model_zoo.items():
        data = data_gen_fn()
        if attribute.has_control_flow:
            meta_args = {k: v.to('meta') for k, v in data.items()}
        else:
            meta_args = None

        trace_and_compare(model_fn, data, meta_args)


if __name__ == '__main__':
    test_timm_models()
