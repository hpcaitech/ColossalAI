import pytest
import torch

from colossalai._analyzer.fx import symbolic_trace
from colossalai.testing import clear_cache_before_run
from tests.kit.model_zoo import model_zoo

BATCH = 2
SHAPE = 10


def trace_and_compare(model_cls, data, output_transform_fn, meta_args=None):
    # trace
    model = model_cls()

    # convert to eval for inference
    # it is important to set it to eval mode before tracing
    # without this statement, the torch.nn.functional.batch_norm will always be in training mode
    model.eval()

    gm = symbolic_trace(model, meta_args=meta_args)
    gm.eval()
    # run forward
    with torch.no_grad():
        fx_out = gm(**data)
        non_fx_out = model(**data)

    # compare output
    transformed_fx_out = output_transform_fn(fx_out)
    transformed_non_fx_out = output_transform_fn(non_fx_out)

    assert len(transformed_fx_out) == len(transformed_non_fx_out)
    if torch.is_tensor(fx_out):
        assert torch.allclose(
            fx_out, non_fx_out), f'{model.__class__.__name__} has inconsistent outputs, {fx_out} vs {non_fx_out}'
    else:
        assert torch.allclose(
            fx_out.values(),
            non_fx_out.values()), f'{model.__class__.__name__} has inconsistent outputs, {fx_out} vs {non_fx_out}'
    for key in transformed_fx_out.keys():
        fx_output_val = transformed_fx_out[key]
        non_fx_output_val = transformed_non_fx_out[key]
        if torch.is_tensor(fx_output_val):
            assert torch.allclose(fx_output_val, non_fx_output_val, atol=1e-5), \
                f'{model.__class__.__name__} has inconsistent outputs, {fx_output_val} vs {non_fx_output_val}'
        else:
            assert torch.allclose(fx_output_val.values(), non_fx_output_val.values()
                                 ), f'{model.__class__.__name__} has inconsistent outputs, {fx_out} vs {non_fx_out}'


@clear_cache_before_run()
def test_torchrec_dlrm_models():
    torch.backends.cudnn.deterministic = True
    dlrm_models = model_zoo.get_sub_registry('dlrm')

    for name, (model_fn, data_gen_fn, output_transform_fn, _, attribute) in dlrm_models.items():
        data = data_gen_fn()

        # dlrm_interactionarch is not supported
        # TODO(FrankLeeeee): support this model
        if name == 'dlrm_interactionarch':
            continue

        if attribute is not None and attribute.has_control_flow:
            meta_args = {k: v.to('meta') for k, v in data.items()}
        else:
            meta_args = None

        trace_and_compare(model_fn, data, output_transform_fn, meta_args)


if __name__ == "__main__":
    test_torchrec_dlrm_models()
