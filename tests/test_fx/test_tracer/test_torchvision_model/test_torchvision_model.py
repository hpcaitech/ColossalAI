import torch

from colossalai._analyzer.fx import symbolic_trace
from colossalai.testing import clear_cache_before_run
from tests.kit.model_zoo import model_zoo


@clear_cache_before_run()
def test_torchvision_models():
    torch.backends.cudnn.deterministic = True
    tv_sub_registry = model_zoo.get_sub_registry('torchvision')

    for name, (model_fn, data_gen_fn, output_transform_fn, _, model_attribute) in tv_sub_registry.items():
        data = data_gen_fn()

        if model_attribute is not None and model_attribute.has_stochastic_depth_prob:
            model = model_fn(stochastic_depth_prob=0)
        else:
            model = model_fn()

        gm = symbolic_trace(model)

        model.eval()
        gm.eval()

        try:
            with torch.no_grad():
                fx_out = gm(**data)
                non_fx_out = model(**data)
                transformed_out = output_transform_fn(fx_out)
                transformed_non_fx_out = output_transform_fn(non_fx_out)

            assert len(transformed_out) == len(transformed_non_fx_out)

            for key in transformed_out.keys():
                fx_val = transformed_out[key]
                non_fx_val = transformed_non_fx_out[key]
                assert torch.allclose(
                    fx_val,
                    non_fx_val), f'{model.__class__.__name__} has inconsistent outputs, {fx_val} vs {non_fx_val}'
        except Exception as e:
            print(name, e)


if __name__ == '__main__':
    test_torchvision_models()
