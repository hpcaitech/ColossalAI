import torch

from colossalai._analyzer.fx import symbolic_trace


def trace_and_compare(model, data_gen, output_transform_fn, need_meta=False, need_concrete=False):
    data = data_gen()
    concrete_args = data if need_concrete else {}
    meta_args = {k: v.to("meta") for k, v in data.items()} if need_meta else {}

    model.eval()

    gm = symbolic_trace(model, concrete_args=concrete_args, meta_args=meta_args)

    with torch.no_grad():
        non_fx_out = model(**data)
        fx_out = gm(**data)

    # compare output
    transformed_fx_out = output_transform_fn(fx_out)
    transformed_non_fx_out = output_transform_fn(non_fx_out)

    assert len(transformed_fx_out) == len(transformed_non_fx_out)

    for key, fx_output_val in transformed_fx_out.items():
        non_fx_output_val = transformed_non_fx_out[key]
        assert torch.allclose(
            fx_output_val, non_fx_output_val, atol=1e-5
        ), f"{model.__class__.__name__} has inconsistent outputs, {fx_output_val} vs {non_fx_output_val}"
