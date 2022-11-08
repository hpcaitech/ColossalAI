import torch

from colossalai.fx import symbolic_trace


def trace_and_compare(model, data_gen, need_meta=False, need_concrete=False, kwargs_transform=False):
    data = data_gen()
    concrete_args = data if need_concrete else {}
    meta_args = {k: v.to('meta') for k, v in data.items()} if need_meta else {}

    model.eval()

    gm = symbolic_trace(model, concrete_args=concrete_args, meta_args=meta_args)

    with torch.no_grad():
        non_fx_out = model(**data)

        if kwargs_transform:
            data = kwargs_transform(data)

        fx_out = gm(**data)
    if isinstance(fx_out, tuple):
        for non_fx, fx in zip(non_fx_out, fx_out):
            assert torch.allclose(
                non_fx, fx, atol=1e-5), f'{model.__class__.__name__} has inconsistent outputs, {fx_out} vs {non_fx_out}'
    else:
        assert torch.allclose(
            fx_out, non_fx_out,
            atol=1e-5), f'{model.__class__.__name__} has inconsistent outputs, {fx_out} vs {non_fx_out}'
