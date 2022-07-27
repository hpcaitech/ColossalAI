from colossalai.fx import ColoTracer
import torch
from torch.fx import GraphModule, Tracer


def trace_and_compare(model, data_gen, need_meta=False, need_concrete=False):
    data = data_gen()
    concrete_args = data if need_concrete else {}
    meta_args = {k: v.to('meta') for k, v in data.items()} if need_meta else {}
    tracer = ColoTracer()

    graph = tracer.trace(root=model, concrete_args=concrete_args, meta_args=meta_args)
    gm = GraphModule(model, graph, model.__class__.__name__)
    gm.recompile()

    model.eval()
    gm.eval()

    with torch.no_grad():
        non_fx_out = model(**data)
        fx_out = gm(**data)
    if isinstance(fx_out, tuple):
        for non_fx, fx in zip(non_fx_out, fx_out):
            assert torch.allclose(non_fx,
                                  fx), f'{model.__class__.__name__} has inconsistent outputs, {fx_out} vs {non_fx_out}'
    else:
        assert torch.allclose(
            fx_out, non_fx_out), f'{model.__class__.__name__} has inconsistent outputs, {fx_out} vs {non_fx_out}'
