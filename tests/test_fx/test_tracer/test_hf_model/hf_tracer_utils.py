from typing import List

import torch
from numpy import isin
from torch.fx import GraphModule
from torch.utils._pytree import tree_flatten

# from colossalai.fx import symbolic_trace
from colossalai._analyzer.fx import symbolic_trace


def trace_model_and_compare_output(model, data_gen, ignore_data: List[str] = None):
    # must turn on eval mode to ensure the output is consistent
    model.eval()

    inputs = data_gen()

    if ignore_data is not None:
        # drop the ignore_data key
        inputs = {k: v for k, v in inputs.items() if k not in ignore_data}

    try:
        meta_args = {k: v.to('meta') for k, v in inputs.items()}
        gm = symbolic_trace(model, meta_args=meta_args)
    except Exception as e:
        raise RuntimeError(f"Failed to trace {model.__class__.__name__}, error: {e}")

    # run forward
    non_fx_out = model(**inputs)
    fx_out = gm(**inputs)

    # check output
    for k in non_fx_out.keys():
        if torch.is_tensor(fx_out[k]):
            assert torch.equal(
                fx_out[k], non_fx_out[k]
            ), f'{model.__class__.__name__} has incorrect output {k}, expect {non_fx_out[k]}, but got {fx_out[k]}'
