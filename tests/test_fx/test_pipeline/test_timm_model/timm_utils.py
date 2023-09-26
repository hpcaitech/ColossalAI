import inspect
import random

import numpy as np
import torch
from torch.fx import GraphModule

from colossalai.fx import ColoTracer
from colossalai.fx.passes.adding_split_node_pass import balanced_split_pass, split_with_split_nodes_pass

MANUAL_SEED = 0
random.seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)
torch.backends.cudnn.deterministic = True


def split_model_and_compare_output(model, data, meta_args=None):
    model.eval()

    # get origin output and rng state
    cpu_rng_state = torch.get_rng_state()
    output = model(data)

    # tracing model
    tracer = ColoTracer()
    try:
        graph = tracer.trace(root=model, meta_args=meta_args)
    except Exception as e:
        raise RuntimeError(f"Failed to trace {model.__class__.__name__}, error: {e}")
    gm = GraphModule(model, graph, model.__class__.__name__)
    gm.recompile()

    # apply transform passes
    annotated_model = balanced_split_pass(gm, 2)
    split_model, split_submodules = split_with_split_nodes_pass(annotated_model)

    # get split model
    model_part0 = list(split_model.children())[0]
    model_part1 = list(split_model.children())[1]

    # set rng state and compute output of split model
    torch.set_rng_state(cpu_rng_state)
    output_part0 = model_part0(data)
    sig = inspect.signature(model_part1.forward)
    if isinstance(output_part0, torch.Tensor):
        output_part1 = model_part1(output_part0)
    else:
        if len(output_part0) > len(sig.parameters):
            output_part0 = output_part0[: len(sig.parameters)]
        output_part1 = model_part1(*output_part0)
    assert output.equal(output_part1)
