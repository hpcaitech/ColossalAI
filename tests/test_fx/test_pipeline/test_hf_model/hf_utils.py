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


def split_model_and_compare_output(model, data_gen):
    model.eval()

    # generate input sample
    kwargs = data_gen()

    # get origin output and rng state
    cpu_rng_state = torch.get_rng_state()
    output = model(**kwargs)

    # tracing model
    tracer = ColoTracer()
    try:
        meta_args = {k: v.to("meta") for k, v in kwargs.items()}
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
    output_part0 = model_part0(**kwargs)
    sig = inspect.signature(model_part1.forward)
    if isinstance(output_part0, torch.Tensor):
        output_part1 = model_part1(output_part0)
    else:
        if len(output_part0) > len(sig.parameters):
            output_part0 = output_part0[: len(sig.parameters)]
        output_part1 = model_part1(*output_part0)

    # get output tensor from HFOutput datastructure
    if "logits" in output:
        output_to_compare = output["logits"]
    elif "prediction_logits" in output:
        output_to_compare = output["prediction_logits"]
    else:
        output_to_compare = output["last_hidden_state"]

    # compare output
    if isinstance(output_part1, torch.Tensor):
        assert output_to_compare.equal(output_part1)
    elif isinstance(output_part1, (tuple, list)):
        assert output_to_compare.equal(output_part1[0])
    else:
        assert False
