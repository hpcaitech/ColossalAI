from typing import Dict, List

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node, symbolic_trace

from colossalai.elixir.tracer.utils import meta_copy


def generate_fx_order(model: nn.Module) -> List[Dict[str, nn.Parameter]]:
    fxf_name_mark = '_fxf_name'
    fxf_param_mark = '_fxf_param'

    def tensor_trans(t):
        meta_t = t.data.to('meta')
        if isinstance(t, nn.Parameter):
            meta_t = nn.Parameter(meta_t)
        return meta_t

    meta_model = meta_copy(model, tensor_trans)

    # attach names for parameters
    for name, param in meta_model.named_parameters():
        setattr(param, fxf_name_mark, name)

    fx_forward_order: List[Dict[str, nn.Parameter]] = list()

    gm: GraphModule = symbolic_trace(meta_model)

    for node in gm.graph.nodes:
        if node.op in ('output', 'placeholder'):
            continue

        step_dict = None
        if node.op == 'get_attr':
            maybe_param = getattr(gm, node.target)
            # mark this node as a parameter
            if maybe_param is not None:
                setattr(node, fxf_param_mark, maybe_param)
            continue
        elif node.op == 'call_module':
            target_module = gm.get_submodule(node.target)
            step_dict = dict()
            # collect all parameters in the module
            for maybe_param in target_module.parameters():
                if maybe_param is not None:
                    param_name = getattr(maybe_param, fxf_name_mark)
                    step_dict[param_name] = maybe_param
        elif node.op in ('call_function', 'call_method'):
            step_dict = dict()
            for pre in node.args:
                if hasattr(pre, fxf_param_mark):
                    param = getattr(pre, fxf_param_mark)
                    param_name = getattr(param, fxf_name_mark)
                    step_dict[param_name] = param
        else:
            raise RuntimeError(f'Unsupported node op {node.op}!')

        if step_dict is not None and len(step_dict) > 0:
            fx_forward_order.append(step_dict)

    return fx_forward_order
