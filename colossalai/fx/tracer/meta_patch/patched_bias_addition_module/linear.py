import copy
import operator
from weakref import ProxyType

import torch

from ..registry import bias_addition_module


@bias_addition_module.register(torch.nn.Linear)
def non_bias_nn_linear(tracer, target, args, kwargs):
    weight_node_kind = 'get_attr'
    weight_node_target = target + '.weight'
    weight_proxy = tracer.create_proxy(weight_node_kind, weight_node_target, (), {})

    bias_node_kind = 'get_attr'
    bias_node_target = target + '.bias'
    bias_proxy = tracer.create_proxy(bias_node_kind, bias_node_target, (), {})

    linear_node_kind = 'call_function'
    linear_node_target = torch.nn.functional.linear
    linear_node_args = list(args)
    linear_node_args.append(weight_proxy)
    linear_proxy = tracer.create_proxy(linear_node_kind, linear_node_target, tuple(linear_node_args), kwargs)

    bias_add_node_kind = 'call_function'
    bias_add_node_target = operator.add
    bias_add_args = (linear_proxy, bias_proxy)
    bias_add_proxy = tracer.create_proxy(bias_add_node_kind, bias_add_node_target, tuple(bias_add_args), kwargs)

    return bias_add_proxy
