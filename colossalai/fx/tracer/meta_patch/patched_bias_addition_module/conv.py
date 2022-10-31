import copy
import operator

import torch

from ..registry import bias_addition_module


@bias_addition_module.register(torch.nn.Conv1d)
def non_bias_nn_conv1d(tracer, target, args, kwargs):
    return _non_bias_nn_conv(tracer, target, args, kwargs, torch.nn.functional.conv1d)


@bias_addition_module.register(torch.nn.Conv2d)
def non_bias_nn_conv2d(tracer, target, args, kwargs):
    return _non_bias_nn_conv(tracer, target, args, kwargs, torch.nn.functional.conv2d)


@bias_addition_module.register(torch.nn.Conv3d)
def non_bias_nn_conv3d(tracer, target, args, kwargs):
    return _non_bias_nn_conv(tracer, target, args, kwargs, torch.nn.functional.conv3d)


def _non_bias_nn_conv(tracer, target, args, kwargs, conv_function):
    weight_node_kind = 'get_attr'
    weight_node_target = target + '.weight'
    weight_proxy = tracer.create_proxy(weight_node_kind, weight_node_target, (), {})

    bias_node_kind = 'get_attr'
    bias_node_target = target + '.bias'
    bias_proxy = tracer.create_proxy(bias_node_kind, bias_node_target, (), {})

    conv_node_kind = 'call_function'
    conv_node_target = conv_function
    conv_node_args = list(args)
    conv_node_args.append(weight_proxy)
    conv_proxy = tracer.create_proxy(conv_node_kind, conv_node_target, tuple(conv_node_args), {})

    bias_dim = conv_proxy.meta_data.dim()
    bias_shape = [1] * bias_dim
    bias_shape[1] = -1
    bias_reshape_node_kind = 'call_method'
    bias_reshape_node_target = 'view'
    bias_reshape_node_args = (bias_proxy, bias_shape)
    bias_reshape_kind = tracer.create_proxy(bias_reshape_node_kind, bias_reshape_node_target, bias_reshape_node_args,
                                            {})

    bias_add_node_kind = 'call_function'
    bias_add_node_target = operator.add
    bias_add_args = (conv_proxy, bias_reshape_kind)
    bias_add_proxy = tracer.create_proxy(bias_add_node_kind, bias_add_node_target, tuple(bias_add_args), kwargs)

    return bias_add_proxy
