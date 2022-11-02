import operator
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class BiasAdditionModule(ABC):
    """
    This class is used to construct the restructure computation graph for
    call_module node with bias addition inside.
    """

    def __init__(self, tracer, target, args, kwargs, substitute_func):
        self.tracer = tracer
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.substitute_func = substitute_func
        self.weight_proxy = self._create_weight_proxy()
        self.bias_proxy = self._create_bias_proxy()

    def _create_weight_proxy(self):
        """
        Create weight proxy, the node created by this proxy contains module weight.

        Note: this function will be invoked during module initializing,
              you should never call this function.
        """
        weight_node_kind = 'get_attr'
        weight_node_target = self.target + '.weight'
        weight_proxy = self.tracer.create_proxy(weight_node_kind, weight_node_target, (), {})
        return weight_proxy

    def _create_bias_proxy(self):
        """
        Create bias proxy, the node created by this proxy contains module bias.

        Note: this function will be invoked during module initializing,
              you should never call this function.
        """
        bias_node_kind = 'get_attr'
        bias_node_target = self.target + '.bias'
        bias_proxy = self.tracer.create_proxy(bias_node_kind, bias_node_target, (), {})
        return bias_proxy

    @abstractmethod
    def extract_kwargs_from_mod(self):
        """
        This method is used to extract the kwargs for non-bias computation.

        For example:
            The kwargs for conv2d module is {} because the attributes like 'padding' or 'groups' are
            considered during module initilizing. However, we need to consider those attributes as kwargs
            in F.conv2d.
        """
        pass

    def create_non_bias_func_proxy(self, input_proxy=None):
        """
        This method is used to create the non_bias_func proxy, the node created by this proxy will
        compute the main computation, such as convolution, with bias option banned.
        """
        node_kind = 'call_function'
        node_target = self.substitute_func
        if input_proxy is None:
            input_proxy = self.args[0]
        node_args = (input_proxy, self.weight_proxy)
        node_kwargs = self.extract_kwargs_from_mod()
        non_bias_func_proxy = self.tracer.create_proxy(node_kind, node_target, node_args, node_kwargs)
        return non_bias_func_proxy

    def create_bias_addition_proxy(self, non_bias_func_proxy, bias_proxy):
        """
        This method is used to create the bias_addition_proxy, the node created by this proxy will
        compute the sum of non_bias_func result and bias with some reshape operation if needed.
        """
        bias_add_node_kind = 'call_function'
        bias_add_node_target = operator.add
        bias_add_args = (non_bias_func_proxy, bias_proxy)
        bias_add_proxy = self.tracer.create_proxy(bias_add_node_kind, bias_add_node_target, tuple(bias_add_args), {})
        return bias_add_proxy

    @abstractmethod
    def generate(self):
        """
        This method is used to construct the whole restructure computation graph for call_module node with bias
        addition inside.

        A whole restructure computation graph will contain a weight node, a bias node, a non-bias addition computation node,
        a bias reshape node if needed and a bias addition node.

        Use Conv2d module as an example:
        The origin node is:
            %conv: call_module[target=conv](args = (%x,), kwargs = {})
        Restructured graph is:
            %conv_weight : [#users=1] = get_attr[target=conv.weight]
            %conv_bias : [#users=1] = get_attr[target=conv.bias]
            %conv2d : [#users=1] = call_function[target=torch.conv2d](args = (%x, %conv_weight), kwargs = {})
            %view : [#users=1] = call_method[target=view](args = (%conv_bias, [1, -1, 1, 1]), kwargs = {})
            %add : [#users=1] = call_function[target=operator.add](args = (%conv2d, %view), kwargs = {})
        """
        pass


module_to_func_dict = {
    torch.nn.Linear: F.linear,
    torch.nn.Conv1d: F.conv1d,
    torch.nn.Conv2d: F.conv2d,
    torch.nn.Conv3d: F.conv3d,
}
