import operator
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class BiasAdditionFunc(ABC):
    """
    This class is used to construct the restructure computation graph for
    call_func node with bias addition inside.
    """

    def __init__(self, tracer, target, args, kwargs, substitute_func):
        self.tracer = tracer
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.substitute_func = substitute_func

    @abstractmethod
    def extract_kwargs_from_origin_func(self):
        """
        This method is used to extract the kwargs for further graph transform.

        For example:
            The formula for torch.addmm is out = beta * input + alpha * (m1 @ m2)
            The kwargs for addmm function is {beta=1, alpha=1, output=None}, then we need
            to insert two more operator.mul nodes for the computation graph to compute the
            final result.
        """
        pass

    @abstractmethod
    def generate(self):
        """
        This method is used to construct the whole restructure computation graph for call_func node with bias
        addition inside.

        A whole restructure computation graph will contain a weight node, a bias node, a non-bias addition computation node,
        a bias reshape node if needed and a bias addition node.

        Use torch.addmm as an example:
        The origin node is:
            %addmm: call_func[target=torch.addmm](args = (%input_1, m1, m2), kwargs = {beta=1, alpha=1})
        Restructured graph is:
            %transpose : [#users=1] = call_function[target=torch.transpose](args = (%m2, 0, 1), kwargs = {})
            %linear : [#users=1] = call_function[target=torch._C._nn.linear](args = (%m1, %transpose), kwargs = {})
            %mul : [#users=1] = call_function[target=operator.mul](args = (%input_1, 3), kwargs = {})
            %mul_1 : [#users=1] = call_function[target=operator.mul](args = (2, %linear), kwargs = {})
            %add : [#users=1] = call_function[target=operator.add](args = (%mul_1, %mul), kwargs = {})
        """
        pass

    def create_mul_node(self, input_proxy, coefficent):
        """
        This method is used to create a coefficent node for the numerical correctness.
        The formula for torch.addmm is out = beta * input + alpha * (m1 @ m2)
        Therefore, we need to use this method insert two more operator.mul nodes for
        the computation graph to compute the final result.
        """
        node_kind = 'call_function'
        node_target = operator.mul
        node_args = (
            input_proxy,
            coefficent,
        )
        node_kwargs = {}
        mul_proxy = self.tracer.create_proxy(node_kind, node_target, node_args, node_kwargs)
        return mul_proxy


class LinearBasedBiasFunc(BiasAdditionFunc):
    """
    This class is used to construct the restructure computation graph for
    call_func node based on F.linear.
    """

    def create_non_bias_func_proxy(self, input_proxy, other_proxy):
        """
        This method is used to create the non_bias_func proxy, the node created by this proxy will
        compute the main computation, such as convolution, with bias option banned.
        """
        assert self.substitute_func == torch.nn.functional.linear
        node_kind = 'call_function'
        node_target = self.substitute_func

        node_args = (input_proxy, other_proxy)
        # non-bias linear does not have any kwargs
        node_kwargs = {}
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


func_to_func_dict = {
    torch.addmm: F.linear,
    torch.addbmm: torch.bmm,
    F.linear: F.linear,
}

method_to_func_dict = {
    torch.Tensor.addmm: F.linear,
    torch.Tensor.addbmm: torch.bmm,
}
