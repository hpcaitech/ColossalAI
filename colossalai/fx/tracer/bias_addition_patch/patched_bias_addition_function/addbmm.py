import operator

import torch
import torch.nn.functional as F

from ...registry import bias_addition_function, bias_addition_method
from .bias_addition_function import LinearBasedBiasFunc


@bias_addition_method.register(torch.Tensor.addbmm)
@bias_addition_function.register(torch.addbmm)
class Addbmm(LinearBasedBiasFunc):

    def extract_kwargs_from_origin_func(self):
        kwargs = {}
        if 'beta' in self.kwargs:
            kwargs['beta'] = self.kwargs['beta']
        if 'alpha' in self.kwargs:
            kwargs['alpha'] = self.kwargs['alpha']
        return kwargs

    def create_non_bias_func_proxy(self, input_proxy, other_proxy):
        """
        This method is used to create the non_bias_func proxy, the node created by this proxy will
        compute the main computation, such as convolution, with bias option banned.
        """
        assert self.substitute_func == torch.bmm
        node_kind = 'call_function'
        node_target = self.substitute_func

        node_args = (input_proxy, other_proxy)
        # torch.bmm does not have any kwargs
        node_kwargs = {}
        non_bias_func_proxy = self.tracer.create_proxy(node_kind, node_target, node_args, node_kwargs)
        return non_bias_func_proxy

    def insert_sum_node(self, input_proxy, sum_dims=0):
        '''
        This method is used to sum the input_proxy through the sum_dims.
        '''
        node_kind = 'call_function'
        node_target = torch.sum
        node_args = (input_proxy, sum_dims)
        node_kwargs = {}
        sum_proxy = self.tracer.create_proxy(node_kind, node_target, node_args, node_kwargs)
        return sum_proxy

    def generate(self):
        # The formula for addbmm is output = beta * input + alpha * (torch.bmm(b1, b2))

        # doing the non-bias computation(temp_0 = torch.bmm(b1, b2))
        non_bias_linear_func_proxy = self.create_non_bias_func_proxy(self.args[1], self.args[2])

        # doing sum on the batch dimension(temp_1 = torch.sum(temp_0, 0))
        sum_proxy = self.insert_sum_node(non_bias_linear_func_proxy)
        kwargs = self.extract_kwargs_from_origin_func()

        if 'beta' in kwargs:
            beta = kwargs['beta']
            # doing the multiplication with beta if it exists(temp_2 = beta * input)
            beta_proxy = self.create_mul_node(self.args[0], beta)
        else:
            beta_proxy = self.args[0]

        if 'alpha' in kwargs:
            alpha = kwargs['alpha']
            # doing the multiplication with alpha if it exists(temp_3 = alpha * temp_1)
            alpha_proxy = self.create_mul_node(alpha, sum_proxy)
        else:
            alpha_proxy = sum_proxy

        # doing the addition(temp_4 = temp_2 + temp_3)
        bias_addition_proxy = self.create_bias_addition_proxy(alpha_proxy, beta_proxy)

        return bias_addition_proxy
