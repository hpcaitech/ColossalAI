import torch

from ...registry import bias_addition_function, bias_addition_method
from .bias_addition_function import LinearBasedBiasFunc


@bias_addition_method.register(torch.Tensor.addmm)
@bias_addition_function.register(torch.addmm)
class Addmm(LinearBasedBiasFunc):
    def extract_kwargs_from_origin_func(self):
        kwargs = {}
        if "beta" in self.kwargs:
            kwargs["beta"] = self.kwargs["beta"]
        if "alpha" in self.kwargs:
            kwargs["alpha"] = self.kwargs["alpha"]
        return kwargs

    def transpose_other_operand_for_linear(self, other_proxy):
        """
        This method is used to transpose the other operand for linear function.
        For example:
            input = torch.rand(3, 4)
            m1 = torch.rand(3, 5)
            m2 = torch.rand(5, 4)
            original_output = torch.addmm(input, m1, m2)
            # To keep the computation graph consistent with the origin computation graph, we need to transpose the m2
            # before we call the linear function.
            new_output = torch.linear(m1, m2.transpose(0, 1)) + input
        """
        node_kind = "call_function"
        node_target = torch.transpose
        node_args = (other_proxy, 0, 1)
        node_kwargs = {}
        transpose_proxy = self.tracer.create_proxy(node_kind, node_target, node_args, node_kwargs)
        return transpose_proxy

    def generate(self):
        transpose_proxy = self.transpose_other_operand_for_linear(self.args[2])
        non_bias_linear_func_proxy = self.create_non_bias_func_proxy(self.args[1], transpose_proxy)
        kwargs = self.extract_kwargs_from_origin_func()

        if "beta" in kwargs:
            beta = kwargs["beta"]
            beta_proxy = self.create_mul_node(self.args[0], beta)
        else:
            beta_proxy = self.args[0]

        if "alpha" in kwargs:
            alpha = kwargs["alpha"]
            alpha_proxy = self.create_mul_node(alpha, non_bias_linear_func_proxy)
        else:
            alpha_proxy = non_bias_linear_func_proxy

        bias_addition_proxy = self.create_bias_addition_proxy(alpha_proxy, beta_proxy)

        return bias_addition_proxy
