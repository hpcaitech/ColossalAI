import torch
import torch.nn.functional as F

from ...registry import bias_addition_module
from .bias_addition_module import BiasAdditionModule


@bias_addition_module.register(torch.nn.Linear)
class BiasAdditionLinear(BiasAdditionModule):

    def extract_kwargs_from_mod(self):
        return {}

    def generate(self):
        non_bias_linear_func_proxy = self.create_non_bias_func_proxy()
        bias_addition_proxy = self.create_bias_addition_proxy(non_bias_linear_func_proxy, self.bias_proxy)
        return bias_addition_proxy
