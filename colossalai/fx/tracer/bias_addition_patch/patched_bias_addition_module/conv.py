import torch
import torch.nn.functional as F

from ...registry import bias_addition_module
from .bias_addition_module import BiasAdditionModule


@bias_addition_module.register(torch.nn.Conv1d)
@bias_addition_module.register(torch.nn.Conv2d)
@bias_addition_module.register(torch.nn.Conv3d)
class BiasAdditionConv(BiasAdditionModule):

    def create_bias_reshape_proxy(self, dimensions):
        """
        This method is used to reshape the bias node in order to make bias and
        output of non-bias convolution broadcastable.
        """
        bias_shape = [1] * dimensions
        bias_shape[1] = -1
        bias_reshape_node_kind = 'call_method'
        bias_reshape_node_target = 'view'
        bias_reshape_node_args = (self.bias_proxy, bias_shape)
        bias_reshape_proxy = self.tracer.create_proxy(bias_reshape_node_kind, bias_reshape_node_target,
                                                      bias_reshape_node_args, {})
        return bias_reshape_proxy

    def generate(self):
        non_bias_conv_func_proxy = self.create_non_bias_func_proxy()
        output_dims = non_bias_conv_func_proxy.meta_data.dim()
        bias_reshape_proxy = self.create_bias_reshape_proxy(output_dims)
        bias_addition_proxy = self.create_bias_addition_proxy(non_bias_conv_func_proxy, bias_reshape_proxy)
        return bias_addition_proxy
