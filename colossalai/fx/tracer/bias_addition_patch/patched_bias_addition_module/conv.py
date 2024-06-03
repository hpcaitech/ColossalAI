import torch
from torch.nn.modules.utils import _pair, _single, _triple

from ...registry import bias_addition_module
from .bias_addition_module import BiasAdditionModule


@bias_addition_module.register(torch.nn.Conv1d)
@bias_addition_module.register(torch.nn.Conv2d)
@bias_addition_module.register(torch.nn.Conv3d)
class BiasAdditionConv(BiasAdditionModule):
    def extract_kwargs_from_mod(self):
        root = self.tracer.root
        conv_module = root.get_submodule(self.target)
        kwarg_attributes = ["groups", "dilation", "stride"]
        non_bias_kwargs = {}
        for attr_name in kwarg_attributes:
            if hasattr(conv_module, attr_name):
                non_bias_kwargs[attr_name] = getattr(conv_module, attr_name)
        if conv_module.padding_mode != "zeros":
            # TODO: non zeros mode requires some extra processing for input
            conv_type = type(conv_module)
            if conv_type == "torch.nn.Conv1d":
                padding_element = _single(0)
            elif conv_type == "torch.nn.Conv2d":
                padding_element = _pair(0)
            elif conv_type == "torch.nn.Conv3d":
                padding_element = _triple(0)
            non_bias_kwargs["padding"] = padding_element
        else:
            non_bias_kwargs["padding"] = getattr(conv_module, "padding")

        return non_bias_kwargs

    def create_bias_reshape_proxy(self, dimensions):
        """
        This method is used to reshape the bias node in order to make bias and
        output of non-bias convolution broadcastable.
        """
        bias_shape = [1] * (dimensions - 1)
        bias_shape[0] = -1
        bias_reshape_node_kind = "call_method"
        bias_reshape_node_target = "view"
        bias_reshape_node_args = (self.bias_proxy, torch.Size(bias_shape))
        bias_reshape_proxy = self.tracer.create_proxy(
            bias_reshape_node_kind, bias_reshape_node_target, bias_reshape_node_args, {}
        )
        return bias_reshape_proxy

    def generate(self):
        non_bias_conv_func_proxy = self.create_non_bias_func_proxy()
        output_dims = non_bias_conv_func_proxy.meta_data.dim()
        bias_reshape_proxy = self.create_bias_reshape_proxy(output_dims)
        bias_addition_proxy = self.create_bias_addition_proxy(non_bias_conv_func_proxy, bias_reshape_proxy)
        return bias_addition_proxy
