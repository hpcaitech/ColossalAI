import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter


class Linear(nn.Module):
    """Linear layer with column parallelism.
    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """

    def __init__(self, input_size, output_size, bias=True, skip_bias_add=False):
        super(Linear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add

        self.weight = Parameter(
            torch.empty(
                self.output_size,
                self.input_size,
            )
        )
        init.normal_(self.weight)
        if bias:
            self.bias = Parameter(torch.empty(self.output_size))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

    def forward(self, input_):
        # Matrix multiply.
        bias = self.bias if not self.skip_bias_add else None
        output = F.linear(input_, self.weight, bias)

        if self.skip_bias_add:
            return output, self.bias
        else:
            return output

    def __repr__(self):
        return (
            f"Linear(in_features={self.input_size}, out_features={self.output_size}, "
            + f"bias={self.bias is not None}, skip_bias_add={self.skip_bias_add})"
        )
