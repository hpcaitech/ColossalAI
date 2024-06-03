import operator
from functools import reduce
from typing import Optional, Tuple

import torch

from ..registry import meta_profiler_module


def _rnn_flops(
    flops: int, macs: int, module: torch.nn.RNNBase, w_ih: torch.Tensor, w_hh: torch.Tensor
) -> Tuple[int, int]:
    # copied from https://github.com/sovrasov/flops-counter.pytorch/blob/master/ptflops/pytorch_ops.py

    # matrix matrix mult ih state and internal state
    macs += reduce(operator.mul, w_ih.shape)
    flops += 2 * reduce(operator.mul, w_ih.shape)
    # matrix matrix mult hh state and internal state
    macs += reduce(operator.mul, w_hh.shape)
    flops += 2 * reduce(operator.mul, w_hh.shape)
    if isinstance(module, (torch.nn.RNN, torch.nn.RNNCell)):
        # add both operations
        flops += module.hidden_size
    elif isinstance(module, (torch.nn.GRU, torch.nn.GRUCell)):
        # hadamard of r
        flops += module.hidden_size
        # adding operations from both states
        flops += module.hidden_size * 3
        # last two hadamard product and add
        flops += module.hidden_size * 3
    elif isinstance(module, (torch.nn.LSTM, torch.nn.LSTMCell)):
        # adding operations from both states
        flops += module.hidden_size * 4
        # two hadamard product and add for C state
        flops += module.hidden_size * 3
        # final hadamard
        flops += module.hidden_size * 3
    return flops, macs


@meta_profiler_module.register(torch.nn.LSTM)
@meta_profiler_module.register(torch.nn.GRU)
@meta_profiler_module.register(torch.nn.RNN)
def torch_nn_rnn(self: torch.nn.RNNBase, input: torch.Tensor, hx: Optional[torch.Tensor] = None) -> Tuple[int, int]:
    flops = 0
    macs = 0
    for i in range(self.num_layers):
        w_ih = self.__getattr__("weight_ih_l" + str(i))
        w_hh = self.__getattr__("weight_hh_l" + str(i))
        flops, macs = _rnn_flops(flops, macs, self, w_ih, w_hh)
        if self.bias:
            b_ih = self.__getattr__("bias_ih_l" + str(i))
            b_hh = self.__getattr__("bias_hh_l" + str(i))
            flops += reduce(operator.mul, b_ih) + reduce(operator.mul, b_hh)
    flops *= reduce(operator.mul, input.shape[:2])
    macs *= reduce(operator.mul, input.shape[:2])
    if self.bidirectional:
        flops *= 2
        macs *= 2
    return flops, macs


@meta_profiler_module.register(torch.nn.LSTMCell)
@meta_profiler_module.register(torch.nn.GRUCell)
@meta_profiler_module.register(torch.nn.RNNCell)
def torch_nn_rnn(self: torch.nn.RNNCellBase, input: torch.Tensor, hx: Optional[torch.Tensor] = None) -> Tuple[int, int]:
    flops = 0
    macs = 0
    w_ih = self.__getattr__("weight_ih_l")
    w_hh = self.__getattr__("weight_hh_l")
    flops, macs = _rnn_flops(flops, macs, self, w_ih, w_hh)
    if self.bias:
        b_ih = self.__getattr__("bias_ih_l")
        b_hh = self.__getattr__("bias_hh_l")
        flops += reduce(operator.mul, b_ih) + reduce(operator.mul, b_hh)
    flops *= input.shape[0]
    macs *= input.shape[0]
    return flops, macs
