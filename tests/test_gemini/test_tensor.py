import torch
from torch import nn
from colossalai.gemini.tensor.stateful_tensor import StatefulTensorV2
# TODO(jiaruifang) auto import
from colossalai.gemini.tensor._ops import *


def test_linear():
    in_dim = 4
    out_dim = 5

    fc = torch.nn.Linear(in_dim, out_dim, bias=True)

    sharded_weight = StatefulTensorV2(torch.randn(out_dim, in_dim, requires_grad=True))
    bias = torch.randn(out_dim, requires_grad=True)
    sharded_bias = StatefulTensorV2(bias)

    # replace the torch nn.Parameters with ShardedTensor
    delattr(fc, 'weight')
    setattr(fc, 'weight', sharded_weight)
    delattr(fc, 'bias')
    setattr(fc, 'bias', sharded_bias)

    fc.weight.requires_grad = True
    fc.bias.requires_grad = True

    # torch.nn.functional.linear(torch.randn(1, in_dim), sharded_weight, sharded_bias)
    out = fc(torch.randn(1, in_dim))

    loss = out.sum()
    loss.backward()


def test_stateful_tensor():
    t = StatefulTensorV2(torch.empty(3, 5))
    torch.nn.init.uniform_(t)


if __name__ == '__main__':
    test_linear()
