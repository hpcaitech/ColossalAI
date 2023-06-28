import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing import assert_close

import colossalai
from colossalai.shardformer.layer import GPT2FusedLinearConv1D_Col, GPT2FusedLinearConv1D_Row
from colossalai.shardformer.layer.qkv_fused_linear import split_fused_qkv_in_gpt2_style
from colossalai.testing import rerun_if_address_is_in_use, spawn


# This code is copied from https://github.com/huggingface/transformers
class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


def rearrange(tensor: torch.Tensor, dim: int):
    tensor = tensor.clone()
    world_size = 2
    order = torch.arange(world_size * 3)
    new_order = []
    for i in range(world_size):
        new_order.append(order[i::world_size])
    new_order = torch.cat(new_order)

    tensor_chunks = torch.chunk(tensor, world_size * 3, dim=dim)
    rearanged_tensor_chunks = [tensor_chunks[i] for i in new_order]
    rearanged_tensor = torch.cat(rearanged_tensor_chunks, dim=dim)
    return rearanged_tensor


def check_linear_conv_1d_col():
    linear = Conv1D(192, 48).cuda()
    linear_conv_col = GPT2FusedLinearConv1D_Col.from_native_module(linear,
                                                                   process_group=None,
                                                                   gather_output=True,
                                                                   n_fused=3)

    assert linear.weight.shape == torch.Size([48, 192])
    assert linear.bias.shape == torch.Size([192])
    assert linear_conv_col.weight.shape == torch.Size([48, 96])
    assert linear_conv_col.bias.shape == torch.Size([96])

    # ensure weights are reversibly loadable
    linear_conv_col.load_state_dict(linear.state_dict())
    linear.load_state_dict(linear_conv_col.state_dict())

    # check computation correctness
    x = torch.rand(4, 48).cuda()
    out = linear(x)
    gather_out = linear_conv_col(x)
    assert_close(rearrange(out, 1), gather_out)

    # check backward correctness
    out.sum().backward()
    gather_out.sum().backward()

    target_grad = split_fused_qkv_in_gpt2_style(linear.weight.grad, 3, None, True)
    assert_close(target_grad, linear_conv_col.weight.grad)


def check_linear_conv_1d_row():
    linear = Conv1D(192, 48).cuda()
    linear_row = GPT2FusedLinearConv1D_Row.from_native_module(linear, process_group=None, parallel_input=False)

    assert linear.weight.shape == torch.Size([48, 192])
    assert linear_row.weight.shape == torch.Size([24, 192])
    assert linear_row.bias.shape == torch.Size([192])

    # check computation correctness
    x = torch.rand(4, 48).cuda()
    out = linear(x)
    gather_out = linear_row(x)
    assert_close(out, gather_out)

    # check backward correctness
    out.sum().backward()
    gather_out.sum().backward()

    rank = dist.get_rank()
    target_grad = torch.chunk(linear.weight.grad, 2, dim=0)[rank]
    assert_close(target_grad, linear_row.weight.grad)


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    # test for linear conv
    check_linear_conv_1d_col()
    check_linear_conv_1d_row()


@rerun_if_address_is_in_use()
def test_linearconv():
    spawn(run_dist, nprocs=2)


if __name__ == '__main__':
    test_linearconv()
