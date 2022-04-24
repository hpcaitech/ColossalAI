from numpy import allclose
import torch
from colossalai.tensor import ColoTensor
from copy import deepcopy
from colossalai.utils import get_current_device


def test_layernorm():
    ln_op = torch.nn.LayerNorm(2, 3, device=get_current_device())
    ln_op_colo = deepcopy(ln_op)

    input_t = torch.randn(3, 2, device=get_current_device())
    input_t_colo = ColoTensor.init_from_torch_tensor(tensor=input_t.clone().detach())

    # prepare colossalai LN
    delattr(ln_op_colo, 'weight')
    weight_clone = ln_op.weight.clone().detach()
    weight_clone.requires_grad = True
    setattr(ln_op_colo, 'weight', ColoTensor.init_from_torch_tensor(tensor=weight_clone))

    output = ln_op(input_t)
    output_colo = ln_op_colo(input_t_colo)

    assert allclose(output_colo.torch_tensor().detach().cpu(), output.detach().cpu())

    torch.mean(output).backward()
    torch.mean(output_colo).backward()

    assert allclose(ln_op.weight.grad.cpu(), ln_op_colo.weight.torch_tensor().grad.cpu())


def test_linear():
    in_dim = 4
    out_dim = 5

    fc = torch.nn.Linear(in_dim, out_dim, bias=True)
    fc_ref = deepcopy(fc)

    input_ref = torch.randn(1, in_dim)
    input_tensor = input_ref.clone()

    sharded_weight = ColoTensor.init_from_torch_tensor(fc_ref.weight)
    sharded_bias = ColoTensor.init_from_torch_tensor(fc_ref.bias)

    # replace the torch nn.Parameters with ShardedTensor
    delattr(fc, 'weight')
    setattr(fc, 'weight', sharded_weight)
    delattr(fc, 'bias')
    setattr(fc, 'bias', sharded_bias)

    fc.weight.requires_grad = True
    fc.bias.requires_grad = True

    # torch.nn.functional.linear(torch.randn(1, in_dim), sharded_weight, sharded_bias)
    out = fc(input_tensor)
    loss = out.sum()
    loss.backward()

    out_ref = fc_ref(input_ref)
    loss_ref = out_ref.sum()
    loss_ref.backward()

    assert (loss_ref == loss)
    assert allclose(fc_ref.weight.grad, fc.weight.torch_tensor().grad)


# The test case failed
# def test_uniform():
#     t = ColoTensor(torch.zeros(3, 5))
#     torch.nn.init.uniform_(t)
#     print(t)


def test_element_wise():
    t_ref = torch.randn(3, 5)
    t = ColoTensor.init_from_torch_tensor(t_ref.clone())
    assert torch.mean(t) == torch.mean(t_ref)
    assert allclose(torch.nn.functional.gelu(t).torch_tensor(), torch.nn.functional.gelu(t_ref))
    assert allclose(torch.nn.functional.relu(t).torch_tensor(), torch.nn.functional.relu(t_ref))


# Test a function not wrapped by
def test_no_wrap_op():
    t_ref = torch.randn(3, 5)
    t = ColoTensor.init_from_torch_tensor(t_ref.clone())
    assert torch.sum(t) == torch.sum(t_ref)
    assert torch.sum(input=t) == torch.sum(input=t_ref)


def test_lazy_init_tensor():
    lazy_t = ColoTensor(2, 3, dtype=torch.float32, requires_grad=True)
    assert lazy_t._torch_tensor.numel() == 0
    assert lazy_t.numel() == 6 == lazy_t.torch_tensor().numel()


def check_all():
    test_linear()
    test_element_wise()
    test_no_wrap_op()
    test_lazy_init_tensor()


if __name__ == '__main__':
    # test_lazy_init_ptensor()
    test_layernorm()
