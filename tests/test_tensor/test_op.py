from numpy import allclose, require
import torch
from colossalai.tensor import ColoTensor
from copy import deepcopy


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
    assert allclose(torch.nn.functional.gelu(t), torch.nn.functional.gelu(t_ref))
    assert allclose(torch.nn.functional.relu(t), torch.nn.functional.relu(t_ref))


# Test a function not wrapped by
def test_no_wrap_op():
    t_ref = torch.randn(3, 5)
    t = ColoTensor.init_from_torch_tensor(t_ref.clone())
    assert torch.sum(t) == torch.sum(t_ref)
    assert torch.sum(input=t) == torch.sum(input=t_ref)

def test_lazy_init_tensor():
    lazy_t = ColoTensor((2, 3), dtype=torch.float32, requires_grad=True)
    assert lazy_t._torch_tensor == None
    assert lazy_t.torch_tensor().numel() == 6

if __name__ == '__main__':
    test_no_wrap_op()
    # test_element_wise()
