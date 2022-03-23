import torch
from functools import partial
import torch.multiprocessing as mp
from colossalai.zero.sharded_param import ShardedTensor


def run_dist(rank, world_size, port):
    import colossalai
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    in_dim = 4
    out_dim = 5

    fc = torch.nn.Linear(in_dim, out_dim, bias=True)

    sharded_weight = ShardedTensor(torch.randn(in_dim, out_dim, requires_grad=True))
    bias = torch.randn(out_dim, requires_grad=True)
    sharded_bias = ShardedTensor(bias)

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


def test_customized_linear(world_size):
    from colossalai.utils import free_port
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_customized_linear(4)
