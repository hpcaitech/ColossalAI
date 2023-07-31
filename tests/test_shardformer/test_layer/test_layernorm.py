import torch
import torch.nn as nn
from torch.testing import assert_close

import colossalai
from colossalai.shardformer.layer import FusedLayerNorm
from colossalai.testing import rerun_if_address_is_in_use, spawn


def check_layernorm():
    norm = nn.LayerNorm(128, 0.00001).cuda()
    norm1d = FusedLayerNorm.from_native_module(norm, process_group=None)

    assert norm1d.weight.shape == torch.Size([128])

    # ensure state dict is reversibly loadable
    norm.load_state_dict(norm1d.state_dict())
    norm1d.load_state_dict(norm.state_dict())

    # check computation correctness
    x = torch.rand(4, 128).cuda()
    out = norm(x)
    gather_out = norm1d(x)
    assert_close(out, gather_out)

    # check backward correctness
    out.sum().backward()
    gather_out.sum().backward()

    assert_close(norm.weight.grad, norm1d.weight.grad)


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    check_layernorm()


@rerun_if_address_is_in_use()
def test_layernorm():
    spawn(run_dist, nprocs=2)


if __name__ == '__main__':
    test_layernorm_1d()
