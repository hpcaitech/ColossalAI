import torch
import torch.distributed as dist

import colossalai
from colossalai.testing import spawn


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    # ranks
    group1_ranks = [0, 1]
    group2_ranks = [2, 3]

    # create groups
    group1 = dist.new_group(ranks=group1_ranks, backend='nccl')
    group2 = dist.new_group(ranks=group2_ranks, backend='nccl')

    current_rank = dist.get_rank()
    val = torch.arange(4)[current_rank].cuda()

    #
    if current_rank in group1_ranks:
        src_rank = dist.distributed_c10d._get_global_rank(group1, 0)
        dist.broadcast(val, src_rank, group1)
    else:
        src_rank = dist.distributed_c10d._get_global_rank(group2, 0)
        dist.broadcast(val, src_rank, group2)

    print(f'{current_rank}: {val}')


def test_broadcast():
    spawn(run_dist, 4)


if __name__ == '__main__':
    test_broadcast()
