import time

import torch
import torch.distributed as dist
from colossalai.communication import all_gather, reduce_scatter, all_reduce
from colossalai.context import ParallelMode, seed
from colossalai.core import global_context as gpc
from colossalai.initialize import launch_from_torch
from colossalai.utils import get_current_device, print_rank_0

# ARGS = parse_args()
# size = ARGS.world_size
# rank = ARGS.rank

# init_method = f'tcp://{ARGS.host}:{ARGS.port}'
# dist.init_process_group(backend='nccl', rank=rank, world_size=size, init_method=init_method)
CONFIG = dict(parallel=dict(data=8, pipeline=1, tensor=dict(mode=None, size=1)))
launch_from_torch(CONFIG)

assert dist.get_rank() == gpc.get_global_rank()

print('Rank {} / {}'.format(dist.get_rank(), dist.get_world_size()))

SIZE = 8
tensor = torch.tensor([dist.get_rank() * SIZE + j for j in range(SIZE)])
tensor = tensor.to(get_current_device())
print('Before:   Rank {0} - {1}'.format(dist.get_rank(), tensor))
tensor, op = all_gather(tensor, 0, ParallelMode.GLOBAL, async_op=True)
# tensor, op = reduce_scatter(tensor, 0, ParallelMode.GLOBAL, async_op=True)
# tensor, op = all_reduce(tensor, ParallelMode.GLOBAL, async_op=True)
print_rank_0('After:    Rank {0} - {1}'.format(dist.get_rank(), tensor))
op.wait()
print_rank_0('Complete: Rank {0} - {1}'.format(dist.get_rank(), tensor))
