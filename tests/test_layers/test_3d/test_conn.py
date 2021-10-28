import torch
import torch.distributed as dist

from colossalai.initialize import parse_args
from colossalai.utils import get_current_device

ARGS = parse_args()
size = ARGS.world_size
rank = ARGS.local_rank

init_method = f'tcp://{ARGS.host}:{ARGS.port}'
dist.init_process_group(backend='nccl', rank=rank, world_size=size, init_method=init_method)
print('Rank {} / {}'.format(dist.get_rank(), dist.get_world_size()))

SIZE = 8
tensor = torch.randn(SIZE)
tensor = tensor.to(get_current_device())
dist.all_reduce(tensor)
print('Rank {0}: {1}'.format(rank, tensor.detach().cpu().numpy().tolist()))
