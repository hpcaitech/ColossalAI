import os
import resource
from contextlib import nullcontext
from functools import partial
from time import time

import psutil
import torch
import torch.nn as nn
from commons.model_zoo import model_builder
from commons.utils import get_data, get_profile_context, get_tflops, get_time_stamp
from packaging import version
from torch.nn.parallel import DistributedDataParallel as DDP

import colossalai
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.nn.parallel import zero_model_wrapper, zero_optim_wrapper
from colossalai.tensor import ColoParameter, ComputePattern, ComputeSpec, ProcessGroup, ReplicaSpec, ShardSpec
from colossalai.utils import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.utils.model.experimental import LazyInitContext


class Fool(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.empty(1024, 1024, 1024))


CAI_VERSION = colossalai.__version__


def parse_args():
    parser = colossalai.get_default_parser()
    parser.add_argument(
        "--placement",
        type=str,
        default='cuda',
        help="Placement Policy for Gemini. Valid when using colossalai as dist plan.",
    )
    parser.add_argument("--init_method",
                        choices=['naive', 'colo', 'lazy'],
                        default='naive',
                        help='Model initialization method.')
    parser.add_argument(
        "--model_type",
        type=str,
        default="gpt2_medium",
        help="model model scale",
    )
    parser.add_argument(
        "--use_gemini",
        default=False,
        action='store_true',
    )

    args = parser.parse_args()
    return args


class GPTLMLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2


def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024**2


def get_peak_gpu_mem():
    return torch.cuda.max_memory_allocated() / 1024**2


def get_peak_cpu_mem():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def get_mem_info(prefix=''):
    return f'{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB'


def get_peak_mem_info(prefix=''):
    return f'{prefix}Peak GPU memory usage: {get_peak_gpu_mem():.2f} MB, Peak CPU memory usage: {get_peak_cpu_mem():.2f} MB'


def get_model_size(model: nn.Module):
    total_numel = 0
    for module in model.modules():
        for p in module.parameters(recurse=False):
            total_numel += p.numel()
    return total_numel


def model_size_formatter(numel: int) -> str:
    GB_SIZE = 10**9
    MB_SIZE = 10**6
    KB_SIZE = 10**3
    if numel >= GB_SIZE:
        return f'{numel / GB_SIZE:.1f}B'
    elif numel >= MB_SIZE:
        return f'{numel / MB_SIZE:.1f}M'
    elif numel >= KB_SIZE:
        return f'{numel / KB_SIZE:.1f}K'
    else:
        return str(numel)


def set_cpu_maximum_parallelism():
    conf_str = torch.__config__.parallel_info()
    inter_str = conf_str.split("hardware_concurrency() : ")[1]
    max_concurrency = inter_str.split('\n')[0]
    os.environ["OMP_NUM_THREADS"] = max_concurrency
    print(f"environmental variable OMP_NUM_THREADS is set to {max_concurrency}.")


def main():
    # version check
    # this example is supposed to work for versions greater than 0.2.0
    assert version.parse(CAI_VERSION) >= version.parse("0.2.0")

    set_cpu_maximum_parallelism()
    args = parse_args()

    # batch size per DP degree
    disable_existing_loggers()
    colossalai.launch_from_torch(config={})

    logger = get_dist_logger()
    logger.info(f"{args.model_type}, {args.init_method}", ranks=[0])

    # build criterion

    torch.manual_seed(123)
    # all param must use the same process group.
    world_size = torch.distributed.get_world_size()

    if args.init_method == 'naive':
        ctx = nullcontext()
    elif args.init_method == 'colo':
        shard_pg = ProcessGroup(tp_degree=world_size)
        default_dist_spec = ShardSpec([-1], [world_size])
        ctx = ColoInitContext(default_pg=shard_pg, default_dist_spec=default_dist_spec)
    else:
        ctx = LazyInitContext()

    # build GPT model
    with ctx:
        if args.model_type == 'fool':
            model = Fool()
        else:
            model = model_builder(args.model_type)(checkpoint=True)

    logger.info(get_mem_info(prefix='After init model, '), ranks=[0])
    logger.info(get_peak_mem_info(prefix='After init model, '), ranks=[0])
    # asign running configurations

    if args.use_gemini:
        if args.model_type == 'fool':
            hidden_dim = None
        else:
            hidden_dim = model.config.n_embd
        gemini_config = dict(strict_ddp_mode=True,
                             device='cpu',
                             placement_policy=args.placement,
                             pin_memory=False,
                             hidden_dim=hidden_dim,
                             search_range_mb=128)

        # build a highly optimized gpu/cpu optimizer

        # wrap your model and optimizer
        model = zero_model_wrapper(model, 3, gemini_config)

        logger.info(get_mem_info(prefix='After init gemini, '), ranks=[0])
        logger.info(get_peak_mem_info(prefix='After init gemini, '), ranks=[0])
    else:
        if args.init_method == 'colo':
            logger.info('ColoInitContext is coupled with Gemini, ignore', ranks=[0])
        elif args.init_method == 'lazy':
            model = LazyInitContext.materialize(model)
            logger.info(get_mem_info(prefix='After materialization, '), ranks=[0])
            logger.info(get_peak_mem_info(prefix='After materialization, '), ranks=[0])


if __name__ == '__main__':
    main()
