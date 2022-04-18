import torch
from .simple_model import MLP
from colossalai.utils import Timer, synchronize
from colossalai.core import global_context as gpc
from colossalai.context.parallel_mode import ParallelMode
from argparse import ArgumentParser

BATCH_SIZE = 8
SEQ_LENGTH = 120
HIDDEN_DIM = 1024
ITER_TIMES = 2000

def build_args_parser() -> ArgumentParser:
    """Helper function parsing the command line options."""

    parser = ArgumentParser(description="colossal benchmark")

    parser.add_argument("--num_gpus",
                        type=int,
                        default=-1,
                        help="Total number of devices to use.")
    parser.add_argument("--bs",
                        type=int,
                        default=BATCH_SIZE,
                        help="Batch size of the input tensor.")
    parser.add_argument("--seq_len",
                        type=int,
                        default=SEQ_LENGTH,
                        help="Sequence length of the input tensor.")
    parser.add_argument("--hid_dim",
                        type=int,
                        default=HIDDEN_DIM,
                        help="Hidden dimension of the input tensor.")
    parser.add_argument("--num_steps",
                        type=int,
                        default=ITER_TIMES,
                        help="The number of iteration times.")
    return parser

def build_input_tensor(args):
    return torch.rand(args.bs, args.seq_len, args.hid_dim)

def build_configs_helper(device_cnt: int):
    config_dict = {}

    if device_cnt < 2:
        return config_dict
    
    if device_cnt < 4:
        config_dict["1d"] = dict(parallel=dict(tensor=dict(size=2, mode='1d')))
    elif device_cnt < 8:
        config_dict["1d"] = dict(parallel=dict(tensor=dict(size=4, mode='1d')))
        config_dict["2d"] = dict(parallel=dict(tensor=dict(size=4, mode='2d')))
    else:
        config_dict["1d"] = dict(parallel=dict(tensor=dict(size=8, mode='1d')))
        config_dict["2d"] = dict(parallel=dict(data=2, tensor=dict(size=4, mode='2d')))
        config_dict["2p5d"] = dict(parallel=dict(tensor=dict(size=8, mode='2.5d', depth=2)))
        config_dict["3d"] = dict(parallel=dict(tensor=dict(size=8, mode='3d')))
    
    return config_dict

def build_configs(args):
    total_device_cnt = torch.cuda.device_count()
    if args.num_gpus == -1:
        config_dict = build_configs_helper(total_device_cnt)
    else:
        valid_device_cnt = min(args.num_gpus, total_device_cnt)
        config_dict = build_configs_helper(valid_device_cnt)
    return config_dict

def profile_1d(input_tensor, config, args):
    gpc.load_config(config)
    gpc.init_parallel_groups()
    assert gpc.is_initialized(ParallelMode.PARALLEL_1D)
    model = MLP(args.hid_dim).cuda()
    input_tensor = input_tensor.cuda()
    torch.distributed.broadcast(input_tensor, src=0)
    timer = Timer()
    iter_times = args.num_steps
    timer.start()
    for i in range(iter_times):
        input_tensor = model(input_tensor)
        synchronize()
    result_1d = timer.stop()
    return result_1d

def profile_2d(input_tensor, config, args):
    gpc.load_config(config)
    gpc.init_parallel_groups()
    assert gpc.is_initialized(ParallelMode.PARALLEL_2D_COL)
    assert gpc.is_initialized(ParallelMode.PARALLEL_2D_ROW)
    model = MLP(args.hid_dim).cuda()
    input_tensor = input_tensor.cuda()
    torch.distributed.broadcast(input_tensor, src=0)
    input_tensor = torch.chunk(input_tensor, 2, dim=0)[gpc.get_local_rank(ParallelMode.PARALLEL_2D_COL)]
    input_tensor = torch.chunk(input_tensor, 2, dim=-1)[gpc.get_local_rank(ParallelMode.PARALLEL_2D_ROW)]
    timer = Timer()
    iter_times = args.num_steps
    timer.start()
    for i in range(iter_times):
        input_tensor = model(input_tensor)
        synchronize()
    result_2d = timer.stop()
    return result_2d

def profile_2p5d(input_tensor, config, args):
    gpc.load_config(config)
    gpc.init_parallel_groups()
    assert gpc.is_initialized(ParallelMode.PARALLEL_2P5D_COL)
    assert gpc.is_initialized(ParallelMode.PARALLEL_2P5D_ROW)
    assert gpc.is_initialized(ParallelMode.PARALLEL_2P5D_DEP)
    model = MLP(args.hid_dim).cuda()
    input_tensor = input_tensor.cuda()
    torch.distributed.broadcast(input_tensor, src=0)
    input_tensor = torch.chunk(input_tensor, 2, dim=0)[gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_DEP)]
    input_tensor = torch.chunk(input_tensor, 2, dim=0)[gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL)]
    input_tensor = torch.chunk(input_tensor, 2, dim=-1)[gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_ROW)]
    timer = Timer()
    iter_times = args.num_steps
    timer.start()
    for i in range(iter_times):
        input_tensor = model(input_tensor)
        synchronize()
    result_2p5d = timer.stop()
    return result_2p5d

def profile_3d(input_tensor, config, args):
    gpc.load_config(config)
    gpc.init_parallel_groups()
    assert gpc.is_initialized(ParallelMode.PARALLEL_3D_WEIGHT)
    assert gpc.is_initialized(ParallelMode.PARALLEL_3D_INPUT)
    assert gpc.is_initialized(ParallelMode.PARALLEL_3D_OUTPUT)
    model = MLP(args.hid_dim).cuda()
    input_tensor = input_tensor.cuda()
    torch.distributed.broadcast(input_tensor, src=0)
    input_tensor = torch.chunk(input_tensor, 2, dim=0)[gpc.get_local_rank(ParallelMode.PARALLEL_3D_WEIGHT)]
    input_tensor = torch.chunk(input_tensor, 2, dim=0)[gpc.get_local_rank(ParallelMode.PARALLEL_3D_INPUT)]
    input_tensor = torch.chunk(input_tensor, 2, dim=-1)[gpc.get_local_rank(ParallelMode.PARALLEL_3D_OUTPUT)]
    timer = Timer()
    iter_times = args.num_steps
    timer.start()
    for i in range(iter_times):
        input_tensor = model(input_tensor)
        synchronize()
    result_3d = timer.stop()
    return result_3d
