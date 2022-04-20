import torch
import inspect
import os
import subprocess
import sys

from colossalai.initialize import launch_from_torch
from colossalai.logging import disable_existing_loggers
from colossalai.utils import print_rank_0
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.utils import free_port
from colossalai.cli.benchmark import build_args_parser, build_configs, \
        build_input_tensor, profile_1d, profile_2d, profile_2p5d, profile_3d, \
        BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM, ITER_TIMES


def launch(args=None):
    train_script = inspect.getfile(inspect.currentframe())
    assert args is not None, "args should not be None"
    env = os.environ.copy()
    if args.num_gpus == -1 or args.num_gpus > torch.cuda.device_count():
            nproc_per_node = torch.cuda.device_count()
    else:
        nproc_per_node = args.num_gpus

    train_args = [f"--num_gpus={nproc_per_node}"]
    if args.bs != BATCH_SIZE:
        train_args.append(f"--bs={args.bs}")
    if args.hid_dim != HIDDEN_DIM:
        train_args.append(f"--hid_dim={args.hid_dim}")
    if args.num_steps != ITER_TIMES:
        train_args.append(f"--num_steps={args.num_steps}")
    if args.seq_len != SEQ_LENGTH:
        train_args.append(f"--seq_len={args.seq_len}")
    
    master_port = free_port()
    if torch.__version__ <= "1.09":
        cmd = [sys.executable, "-u", "-m", 
                "torch.distributed.launch",
                f"--nproc_per_node={nproc_per_node}",
                f"--master_port={master_port}"] + [train_script] + train_args
    else:
        cmd = ["torchrun",
                f"--nproc_per_node={nproc_per_node}",
                f"--master_port={master_port}"] + [train_script] + train_args

    result = subprocess.Popen(cmd, env=env)
    result.wait()
    if result.returncode > 0:
        sys.exit(result.returncode)

def main():
    parser = build_args_parser()
    args = parser.parse_args()
    disable_existing_loggers()
    logger = get_dist_logger()
    launch_from_torch(config={}, verbose=False)
    input_tensor = build_input_tensor(args)
    config_dict = build_configs(args)
    if len(config_dict) == 0:
        print_rank_0(f"WARNING: We need at least two devices to profile TP strategies performance.")
        gpc.destroy()
        return
    for parallel_mode, config in config_dict.items():
        if parallel_mode == "1d":
            result_1d = profile_1d(input_tensor, config, args)
            print_rank_0(f"INFO: Totoal time cost in 1D TP is {result_1d}.")
        if parallel_mode == "2d":
            result_2d = profile_2d(input_tensor, config, args)
            print_rank_0(f"INFO: Totoal time cost in 2D TP is {result_2d}.")
        if parallel_mode == "2p5d":
            result_2p5d = profile_2p5d(input_tensor, config, args)
            print_rank_0(f"INFO: Totoal time cost in 2P5D TP is {result_2p5d}.")
        if parallel_mode == "3d":
            result_3d = profile_3d(input_tensor, config, args)
            print_rank_0(f"INFO: Totoal time cost in 3D TP is {result_3d}.")
    if "2d" not in config_dict:
        print_rank_0(f"WARNING: To use 2D tensor parallel, you have to provide at least 4 computing devices.")
    if "2p5d" not in config_dict:
        print_rank_0(f"WARNING: To use 2P5D tensor parallel, you have to provide at least 8 computing devices.")
        print_rank_0(f"WARNING: To use 3D tensor parallel, you have to provide at least 8 computing devices.")
    gpc.destroy()

if __name__=="__main__":
    main()
