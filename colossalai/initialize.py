#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os

# set CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that when overlapping communication and computation,
# the order of of kernel launches on GPUs are the same as on the CPU so that comm is launched first.
# see https://github.com/NVIDIA/Megatron-LM/issues/533
# https://forums.developer.nvidia.com/t/how-many-streams-maximum-number-of-streams/6571/16
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

import torch.distributed as dist

from colossalai.accelerator import get_accelerator
from colossalai.logging import get_dist_logger
from colossalai.utils import set_seed


def launch(
    rank: int,
    world_size: int,
    host: str,
    port: int,
    backend: str = "nccl",
    local_rank: int = None,
    seed: int = 1024,
    verbose: bool = True,
):
    """This function first parses the configuration arguments, using :func:`parse_args()` in case one of the input
    arguments are not given. Then initialize and set distributed environment by calling global_context's functions.

    Args:
        config (Union[str, dict, Config]): Config file or config file path are both acceptable
        rank (int): Rank for the default process group
        world_size (int): World size of the default process group
        host (str): The master address for distributed training
        port (str): The master port for distributed training
        backend (str, optional): Backend for ``torch.distributed``, defaults to ``nccl``
        local_rank (int, optional):
            Rank for the process on the node and is used to set the default CUDA device,
            defaults to None. If local_rank = None, the default device ordinal will be calculated automatically.
        seed (int, optional): Specified random seed for every process. Defaults to 1024.
        verbose (bool, optional): Whether to print logs. Defaults to True.

    Raises:
        Exception: Raise exception when config type is wrong
    """

    cur_accelerator = get_accelerator()

    backend = cur_accelerator.communication_backend

    # init default process group
    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    dist.init_process_group(rank=rank, world_size=world_size, backend=backend, init_method=init_method)

    # set cuda device
    # if local rank is not given, calculate automatically
    if cur_accelerator.support_set_device:
        cur_accelerator.set_device(local_rank)

    set_seed(seed)

    if verbose:
        logger = get_dist_logger()
        logger.info(f"Distributed environment is initialized, world size: {dist.get_world_size()}", ranks=[0])


def launch_from_slurm(
    host: str,
    port: int,
    backend: str = "nccl",
    seed: int = 1024,
    verbose: bool = True,
):
    """A wrapper for colossalai.launch for SLURM launcher by reading rank and world size from the environment variables
    set by SLURM

    Args:
        config (Union[str, dict, Config]): Config file or config file path are both acceptable
        host (str): The master address for distributed training
        port (str): The master port for distributed training
        backend (str, optional): Backend for ``torch.distributed``, defaults to ``nccl``
        seed (int, optional): Specified random seed for every process. Defaults to 1024.
        verbose (bool, optional): Whether to print logs. Defaults to True.
    """
    try:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NPROCS"])
    except KeyError as e:
        raise RuntimeError(
            f"Could not find {e} in the SLURM environment, visit https://www.colossalai.org/ for more information on launching with SLURM"
        )

    launch(
        rank=rank,
        world_size=world_size,
        host=host,
        port=port,
        backend=backend,
        seed=seed,
        verbose=verbose,
    )


def launch_from_openmpi(
    host: str,
    port: int,
    backend: str = "nccl",
    seed: int = 1024,
    verbose: bool = True,
):
    """A wrapper for colossalai.launch for OpenMPI launcher by reading rank and world size from the environment variables
    set by OpenMPI

    Args:
        config (Union[str, dict, Config]): Config file or config file path are both acceptable
        host (str): The master address for distributed training
        port (str): The master port for distributed training
        backend (str, optional): Backend for ``torch.distributed``, defaults to ``nccl``
        seed (int, optional): Specified random seed for every process. Defaults to 1024.
        verbose (bool, optional): Whether to print logs. Defaults to True.
    """
    try:
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    except KeyError as e:
        raise RuntimeError(
            f"Could not find {e} in the OpenMPI environment, visit https://www.colossalai.org/ for more information on launching with OpenMPI"
        )

    launch(
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
        host=host,
        port=port,
        backend=backend,
        seed=seed,
        verbose=verbose,
    )


def launch_from_torch(backend: str = "nccl", seed: int = 1024, verbose: bool = True):
    """A wrapper for colossalai.launch for torchrun or torch.distributed.launch by reading rank and world size
    from the environment variables set by PyTorch

    Args:
        config (Union[str, dict, Config]): Config file or config file path are both acceptable
        backend (str, optional): Backend for ``torch.distributed``, defaults to ``nccl``
        seed (int, optional): Specified random seed for every process. Defaults to 1024.
        verbose (bool, optional): Whether to print logs. Defaults to True.
    """
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        host = os.environ["MASTER_ADDR"]
        port = int(os.environ["MASTER_PORT"])
    except KeyError as e:
        raise RuntimeError(
            f"Could not find {e} in the torch environment, visit https://www.colossalai.org/ for more information on launching with torch"
        )

    launch(
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
        host=host,
        port=port,
        backend=backend,
        seed=seed,
        verbose=verbose,
    )
