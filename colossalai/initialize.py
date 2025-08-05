#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import time
import socket
from datetime import timedelta

# set CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that when overlapping communication and computation,
# the order of of kernel launches on GPUs are the same as on the CPU so that comm is launched first.
# see https://github.com/NVIDIA/Megatron-LM/issues/533
# https://forums.developer.nvidia.com/t/how-many-streams-maximum-number-of-streams/6571/16
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

import torch
import torch.distributed as dist

from colossalai.accelerator import get_accelerator
from colossalai.logging import get_dist_logger
from colossalai.utils import set_seed


def _wait_for_master_ready(host: str, port: int, timeout: int = 300, retry_interval: int = 5) -> bool:
    """
    Wait for the master node to be ready for distributed training connections.
    
    This is particularly useful in Kubernetes environments where pods start at different times.
    
    Args:
        host (str): Master node hostname or IP address
        port (int): Master node port
        timeout (int): Maximum time to wait in seconds (default: 300)
        retry_interval (int): Time between connection attempts in seconds (default: 5)
        
    Returns:
        bool: True if master is ready, False if timeout exceeded
    """
    start_time = time.time()
    logger = get_dist_logger()
    
    while time.time() - start_time < timeout:
        try:
            # Attempt to connect to the master node
            sock = socket.create_connection((host, port), timeout=10)
            sock.close()
            logger.info(f"Master node {host}:{port} is ready for connections", ranks=[0])
            return True
        except (socket.error, socket.timeout, ConnectionRefusedError, OSError) as e:
            logger.debug(f"Waiting for master node {host}:{port} to be ready... ({e})", ranks=[0])
            time.sleep(retry_interval)
    
    logger.error(f"Master node {host}:{port} did not become ready within {timeout} seconds", ranks=[0])
    return False


def _get_distributed_timeout() -> timedelta:
    """
    Get the distributed training timeout from environment variables or use sensible defaults.
    
    Returns:
        timedelta: Timeout for distributed training initialization
    """
    # Check for user-defined timeout (in seconds)
    timeout_seconds = int(os.environ.get("COLOSSALAI_DIST_TIMEOUT", "1800"))  # 30 minutes default
    return timedelta(seconds=timeout_seconds)


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
    
    logger = get_dist_logger() if verbose else None

    # Wait for master node to be ready (especially important for K8s environments)
    if rank != 0:  # Non-master ranks should wait for master to be ready
        if logger:
            logger.info(f"Rank {rank}: Waiting for master node {host}:{port} to be ready...")
        
        master_ready_timeout = int(os.environ.get("COLOSSALAI_MASTER_READY_TIMEOUT", "300"))
        if not _wait_for_master_ready(host, port, timeout=master_ready_timeout):
            raise RuntimeError(f"Master node {host}:{port} is not ready for connections after {master_ready_timeout} seconds")

    # init default process group with enhanced timeout and error handling
    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    
    # Get timeout from environment or use default
    timeout = _get_distributed_timeout()
    
    if logger:
        logger.info(f"Initializing distributed process group: rank={rank}, world_size={world_size}, "
                   f"backend={backend}, init_method={init_method}, timeout={timeout}")
    
    try:
        dist.init_process_group(
            rank=rank, 
            world_size=world_size, 
            backend=backend, 
            init_method=init_method,
            timeout=timeout
        )
    except Exception as e:
        if logger:
            logger.error(f"Failed to initialize distributed process group: {e}")
            logger.error(f"Please check: 1) Master node {host}:{port} is accessible, "
                        f"2) All nodes use the same MASTER_ADDR/MASTER_PORT, "
                        f"3) Network connectivity between nodes")
        raise RuntimeError(f"Distributed initialization failed: {e}") from e

    # set cuda device
    # if local rank is not given, calculate automatically
    if cur_accelerator.support_set_device:
        cur_accelerator.set_device(local_rank)

    set_seed(seed)

    try:
        torch._dynamo.config.optimize_ddp = world_size > 1
    except AttributeError:
        pass

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
    logger = get_dist_logger() if verbose else None
    
    # Validate required environment variables with detailed error messages
    required_envs = {
        "RANK": "Global rank of the current process",
        "LOCAL_RANK": "Local rank of the process on the current node", 
        "WORLD_SIZE": "Total number of processes across all nodes",
        "MASTER_ADDR": "IP address or hostname of the master node",
        "MASTER_PORT": "Port number for distributed communication"
    }
    
    missing_envs = []
    for env_var, description in required_envs.items():
        if env_var not in os.environ:
            missing_envs.append(f"  - {env_var}: {description}")
    
    if missing_envs:
        error_msg = ("Missing required environment variables for distributed training:\n" + 
                    "\n".join(missing_envs) + 
                    "\n\nFor Kubernetes multi-node training, ensure you're using enhanced torchrun command:\n"
                    "torchrun --nnodes=N --nproc_per_node=M --rdzv_backend=c10d \\\n"
                    "  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --rdzv_id=$JOB_ID \\\n"
                    "  --node_rank=$NODE_RANK your_script.py\n\n"
                    "Visit https://www.colossalai.org/ for more information on launching with torch")
        raise RuntimeError(error_msg)

    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        host = os.environ["MASTER_ADDR"]
        port = int(os.environ["MASTER_PORT"])
    except ValueError as e:
        raise RuntimeError(f"Invalid environment variable value: {e}. All rank and port values must be integers.")
    
    # Additional validation for common misconfigurations
    if rank >= world_size:
        raise RuntimeError(f"RANK ({rank}) must be less than WORLD_SIZE ({world_size})")
    
    if local_rank < 0:
        raise RuntimeError(f"LOCAL_RANK ({local_rank}) must be non-negative")
        
    if port < 1024 or port > 65535:
        raise RuntimeError(f"MASTER_PORT ({port}) must be between 1024 and 65535")
    
    # Log distributed training configuration for debugging
    if logger and verbose:
        logger.info(f"Starting distributed training with configuration:")
        logger.info(f"  RANK: {rank}")
        logger.info(f"  LOCAL_RANK: {local_rank}")
        logger.info(f"  WORLD_SIZE: {world_size}")
        logger.info(f"  MASTER_ADDR: {host}")
        logger.info(f"  MASTER_PORT: {port}")
        logger.info(f"  BACKEND: {backend}")
        
        # Log additional environment variables that might be relevant for debugging
        debug_envs = ["NODE_RANK", "NCCL_DEBUG", "GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME", "RDZV_ID"]
        for env_var in debug_envs:
            if env_var in os.environ:
                logger.info(f"  {env_var}: {os.environ[env_var]}")

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
