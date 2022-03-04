#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import os
import pprint
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from colossalai.amp import AMP_TYPE, convert_to_amp
from colossalai.amp.naive_amp import NaiveAMPModel
from colossalai.builder.builder import build_gradient_handler
from colossalai.context import Config, ConfigException, ParallelMode
from colossalai.core import global_context as gpc
from colossalai.engine import Engine
from colossalai.global_variables import moe_env
from colossalai.logging import get_dist_logger
from colossalai.nn.optimizer.colossalai_optimizer import ColossalaiOptimizer
from colossalai.utils import (accumulate_gradient, get_current_device, is_using_ddp, is_using_pp, is_using_sequence,
                              sync_model_param)
from colossalai.zero import convert_to_zero, ShardedOptimizer
from colossalai.engine.ophooks import register_ophooks_recursively, BaseOpHook


def get_default_parser():
    """Reads user command line and uses an argument parser to parse the input arguments.
    Input arguments include configuration, host, port, world size, local rank, backend for torch.distributed.

    :return: Returns the parser with the default arguments, the user may add customized arguments into this parser
    :rtype: Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='path to the config file')
    parser.add_argument('--host', type=str, help='the master address for distributed training')
    parser.add_argument('--port', type=int, help='the master port for distributed training')
    parser.add_argument('--world_size', type=int, help='world size for distributed training')
    parser.add_argument('--rank', type=int, help='rank for the default process group')
    parser.add_argument('--local_rank', type=int, help='local rank on the node')
    parser.add_argument('--backend', type=str, default='nccl', help='backend for distributed communication')
    return parser


def launch(config: Union[str, Path, Config, Dict],
           rank: int,
           world_size: int,
           host: str,
           port: int,
           backend: str = 'nccl',
           local_rank: int = None,
           seed: int = 1024,
           verbose: bool = True):
    """This function first parses the configuration arguments, using :func:`parse_args()` in case one of the input
    arguments are not given. Then initialize and set distributed environment by calling global_context's functions.

    :param config: Config file or config file path are both acceptable
    :type config: Union[str, dict, Config]
    :param rank: Rank for the default process group
    :type rank: int
    :param world_size: World size of the default process group
    :type world_size: int
    :param host: The master address for distributed training
    :type host: str
    :param port: The master port for distributed training
    :type port: str
    :param backend: Backend for torch.distributed
    :type backend: str, optional
    :param local_rank: Rank for the process on the node and is used to set the default CUDA device, defaults to None.
        If local_rank = None, the default device ordinal will be calculated automatically
    :type local_rank: int, optional
    :param seed: Specified random seed for every processes
    :type seed: int, optional
    :param verbose: Whether to print logs
    :type verbose: bool, optional
    :raises Exception: Raise exception when config type is wrong
    """
    gpc.verbose = verbose

    # set config
    assert isinstance(config, (Config, str, Path, dict)), \
        f'expected argument config to be Config, str or Path, but got {type(config)}'
    if not isinstance(config, Config) and isinstance(config, dict):
        config = Config(config)
    if isinstance(config, (str, Path)):
        config = Config.from_file(config)
    gpc.load_config(config)

    # init default process group
    gpc.init_global_dist(rank, world_size, backend, host, port)

    # init process groups for different parallel modes from config
    gpc.init_parallel_groups()

    # set cuda device
    if torch.cuda.is_available():
        # if local rank is not given, calculate automatically
        gpc.set_device(local_rank)

    gpc.set_seed(seed)

    if verbose:
        logger = get_dist_logger()
        logger.info(
            f'Distributed environment is initialized, '
            f'data parallel size: {gpc.data_parallel_size}, pipeline parallel size: {gpc.pipeline_parallel_size}, '
            f'tensor parallel size: {gpc.tensor_parallel_size}',
            ranks=[0])


def launch_from_slurm(config: Union[str, Path, Config, Dict],
                      host: str,
                      port: int,
                      backend: str = 'nccl',
                      seed: int = 1024,
                      verbose: bool = True):
    """A wrapper for colossalai.launch for SLURM launcher by reading rank and world size from the environment variables
    set by SLURM

    :param config: Config file or config file path are both acceptable
    :type config: Union[str, dict, Config]
    :param host: The master address for distributed training
    :type host: str
    :param port: The master port for distributed training
    :type port: str
    :param backend: Backend for torch.distributed
    :type backend: str, optional
    :param seed: Specified random seed for every processes
    :type seed: int, optional
    :param verbose: Whether to print logs
    :type verbose: bool, optional
    """
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NPROCS'])
    launch(config=config,
           rank=rank,
           world_size=world_size,
           host=host,
           port=port,
           backend=backend,
           seed=seed,
           verbose=verbose)


def launch_from_openmpi(config: Union[str, Path, Config, Dict],
                        host: str,
                        port: int,
                        backend: str = 'nccl',
                        seed: int = 1024,
                        verbose: bool = True):
    """A wrapper for colossalai.launch for OpenMPI launcher by reading rank and world size from the environment variables
    set by OpenMPI

    :param config: Config file or config file path are both acceptable
    :type config: Union[str, dict, Config]
    :param host: The master address for distributed training
    :type host: str
    :param port: The master port for distributed training
    :type port: str
    :param backend: Backend for torch.distributed
    :type backend: str, optional
    :param seed: Specified random seed for every processes
    :type seed: int, optional
    :param verbose: Whether to print logs
    :type verbose: bool, optional
    """
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    launch(config=config,
           local_rank=local_rank,
           rank=rank,
           world_size=world_size,
           host=host,
           port=port,
           backend=backend,
           seed=seed,
           verbose=verbose)


def launch_from_torch(config: Union[str, Path, Config, Dict],
                      backend: str = 'nccl',
                      seed: int = 1024,
                      verbose: bool = True):
    """A wrapper for colossalai.launch for torchrun or torch.distributed.launch by reading rank and world size
    from the environment variables set by PyTorch

    :param config: Config file or config file path are both acceptable
    :type config: Union[str, dict, Config]
    :param backend: Backend for torch.distributed
    :type backend: str, optional
    :param seed: Specified random seed for every processes
    :type seed: int, optional
    :param verbose: Whether to print logs
    :type verbose: bool, optional
    """
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    host = os.environ['MASTER_ADDR']
    port = int(os.environ['MASTER_PORT'])
    launch(config=config,
           local_rank=local_rank,
           rank=rank,
           world_size=world_size,
           host=host,
           port=port,
           backend=backend,
           seed=seed,
           verbose=verbose)


def initialize(model: Union[nn.Module, List[nn.Module]],
               optimizer: Union[Optimizer, List[Optimizer]],
               criterion: Union[_Loss, List[_Loss]],
               train_dataloader: Optional[Union[Iterable, List[Iterable]]] = None,
               test_dataloader: Optional[Union[Iterable, List[Iterable]]] = None,
               lr_scheduler: _LRScheduler = None,
               ophooks: List[BaseOpHook] = [],
               verbose: bool = True
               ) -> Tuple[Engine, DataLoader, DataLoader, _LRScheduler]:
    """Core function to wrap the essential training components with our functionality based on the config which is
    loaded into gpc.config.

    :param model: Your model instance
    :type model: :class:`torch.nn.Module`
    :param optimizer: Your optimizer instance
    :type optimizer: :class:`torch.optim.optimizer.Optimizer`
    :param criterion: Your criterion instance
    :type criterion: :class:`torch.nn.modules.loss._Loss`
    :param train_dataloader: Dataloader for training
    :type train_dataloader: :class:`torch.utils.data.DataLoader`, optional
    :param test_dataloader: Dataloader for testing
    :type test_dataloader: :class:`torch.utils.data.DataLoader`, optional
    :param lr_scheduler: Your lr scheduler instance
    :type lr_scheduler: :class:`torch.nn.lr_scheduler._LRScheduler`, optional
    :param verbose: Whether to print logs
    :type verbose: bool, optional
    :return: (engine, train_dataloader, test_dataloader, lr_scheduler)
    :rtype: Tuple
    """
    # get logger
    logger = get_dist_logger()
    gpc.verbose = verbose

    # get config from gpc
    config = gpc.config

    # print config
    if verbose:
        logger.info(
            f"\n========== Your Config ========\n"
            f"{pprint.pformat(gpc.config)}\n"
            f"================================\n",
            ranks=[0])

    # cudnn
    cudnn_benchmark = config.get('cudnn_benchmark', True)
    cudnn_deterministic = config.get('cudnn_deterministic', False)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic
    if verbose:
        logger.info(f"cuDNN benchmark = {cudnn_benchmark}, deterministic = {cudnn_deterministic}", ranks=[0])

    # first sync model across dp ranks
    model.to(get_current_device())
    use_zero3 = hasattr(gpc.config, 'zero') and gpc.config.zero.level == 3
    if not moe_env.is_initialized() and not use_zero3:
        if is_using_sequence():
            sync_model_param(model, ParallelMode.SEQUENCE_DP)
        elif is_using_ddp():
            sync_model_param(model, ParallelMode.DATA)
    else:
        logger.warning(
            "The parameters of models is not automatically synchronized.\n"
            "Please make sure that all parameters are the same in data parallel group.",
            ranks=[0])

    # check amp and zero
    fp16_cfg = gpc.config.get('fp16', None)
    zero_cfg = gpc.config.get('zero', None)

    if fp16_cfg is not None and fp16_cfg.mode is not None and zero_cfg is not None:
        raise ConfigException(
            "It is not allowed to set fp16 and zero configuration in your config file at the same time")

    # clip grad norm
    clip_grad_norm = gpc.config.get('clip_grad_norm', 0.0)
    if clip_grad_norm > 0:
        if zero_cfg is not None:
            raise ConfigException(
                "clip_grad_norm should be specified with zero, you should specify clip_grad in zero configuration")

    # initialize amp
    amp_mode = None
    if fp16_cfg is not None and fp16_cfg.mode is not None:
        cfg_ = fp16_cfg.copy()
        amp_mode = cfg_.pop('mode')
        if is_using_pp():
            assert amp_mode == AMP_TYPE.NAIVE, 'Pipeline only support NaiveAMP currently'
        if amp_mode == AMP_TYPE.NAIVE:
            cfg_['clip_grad'] = clip_grad_norm
        model, optimizer, criterion = convert_to_amp(model=model,
                                                     optimizer=optimizer,
                                                     criterion=criterion,
                                                     mode=amp_mode,
                                                     amp_config=cfg_)

    if zero_cfg is not None:
        cfg_ = zero_cfg.copy()
        level = cfg_.pop('level')
        model, optimizer = convert_to_zero(model=model, optimizer=optimizer, level=level, zero_config=cfg_)

    # gradient handler
    gradient_handler_cfg = gpc.config.get('gradient_handler', None)
    if gradient_handler_cfg is None:
        # if gradient handler is not specified in the configuration file,
        # check in the following order
        # 1. if optimizer is ZERO, then use zero grad handler
        # 2. if dp size is larger than 1 and pipeline is not used, use pytorch ddp
        # 3. if using pipeline and dp size larger than 1, use data parallel grad handler
        if isinstance(optimizer, ShardedOptimizer):
            gradient_handler_cfg = [dict(type='ZeROGradientHandler')]
            if verbose:
                logger.info(
                    "Training with zero is detected, ZeROGradientHandler is automatically "
                    "added even though not specified in the configuration",
                    ranks=[0])
        elif is_using_ddp() and moe_env.is_initialized():
            gradient_handler_cfg = [dict(type='MoeGradientHandler')]
            if verbose:
                logger.info(
                    "Data parallel training is detected with moe parallel, MoeGradientHandler is automatically "
                    "added even though not specified in the configuration",
                    ranks=[0])
        elif is_using_sequence():
            model = DDP(model,
                        process_group=gpc.get_group(ParallelMode.SEQUENCE_DP),
                        device_ids=[torch.cuda.current_device()])
            if verbose:
                logger.info('Model is using torch.nn.parallel.DistributedDataParallel for Sequence Parallelism',
                            ranks=[0])
        elif is_using_ddp() and not is_using_pp() and amp_mode != AMP_TYPE.NAIVE:
            model = DDP(model, process_group=gpc.get_group(ParallelMode.DATA), device_ids=[torch.cuda.current_device()])
            if verbose:
                logger.info('Model is using torch.nn.parallel.DistributedDataParallel for Data Parallelism', ranks=[0])
        elif is_using_ddp():
            gradient_handler_cfg = [dict(type='DataParallelGradientHandler')]
            if verbose:
                logger.info(
                    "Data parallel training is detected when using pipeline parallel, "
                    "DataParallelGradientHandler is automatically "
                    "added even though not specified in the configuration",
                    ranks=[0])
        # add pipeline parallel gradient handler, if pipeline shared module is detected
        for param in model.parameters():
            if getattr(param, 'pipeline_shared_module_pg', None) is not None:
                if gradient_handler_cfg is None:
                    gradient_handler_cfg = [dict(type='PipelineSharedModuleGradientHandler')]
                else:
                    gradient_handler_cfg.append(dict(type='PipelineSharedModuleGradientHandler'))
                if verbose:
                    logger.info(
                        "pipeline_shared_module is detected, PipelineSharedModuleGradientHandler is automatically "
                        "added even though not specified in the configuration",
                        ranks=[0])
                break
    else:
        if not isinstance(gradient_handler_cfg, list):
            raise ConfigException(
                f"expected gradient_handler in the configuration file to be a list but got {type(gradient_handler_cfg)}"
            )

    # turn off sync buffer for NaiveAMPModel if using torch DDP and NaiveAMPModel at the same time
    # to avoid duplicated buffer synchronization
    if isinstance(model, DDP) and isinstance(model.module, NaiveAMPModel):
        model.module.sync_buffer = False

    if gradient_handler_cfg is None:
        gradient_handlers = None
        if verbose and not isinstance(model, DDP):
            logger.warning(
                "No PyTorch DDP or gradient handler is set up, please make sure you do not need "
                "to all-reduce the gradients after a training step.",
                ranks=[0])
    else:
        gradient_handlers = [build_gradient_handler(cfg, model, optimizer) for cfg in gradient_handler_cfg]

    # check if optimizer is ColossalaiOptimizer
    if not isinstance(optimizer, (ColossalaiOptimizer, ShardedOptimizer)):
        optimizer = ColossalaiOptimizer(optim=optimizer)

    # gradient accumulation
    grad_accum_size = gpc.config.get('gradient_accumulation', None)
    if grad_accum_size is not None:
        optimizer, train_dataloader, gradient_handlers, lr_scheduler = accumulate_gradient(model=model,
                                                                                           optimizer=optimizer,
                                                                                           dataloader=train_dataloader,
                                                                                           accumulate_size=grad_accum_size,
                                                                                           gradient_handlers=gradient_handlers,
                                                                                           lr_scheduler=lr_scheduler)

    engine = Engine(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        gradient_handlers=gradient_handlers,
        clip_grad_norm=clip_grad_norm,
        ophook_list=ophooks
    )

    return engine, train_dataloader, test_dataloader, lr_scheduler
