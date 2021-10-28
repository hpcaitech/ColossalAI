#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import pprint
import random
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from colossalai.engine import AMP_TYPE, NoPipelineSchedule, PipelineSchedule
from colossalai.logging import get_global_dist_logger, init_global_dist_logger
from colossalai.nn import DataParallelSampler
from colossalai.nn.model.base_model import BaseModel
from .builder import (ModelInitializer, build_dataset, build_loss,
                      build_lr_scheduler, build_model, build_optimizer,
                      build_optimizer_wrapper)
from .context import Config, ParallelMode
from .core import global_context as gpc
from .utils import get_current_device, sync_model_param_in_dp


def parse_args():
    '''Reads user command line and uses an argument parser to parse the input arguments.
    Input arguments include configuration, host, port, world size, local rank, backend for torch.distributed.

    :return: call the parse arguments function
    :rtype: Namespace
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='path to the config file')
    parser.add_argument('--host',
                        type=str,
                        default=None,
                        help='the master address for distributed training')
    parser.add_argument('--port',
                        type=str,
                        default=None,
                        help='the master port for distributed training')
    parser.add_argument('--world_size', type=int, help='world size for ')
    parser.add_argument('--local_rank',
                        type=int,
                        help='rank for the default process group')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='backend for torch.distributed')
    return parser.parse_args()


def init_dist(config: Union[str, dict] = None,
              local_rank: int = None,
              world_size: int = None,
              host: str = None,
              port: str = None,
              backend: str = None):
    '''This function first parses the configuration arguments, using :func:parse_args() in case one of the input arguments are not given.
    Then initialize and set distributed environment by calling global_context's functions. 

    :param config: config file or config file path are both acceptable
    :type config: Union[str, dict], optional
    :param local_rank: rank for the default process group, defaults to None
    :type local_rank: int, optional
    :param world_size: world size of GPUs, defaults to None
    :type world_size: int, optional
    :param host: the master address for distributed training, defaults to None
    :type host: str, optional
    :param port: the master port for distributed training, defaults to None
    :type port: str, optional
    :param backend: backend for torch.distributed, defaults to None
    :type backend: str, optional
    :raises Exception: raise exception when config type is wrong
    '''
    args = [config, local_rank, world_size, host, port, backend]
    arg_given = [arg is not None for arg in args]

    if not all(arg_given):
        args = parse_args()

    if config is None:
        config = args.config
    if local_rank is None:
        local_rank = args.local_rank
    if world_size is None:
        world_size = args.world_size
    if host is None:
        host = args.host
    if port is None:
        port = args.port
    if backend is None:
        backend = args.backend
    args = Config(
        dict(config=config,
             host=host,
             port=port,
             world_size=world_size,
             local_rank=local_rank,
             backend=backend))

    # set distributed settings
    dist_args = Config(
        dict(local_rank=args.local_rank,
             world_size=args.world_size,
             backend=args.backend))

    gpc.set_dist_args(dist_args)

    # set config
    if isinstance(args.config, dict):
        cfg = args.config
    elif isinstance(args.config, (str, Path)):
        cfg = Config.from_file(args.config)
    else:
        raise Exception('Config type error: {}'.format(type(args.config)))
    gpc.load_config(cfg)

    # init dist groups
    gpc.init_global_dist(args.host, args.port)
    gpc.init_parallel_groups()

    # init dist logger
    init_global_dist_logger()

    # set cuda device
    if torch.cuda.is_available():
        gpc.set_device()


def get_dataloader(dataset, seed=1024, add_sampler_if_possible=False, **kwargs):
    '''Set up a deterministic dataloader (also configure seed workers, samplers and whether shuffle or not)

    .. note: when pipeline parallel is enabled, shuffle cannot be True 
        as it will result in mismatch between input data on the 1st
        stage and label on the last stage

    :param dataset: a :class:utils.data.dataset dataset
    :param seed: random worker seed, defaults to 1024
    :type seed: int, optional
    :param add_sampler_if_possible: [description], defaults to False
    :type add_sampler_if_possible: bool, optional
    :return: a :class:utils.data.dataset dataloader
    :rtype: torch.utils.data.dataset
    '''
    _kwargs = kwargs.copy()
    if 'shuffle' in _kwargs:
        shuffle = _kwargs.pop('shuffle')
    else:
        shuffle = False

    if add_sampler_if_possible and gpc.is_initialized(ParallelMode.DATA) and gpc.get_world_size(ParallelMode.DATA) > 1:
        sampler = DataParallelSampler(dataset, shuffle=shuffle)
    else:
        sampler = None

    # Deterministic dataloader
    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    if sampler is None:
        return DataLoader(dataset,
                          worker_init_fn=seed_worker,
                          shuffle=shuffle,
                          **_kwargs)
    else:
        return DataLoader(dataset,
                          sampler=sampler,
                          worker_init_fn=seed_worker,
                          **_kwargs)


def initialize(config: Union[str, dict] = None,
               local_rank: int = None,
               world_size: int = None,
               host: str = None,
               port: str = None,
               backend: str = None,
               train_dataloader: Optional[Union[Iterable, Callable]] = None,
               test_dataloader: Optional[Union[Iterable, Callable]] = None,
               ):
    '''Core function that initializes distributed environment, logger, cudnn, data, model, loss function, optimizer, and lr_scheduler(their configs are in gpc.config).

    :param config: config file or config file path are both acceptable
    :type config: Union[str, dict], optional
    :param local_rank: rank for the default process group, defaults to None
    :type local_rank: int, optional
    :param world_size: world size of GPUs, defaults to None
    :type world_size: int, optional
    :param host: the master address for distributed training, defaults to None
    :type host: str, optional
    :param port: the master port for distributed training, defaults to None
    :type port: str, optional
    :param backend: backend for torch.distributed, defaults to None
    :type backend: str, optional
    :param train_dataloader: If None, the config is used to build a dataloder; Else, it should be a dataloader object or a function with no arguments which can build a dataloader, defaults to None
    :type train_dataloader: Optional[Union[Iterable, Callable]], optional
    :param test_dataloader: If None, the config is used to build a dataloder; Else, it should be a dataloader object or a function with no arguments which can build a dataloader, defaults to None
    :type test_dataloader: Optional[Union[Iterable, Callable]], optional
    :return: (model, train_dataloader, test_dataloader, criterion, optimizer, schedule, lr_scheduler)
    :rtype: tuple
    '''
    # initialize distributed environment
    init_dist(config=config,
              local_rank=local_rank,
              world_size=world_size,
              host=host,
              port=port,
              backend=backend)

    # init logger
    logger = get_global_dist_logger()
    logger.info(f'Distributed environment is initialized, '
                f'data parallel size: {gpc.data_parallel_size}, pipeline parallel size: {gpc.pipeline_parallel_size}, '
                f'tensor parallel size: {gpc.tensor_parallel_size}', ranks=[0])

    # print config
    logger.info(f"\n========== Your Config ========\n"
                f"{pprint.pformat(gpc.config)}\n"
                f"================================", ranks=[0])

    # cudnn
    cudnn_benchmark = gpc.config.get('cudnn_benchmark', True)
    cudnn_deterministic = gpc.config.get('cudnn_deterministic', False)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic
    logger.info(
        f"cuDNN benchmark = {cudnn_benchmark}, deterministic = {cudnn_deterministic}", ranks=[0])

    # set seed, cuda seed is only set when cuda is avail
    gpc.set_seed()

    # return_items = list()

    # check fp16 and zero
    should_convert_model_to_half = False
    should_wrap_fp16_optimizer = False
    should_wrap_zero_optimizer_level_2_3 = False

    if hasattr(gpc.config, 'fp16'):
        fp16_mode = gpc.config.fp16.mode
        if fp16_mode == AMP_TYPE.PARALLEL:
            should_convert_model_to_half = True
            should_wrap_fp16_optimizer = True

    if hasattr(gpc.config, 'zero'):
        should_wrap_zero_optimizer_level_2_3 = True
        zero_type = gpc.config.zero.type
        if zero_type in ['ZeroRedundancyOptimizer_Level_2', 'ZeroRedundancyOptimizer_Level_3']:
            should_convert_model_to_half = True
            assert not should_wrap_fp16_optimizer, \
                'AMP_TYPE.PARALLEL is mutually exclusive with zero level 2 and 3'

    # build model
    logger.info('Building model ...', ranks=[0])
    assert hasattr(
        gpc.config, 'model'), "Build error: configuration 'model' is missing"
    if gpc.pipeline_parallel_size > 1:
        model = ModelInitializer(gpc.config.model, 1, verbose=True)
        model = model.model_initialize()
    else:
        model = build_model(gpc.config.model)
        if isinstance(model, BaseModel):
            model.build_from_cfg()
        model = model.to(get_current_device())
    sync_model_param_in_dp(model)
    logger.info('Model is created', ranks=[0])

    if should_convert_model_to_half:
        model = model.half()
        logger.info("Model is cast to fp16", ranks=[0])

    # training data
    if callable(train_dataloader):
        logger.info(
            f'Build train data loader from {train_dataloader}', ranks=[0])
        train_dataloader = train_dataloader()
    if train_dataloader is None and hasattr(gpc.config, 'train_data'):
        logger.info('Preparing data ...', ranks=[0])
        # assert hasattr(gpc.config, 'train_data'), "Build error: configuration 'train_data' is missing."
        train_dataset = build_dataset(gpc.config.train_data.dataset)
        logger.info('Train dataset is ready.', ranks=[0])

        train_dataloader = get_dataloader(train_dataset,
                                          gpc.config.get('seed', 1024),
                                          True,
                                          **gpc.config.train_data.dataloader,
                                          )
        logger.info(
            f'Loaded {len(train_dataset)} samples in {len(train_dataloader)} batches for training', ranks=[0])

    if callable(test_dataloader):
        logger.info(
            f'Build test data loader from {test_dataloader}', ranks=[0])
        test_dataloader = test_dataloader()
    # testing data, allowed to be None
    if test_dataloader is None and hasattr(gpc.config, 'test_data'):
        test_dataset = build_dataset(gpc.config.test_data.dataset)
        test_dataloader = get_dataloader(
            test_dataset, add_sampler_if_possible=True, **gpc.config.test_data.dataloader)
        logger.info(
            f'Loaded {len(test_dataset)} samples in {len(test_dataloader)} batches for testing', ranks=[0])

    # build loss function
    assert hasattr(gpc.config, 'loss'), \
        'Build error: configuration \'loss\' is missing.'
    criterion = build_loss(gpc.config.loss)
    logger.info('Loss function is created', ranks=[0])

    # build optimizer
    assert hasattr(gpc.config, 'optimizer'), \
        "Build error: configuration 'optimizer' is missing."
    optim_type = gpc.config.optimizer.type
    is_pytorch_native_zero_level_1 = optim_type == 'ZeroRedundancyOptimizer'
    if is_pytorch_native_zero_level_1:
        original_cfg_copy = gpc.config.optimizer.copy()
        original_cfg_copy.pop('type')
        cfg = dict(type=optim_type, process_group=gpc.get_group(
            ParallelMode.DATA), **original_cfg_copy)
        optimizer = build_optimizer(cfg, model)
    else:
        optimizer = build_optimizer(gpc.config.optimizer, model)

    if should_wrap_zero_optimizer_level_2_3:
        optimizer = build_optimizer_wrapper(gpc.config.zero, optimizer, model)

    if should_wrap_fp16_optimizer:
        # replace the field mode with type
        fp16_cfg = gpc.config.fp16.copy()
        amp_type = fp16_cfg.pop('mode')
        assert amp_type == AMP_TYPE.PARALLEL, 'FP Optimizer should only be used for AMP_TYPE.PARALLEL'
        fp16_cfg['type'] = 'FP16Optimizer'
        optimizer = build_optimizer_wrapper(fp16_cfg, optimizer)
    logger.info('Optimizer is created', ranks=[0])

    lr_scheduler = None
    if hasattr(gpc.config, 'lr_scheduler'):
        if hasattr(gpc.config, 'num_steps'):
            total_steps = gpc.config.num_steps
        elif hasattr(gpc.config, 'num_epochs'):
            total_steps = int(gpc.config.num_epochs * len(train_dataloader))
        else:
            raise Exception(
                'Please specify training stopping criterion num_steps or num_epochs in your configuration.'
            )
        lr_scheduler = build_lr_scheduler(gpc.config.lr_scheduler, optimizer,
                                          total_steps, len(train_dataloader))
        logger.info('Learning rate scheduler is created', ranks=[0])

    # pipeline or no pipeline schedule
    if hasattr(gpc.config, 'fp16'):
        amp_type = gpc.config.fp16.mode
        amp_cfg = gpc.config.fp16.copy()
        amp_cfg.pop('mode')
    else:
        amp_type = None
        amp_cfg = None

    if gpc.is_initialized(ParallelMode.PIPELINE) and gpc.get_world_size(ParallelMode.PIPELINE) > 1:
        assert hasattr(gpc.config,
                       'schedule'), "Config 'schedule' not found in your configuration file for pipeline parallel training"
        schedule = PipelineSchedule(
            amp_type=amp_type, amp_config=amp_cfg, **gpc.config.schedule.copy())
    else:
        schedule = NoPipelineSchedule(amp_type=amp_type, amp_config=amp_cfg)

    return model, train_dataloader, test_dataloader, criterion, optimizer, schedule, lr_scheduler
