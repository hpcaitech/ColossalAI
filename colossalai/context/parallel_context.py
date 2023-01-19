#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import random
import socket
from collections import Counter
from threading import local
from typing import Union

import numpy as np
import torch
import torch.distributed as dist
from colossalai.constants import ALLOWED_MODES, INITIALIZER_MAPPING
from colossalai.context.config import Config
from colossalai.global_variables import tensor_parallel_env as env
from colossalai.logging import get_dist_logger
from colossalai.registry import DIST_GROUP_INITIALIZER

from .parallel_mode import ParallelMode
from .random import add_seed, get_seeds, set_mode
from colossalai.context.singleton_meta import SingletonMeta


class ParallelContext(metaclass=SingletonMeta):
    """This class provides interface functions for users to get the parallel context,
    such as the global rank, the local rank, the world size, etc. of each device.

    Note:
        The parallel_mode used in this class should be concluded in ``ParallelMode``.
        More details about ``ParallelMode`` could be found in
        `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.
    """

    def __init__(self):
        # distributed settings
        self._global_ranks = dict()
        self._local_ranks = dict()
        self._world_sizes = dict()
        self._groups = dict()
        self._cpu_groups = dict()
        self._ranks_in_group = dict()

        # load config from file
        self._config = None

        # default 3D parallel args, will be overwritten during process group intialization
        self.world_size = 1
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 1
        self.tensor_parallel_size = 1
        self.num_processes_on_current_node = -1
        self.virtual_pipeline_parallel_size = None
        self.virtual_pipeline_parallel_rank = None

        # logging
        self._verbose = False
        self._logger = get_dist_logger()

    @property
    def config(self):
        return self._config

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, verbose_: bool):
        self._verbose = verbose_

    def load_config(self, config: Union[dict, str]):
        """Loads the configuration from either a dict or a file.

        Args:
            config (dict or str): Either a dict containing the configuration information or the filename
                of a file containing the configuration information.

        Raises:
            TypeError: Raises a TypeError if `config` is neither a dict nor a str.
        """
        if isinstance(config, str):
            self._config = Config.from_file(config)
        elif isinstance(config, dict):
            self._config = Config(config)
        else:
            raise TypeError("Invalid type for config, only dictionary or string is supported")

    def detect_num_processes_on_current_node(self):
        hostname = socket.gethostname()
        hostname_list = [None for _ in range(self.get_world_size(ParallelMode.GLOBAL))]
        dist.all_gather_object(hostname_list, hostname, group=self.get_group(ParallelMode.GLOBAL))
        counter = Counter(hostname_list)
        self.num_processes_on_current_node = counter[hostname]

    @staticmethod
    def _check_parallel_mode(parallel_mode: ParallelMode):
        assert isinstance(parallel_mode, ParallelMode), \
            f'expected the argument parallel_mode to be of enum ParallelMode, but got {type(parallel_mode)}'

    def get_global_rank(self):
        """Returns the global rank of the current device.

        Returns:
            int: The global rank of the current device
        """
        return self._global_ranks[ParallelMode.GLOBAL]

    def add_global_rank(self, parallel_mode: ParallelMode, rank: int):
        """Adds the global rank of the current device for `parallel_mode` to the context.

        Args:
            parallel_mode (:class:`colossalai.context.ParallelMode`): The parallel mode for the rank.
            rank (int): The rank to be added

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
                of :class:`colossalai.context.ParallelMode`.
        """
        self._check_parallel_mode(parallel_mode)
        self._global_ranks[parallel_mode] = rank

    def get_local_rank(self, parallel_mode: ParallelMode):
        """Returns the local rank of the current device.

        Args:
            parallel_mode (:class:`colossalai.context.ParallelMode`): The chosen parallel mode.

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
                of :class:`colossalai.context.ParallelMode`.

        Returns:
            int: The local rank of the current device for `parallel_mode`.
        """
        self._check_parallel_mode(parallel_mode)
        return self._local_ranks[parallel_mode]

    def _add_local_rank(self, parallel_mode: ParallelMode, rank: int):
        """Adds the local rank of the current device for `parallel_mode` to the context.

        Args:
            parallel_mode (:class:`colossalai.context.ParallelMode`): The parallel mode for the rank.
            rank (int): The rank to be added.

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
                of :class:`colossalai.context.ParallelMode`.
        """
        self._check_parallel_mode(parallel_mode)
        self._local_ranks[parallel_mode] = rank

    def get_next_global_rank(self, parallel_mode: ParallelMode):
        """Returns the global rank of the next device.

        Args:
            parallel_mode (:class:`colossalai.context.ParallelMode`): The chosen parallel mode.

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
                of :class:`colossalai.context.ParallelMode`.

        Returns:
            int: The global rank of the next device for `parallel_mode`.
        """
        self._check_parallel_mode(parallel_mode)

        # get rank and world size
        local_rank = self.get_local_rank(parallel_mode)
        world_size = self.get_world_size(parallel_mode)
        ranks_in_group = self.get_ranks_in_group(parallel_mode)

        return ranks_in_group[(local_rank + 1) % world_size]

    def get_prev_global_rank(self, parallel_mode: ParallelMode):
        """Returns the global rank of the previous device.

        Args:
            parallel_mode (:class:`colossalai.context.ParallelMode`): The chosen parallel mode.

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
                of :class:`colossalai.context.ParallelMode`.

        Returns:
            int: The global rank of the previous device for `parallel_mode`.
        """
        self._check_parallel_mode(parallel_mode)

        # get rank and world size
        local_rank = self.get_local_rank(parallel_mode)
        world_size = self.get_world_size(parallel_mode)
        ranks_in_group = self.get_ranks_in_group(parallel_mode)

        return ranks_in_group[(local_rank - 1) % world_size]

    def is_first_rank(self, parallel_mode: ParallelMode):
        """Returns a boolean value indicating whether the current device is the first one
        among its group for `parallel_mode`.

        Args:
            parallel_mode (:class:`colossalai.context.ParallelMode`): The chosen parallel mode.

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
                of :class:`colossalai.context.ParallelMode`.

        Returns:
            bool: a boolean value indicating whether the current device is the first one
            among its group for `parallel_mode`.
        """
        rank = self.get_local_rank(parallel_mode)
        return rank == 0

    def is_last_rank(self, parallel_mode: ParallelMode):
        """Returns a boolean value indicating whether the current device is the last one
        among its group for `parallel_mode`.

        Args:
            parallel_mode (:class:`colossalai.context.ParallelMode`): The chosen parallel mode.

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
                of :class:`colossalai.context.ParallelMode`.

        Returns:
            bool: a boolean value indicating whether the current device is the first one
            among its group for `parallel_mode`.
        """
        rank = self.get_local_rank(parallel_mode)
        world_size = self.get_world_size(parallel_mode)
        return rank == world_size - 1

    def is_pipeline_first_stage(self, ignore_virtual=False):
        if not ignore_virtual:
            if self.virtual_pipeline_parallel_size is not None and self.virtual_pipeline_parallel_rank != 0:
                return False
        return self.is_first_rank(ParallelMode.PIPELINE)

    def is_pipeline_last_stage(self, ignore_virtual=False):
        if not ignore_virtual:
            if self.virtual_pipeline_parallel_size \
                    is not None and self.virtual_pipeline_parallel_rank != self.virtual_pipeline_parallel_size - 1:
                return False
        return self.is_last_rank(ParallelMode.PIPELINE)

    def get_world_size(self, parallel_mode: ParallelMode):
        """Returns the world size for `parallel_mode`.

        Args:
            parallel_mode (:class:`colossalai.context.ParallelMode`): The chosen parallel mode.

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
                of :class:`colossalai.context.ParallelMode`.

        Returns:
            int: The world size for `parallel_mode`.
        """
        self._check_parallel_mode(parallel_mode)
        return self._world_sizes[parallel_mode]

    def _add_world_size(self, parallel_mode: ParallelMode, world_size: int):
        """Adds world size for `parallel_mode`.

        Args:
            parallel_mode (:class:`colossalai.context.ParallelMode`): The parallel mode correponding to the process group
            world_size (int): The world size to be added

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
                of :class:`colossalai.context.ParallelMode`.
        """
        self._check_parallel_mode(parallel_mode)
        self._world_sizes[parallel_mode] = world_size

    def get_group(self, parallel_mode: ParallelMode):
        """Returns the group of the current device for `parallel_mode`.

        Args:
            parallel_mode (:class:`colossalai.context.ParallelMode`): The chosen parallel mode.

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
                of :class:`colossalai.context.ParallelMode`.

        Returns:
            torch.distributed.ProcessGroup: The group of the current device for `parallel_mode`.
        """
        self._check_parallel_mode(parallel_mode)
        return self._groups[parallel_mode]

    def _add_group(self, parallel_mode: ParallelMode, group: dist.ProcessGroup):
        """Adds the group of the current device for `parallel_mode`.

        Args:
            parallel_mode (:class:`colossalai.context.ParallelMode`): The chosen parallel mode.
            group (torch.distributed.ProcessGroup): The group to be added

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
                of :class:`colossalai.context.ParallelMode`.
        """
        self._check_parallel_mode(parallel_mode)
        self._groups[parallel_mode] = group

    def get_cpu_group(self, parallel_mode: ParallelMode):
        """Returns the Gloo group of the current device for `parallel_mode`.

        :param parallel_mode: The chosen parallel mode
        :type parallel_mode: :class:`colossalai.context.ParallelMode`
        :raises AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
            of :class:`colossalai.context.ParallelMode`
        :return: The group of the current device for `parallel_mode`
        :rtype: torch.distributed.ProcessGroup
        """
        self._check_parallel_mode(parallel_mode)
        return self._cpu_groups[parallel_mode]

    def _add_cpu_group(self, parallel_mode: ParallelMode, group: dist.ProcessGroup):
        """Adds the Gloo group of the current device for `parallel_mode`.

        :param parallel_mode: The chosen parallel mode
        :type parallel_mode: :class:`colossalai.context.ParallelMode`
        :param group: The group to be added
        :type group: torch.distributed.ProcessGroup
        :raises AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
            of :class:`colossalai.context.ParallelMode`
        """
        self._check_parallel_mode(parallel_mode)
        self._cpu_groups[parallel_mode] = group

    def get_ranks_in_group(self, parallel_mode: ParallelMode):
        """Returns the rank of the current device for `parallel_mode` in the group.

        Args:
            parallel_mode (:class:`colossalai.context.ParallelMode`): The chosen parallel mode.

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
                of :class:`colossalai.context.ParallelMode`.

        Returns:
            int: The rank of the current device for `parallel_mode` in the group.
        """
        self._check_parallel_mode(parallel_mode)
        return self._ranks_in_group[parallel_mode]

    def _add_ranks_in_group(self, parallel_mode: ParallelMode, ranks: list):
        """Adds the ranks of the current device for `parallel_mode` in the group.

        Args:
            parallel_mode (:class:`colossalai.context.ParallelMode`): The chosen parallel mode.
            ranks (list): List of ranks to be added

        Raises:
            AssertionError: Raises an AssertionError if `parallel_mode` is not an instance
                of :class:`colossalai.context.ParallelMode`.
        """
        self._check_parallel_mode(parallel_mode)
        self._ranks_in_group[parallel_mode] = ranks

    def init_global_dist(self, rank: int, world_size: int, backend: str, host: str, port: int):
        """Initializes the global distributed environment

        Args:
           rank (int): rank for the default process group.
           world_size (int): world size of the default process group.
           backend (str): backend for ``torch.distributed``
           host (str): the master address for distributed training.
           port (str): the master port for distributed training
        """
        # initialize the default process group
        init_method = f'tcp://[{host}]:{port}'
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend, init_method=init_method)

        # None will give the default global process group for pytorch dist operations
        ranks = list(range(world_size))
        cpu_group = dist.new_group(ranks, backend='gloo') if dist.get_backend() != 'gloo' else None
        self._register_dist(rank, world_size, dist.GroupMember.WORLD, cpu_group, ranks, ParallelMode.GLOBAL)
        self.add_global_rank(ParallelMode.GLOBAL, rank)

    def _register_dist(self, local_rank, world_size, process_group, cpu_group, ranks_in_group, mode):
        self._add_local_rank(mode, local_rank)
        self._add_world_size(mode, world_size)
        self._add_group(mode, process_group)
        self._add_cpu_group(mode, cpu_group)
        self._add_ranks_in_group(mode, ranks_in_group)

    def check_sanity(self):
        """Checks sanity of the parallel context.

        Raises:
            AssertionError: Raises an AssertionError if the world size does not equal to the product
                of data parallel size, pipeline parallel size and tensor parallel size.
        """
        dps = self.data_parallel_size
        pps = self.pipeline_parallel_size
        tps = self.tensor_parallel_size
        ws = self.world_size
        assert ws == dps * pps * \
            tps, f"Expected the world size {ws} to be equal to data" \
                 f" parallel size ({dps}) * pipeline parallel size " \
                 f"({pps}) * tensor parallel size ({tps})"

    def _set_parallel_size_from_config(self, config: dict, key: str, attr_name: str):
        if key in config:
            ele = config[key]
            if isinstance(ele, int):
                setattr(self, attr_name, ele)
            elif isinstance(ele, dict):
                setattr(self, attr_name, ele['size'])
            else:
                raise NotImplementedError(
                    f'{"Parallel configuration does not support this kind of argument, please use int or dict"}')

    def init_parallel_groups(self):
        """Initializes the parallel groups.

        Raises:
            AssertionError: Raises an AssertionError if the field parallel is not present in the config file.
        """

        # get rank and world size
        rank = self.get_global_rank()
        world_size = self.get_world_size(ParallelMode.GLOBAL)
        self.world_size = world_size

        # set parallel size as attributes for global context
        parallel_config = self.config.get('parallel', None)
        if parallel_config is not None:
            self._set_parallel_size_from_config(parallel_config, 'pipeline', 'pipeline_parallel_size')
            self._set_parallel_size_from_config(parallel_config, 'tensor', 'tensor_parallel_size')

        # the user should not set the data parallel size manually
        # instead, it should be calculated based on other parallel config
        self.data_parallel_size = self.world_size // (self.pipeline_parallel_size * self.tensor_parallel_size)

        # get the tensor parallel mode and check
        tensor_parallel_mode = None
        if parallel_config is not None and 'tensor' in \
                parallel_config and 'mode' in parallel_config['tensor']:
            tensor_parallel_mode = parallel_config['tensor']['mode']
        assert tensor_parallel_mode in ALLOWED_MODES, \
            f"mode in the parallel config must be set to one of {ALLOWED_MODES}"
        env.mode = tensor_parallel_mode

        self.check_sanity()

        pg_init = []
        # LSG: init data parallel process group for compatibility with other parallel module such as zero
        pg_init.append(dict(type=INITIALIZER_MAPPING['data']))

        # LSG: init model parallel process group for compatibility with amp and clip grad
        pg_init.append(dict(type=INITIALIZER_MAPPING['model']))

        if self.pipeline_parallel_size > 1:
            pg_init.append(dict(type=INITIALIZER_MAPPING['pipeline']))
        pg_init.append(dict(type=INITIALIZER_MAPPING['tensor']))

        # init specific tensor parallel group
        if tensor_parallel_mode is not None:
            tensor_parallel_cfg = parallel_config['tensor'].copy()

            # remove duplicate parameters
            tensor_parallel_cfg.pop('mode')
            tensor_parallel_cfg.pop('size')

            # add this config to initialize later
            pg_init.append(dict(type=INITIALIZER_MAPPING[tensor_parallel_mode.lower()], **tensor_parallel_cfg))

        # run initialization of different process groups
        for initializer_cfg in pg_init:
            cfg = initializer_cfg.copy()
            initializer_type = cfg.pop('type')
            initializer = DIST_GROUP_INITIALIZER.get_module(initializer_type)(rank, world_size, self.config,
                                                                              self.data_parallel_size,
                                                                              self.pipeline_parallel_size,
                                                                              self.tensor_parallel_size, **cfg)
            parallel_setting = initializer.init_dist_group()
            if isinstance(parallel_setting, list):
                for args in parallel_setting:
                    self._register_dist(*args)
            else:
                self._register_dist(*parallel_setting)

    def is_initialized(self, parallel_mode: ParallelMode):
        """Returns a boolean value indicating whether `parallel_mode` is initialized
        in the current system.

        Args:
            parallel_mode (:class:`colossalai.context.ParallelMode`): The chosen parallel mode.

        Returns:
            bool: a boolean value indicating whether `parallel_mode` is initialized in the current system.
        """
        return parallel_mode in self._groups

    def destroy(self):
        """Destroys the current distributed parallel environment.
        """
        for mode, group in self._groups.items():
            if mode is not ParallelMode.GLOBAL:
                dist.destroy_process_group(group)
        # destroy global process group
        dist.destroy_process_group()
        self._groups.clear()

    def set_device(self, device_ordinal: int = None):
        """Sets distributed processes to be bound to devices.

        Args:
           device_ordinal (int, optional): the device id to be bound to
        """
        global_rank = self.get_global_rank()
        if device_ordinal is None:
            devices_per_node = torch.cuda.device_count()
            device_ordinal = global_rank % devices_per_node

        torch.cuda.set_device(device_ordinal)
        if self._verbose:
            self._logger.info(f'process rank {global_rank} is bound to device {device_ordinal}')

    def set_seed(self, seed: int):
        """Sets seeds for all random libraries.

        Args:
            seed (int): seed for random states
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        global_rank = self.get_global_rank()

        if torch.cuda.is_available():
            # create random seed for different parallel modes
            # data parallel seed are kept the same
            parallel_seed = seed
            add_seed(ParallelMode.DATA, parallel_seed)

            # model parallel seeds are different across ranks
            pipeline_offset = self._local_ranks.get(ParallelMode.PIPELINE, 0)

            # add seed for data parallel and tensor parallel only
            if self.is_initialized(ParallelMode.TENSOR):
                tp_rank = self.get_local_rank(ParallelMode.TENSOR)
                # 100 is only to increase the diff in seeds between pipeline stages
                tp_rank_with_offset = tp_rank + pipeline_offset * 1024
                tp_seed = seed + tp_rank_with_offset
                add_seed(ParallelMode.TENSOR, tp_seed)

            set_mode(ParallelMode.DATA)
            seeds = get_seeds()
            seed_str = ', '.join([f'{k}: {v}' for k, v in seeds.items()])

            if self._verbose:
                self._logger.info(f"initialized seed on rank {global_rank}, "
                                  f"numpy: {seed}, python random: {seed}, {seed_str},"
                                  f"the default parallel seed is {ParallelMode.DATA}.")
        else:
            if self._verbose:
                self._logger.info(
                    f"initialized seed on rank {global_rank}, "
                    f"numpy: {seed}, python random: {seed}, pytorch: {seed}",
                    ranks=[0])
                self._logger.info(
                    'WARNING: CUDA is not available, thus CUDA RNG cannot be used to track CUDA random number states',
                    ranks=[0])

    def set_virtual_pipeline_parallel_size(self, size):
        self.virtual_pipeline_parallel_size = size

    def set_virtual_pipeline_parallel_rank(self, rank):
        self.virtual_pipeline_parallel_rank = rank


global_context = ParallelContext()
