import json
import pickle
from pathlib import Path
from colossalai.context.parallel_mode import ParallelMode
import torch
from colossalai.engine.ophooks import BaseOpHook
from colossalai.registry import OPHOOKS
from colossalai.logging import get_dist_logger
from colossalai.core import global_context as gpc
from typing import Union
from colossalai.utils.memory_tracer import AsyncMemoryMonitor
import os
import math


@OPHOOKS.register_module
class MemTracerOpHook(BaseOpHook):
    """
    Collect GPU memory usage information
    :param warmup: This parameter indicates how many iterations to truncate before profiling, defaults to 50
    :type warmup: int
    :param refreshrate: This parameter decides the frequency of write file, defaults to 10
    :type refreshrate: int
    :param data_prefix: The prefix of the stats data file, defaults to "memstats"
    :type data_prefix: string
    """

    def __init__(self, warmup: int = 50, refreshrate: int = 10, data_prefix: str = "memstats"):
        super().__init__()
        self.async_mem_monitor = AsyncMemoryMonitor()
        self._curiter = 0
        self._logger = get_dist_logger()
        self._count = 0
        self._warmup = warmup
        self._refreshrate = refreshrate
        self._data_prefix = data_prefix
        # in distributed environment
        if gpc.is_initialized(ParallelMode.GLOBAL):
            self._rank = gpc.get_global_rank()
        else:
            self._rank = 0

    def _isvalid(self, module) -> bool:
        assert isinstance(module, torch.nn.Module)
        return module.training

    def _resample(self):
        # calculate the average iteration time
        total_time = (self.async_mem_monitor.time_stamps[-1] - self.async_mem_monitor.time_stamps[0])
        avg_it_time = total_time / self.warmup
        self._logger.debug(f"total time for {self.warmup} iterations is {total_time}s")
        # adjust the sampling power
        power: int = round(-math.log(avg_it_time, 10)) + 1
        self._logger.debug(f"the power is {power}")
        self.async_mem_monitor.set_interval(power)

    @property
    def refreshrate(self) -> int:
        return self._refreshrate

    @property
    def warmup(self) -> int:
        return self._warmup

    @property
    def curiter(self) -> int:
        return self._curiter

    @property
    def valid_iter(self) -> int:
        return self.curiter - self.warmup

    def pre_fwd_exec(self, module: torch.nn.Module, *args):
        if self._isvalid(module):
            self.async_mem_monitor.finish()
            self.async_mem_monitor.start()

    def post_fwd_exec(self, module: torch.nn.Module, *args):
        if self._isvalid(module):
            self.async_mem_monitor.finish()

    def pre_bwd_exec(self, module: torch.nn.Module, input, output):
        if self._isvalid(module):
            self.async_mem_monitor.finish()
            self.async_mem_monitor.start()

    def post_bwd_exec(self, module: torch.nn.Module, input):
        if self._isvalid(module):
            self.async_mem_monitor.finish()

    def pre_iter(self):
        pass

    def post_iter(self):
        self.async_mem_monitor.finish()
        # in the warmup stage
        if self.curiter < self.warmup:
            pass
        # adjust the sampling rate
        elif self.curiter == self.warmup:
            # use adaptive sample rate
            self._resample()
        # record data to log file
        else:
            # every `refreshrate` times, refresh the file
            if self.valid_iter != 0 and self.valid_iter % self.refreshrate == 0:
                # output file info
                self._logger.info(f"dump a memory statistics as pickle to {self._data_prefix}-{self._rank}.pkl")
                home_dir = Path.home()
                with open (home_dir.joinpath(f".cache/colossal/mem-{self._rank}.pkl"), "wb") as f:
                    pickle.dump(self.async_mem_monitor.state_dict, f)
                self._count += 1
                self._logger.debug(f"data file has been refreshed {self._count} times")
        # finish a iteration
        self._curiter += 1

    def save_results(self, data_file: Union[str, Path]):
        with open(data_file, "w") as f:
            f.write(json.dumps(self.async_mem_monitor.state_dict))