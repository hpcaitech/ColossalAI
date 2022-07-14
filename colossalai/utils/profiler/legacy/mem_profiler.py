from pathlib import Path
from typing import Union
from colossalai.engine import Engine
from torch.utils.tensorboard import SummaryWriter
from colossalai.gemini.ophooks import MemTracerOpHook
from colossalai.utils.profiler.legacy.prof_utils import BaseProfiler


class MemProfiler(BaseProfiler):
    """Wraper of MemOpHook, used to show GPU memory usage through each iteration

    To use this profiler, you need to pass an `engine` instance. And the usage is same like
    CommProfiler.

    Usage::

        mm_prof = MemProfiler(engine)
        with ProfilerContext([mm_prof]) as prof:
            writer = SummaryWriter("mem")
            engine.train()
            ...
            prof.to_file("./log")
            prof.to_tensorboard(writer)

    """

    def __init__(self, engine: Engine, warmup: int = 50, refreshrate: int = 10) -> None:
        super().__init__(profiler_name="MemoryProfiler", priority=0)
        self._mem_tracer = MemTracerOpHook(warmup=warmup, refreshrate=refreshrate)
        self._engine = engine

    def enable(self) -> None:
        self._engine.add_hook(self._mem_tracer)

    def disable(self) -> None:
        self._engine.remove_hook(self._mem_tracer)

    def to_tensorboard(self, writer: SummaryWriter) -> None:
        stats = self._mem_tracer.async_mem_monitor.state_dict['mem_stats']
        for info, i in enumerate(stats):
            writer.add_scalar("memory_usage/GPU", info, i)

    def to_file(self, data_file: Path) -> None:
        self._mem_tracer.save_results(data_file)

    def show(self) -> None:
        stats = self._mem_tracer.async_mem_monitor.state_dict['mem_stats']
        print(stats)
