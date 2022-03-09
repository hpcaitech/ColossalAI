from pathlib import Path
from typing import Union
from colossalai.engine import Engine
from torch.utils.tensorboard import SummaryWriter
from colossalai.engine.ophooks import MemTracerOpHook


class MemProfiler(object):
    """Wraper of MemOpHook, used to show GPU memory usage through each iteration

    """

    def __init__(self, engine: Engine, warmup: int = 50, refreshrate: int = 10) -> None:
        super().__init__(profiler_name="Memory Profiler", priority=0)
        self._mem_tracer = MemTracerOpHook(warmup=warmup, refreshrate=refreshrate)
        self._engine = engine

    def enable(self) -> None:
        self._engine.add_hook(self._mem_tracer)

    def disable(self) -> None:
        self._engine.remove_hook(self._mem_tracer)

    def to_tensorboard(self, writer: SummaryWriter) -> None:
        pass

    def to_file(self, log_dir: Union[str, Path]) -> None:
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
        pass

    def show(self) -> None:
        pass

    def get_latest(self) -> float:
        pass

    def get_avg(self) -> float:
        pass
