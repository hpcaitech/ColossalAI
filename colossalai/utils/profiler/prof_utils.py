from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List
from colossalai.core import global_context as gpc


class BaseProfiler(ABC):

    def __init__(self, profiler_name: str, priority: int):
        self.name = profiler_name
        self.priority = priority

    @abstractmethod
    def enable(self):
        pass

    @abstractmethod
    def disable(self):
        pass

    @abstractmethod
    def to_tensorboard(self, writer):
        pass

    @abstractmethod
    def to_file(self, filename: Path):
        pass

    @abstractmethod
    def show(self):
        pass


class ProfilerContext(object):
    """
    Profiler context manager
    Usage:
    from colossalai.utils.profiler import CommProf, ProfilerContext
    from torch.utils.tensorboard import SummaryWriter
    cc_prof = CommProf()
    with ProfilerContext([cc_prof]) as prof:
        train()
    writer = SummaryWriter('tb/path')
    prof.to_tensorboard(writer)
    prof.to_file('./prof_logs/')
    prof.show()
    """

    def __init__(self, profilers: List[BaseProfiler] = None, enable: bool = True):
        self.enable = enable
        self.profilers = sorted(profilers, key=lambda prof: prof.priority)

    def __enter__(self):
        if self.enable:
            for prof in self.profilers:
                prof.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable:
            for prof in self.profilers:
                prof.disable()

    def to_tensorboard(self, writer):
        from torch.utils.tensorboard import SummaryWriter

        assert isinstance(writer, SummaryWriter), \
            f'torch.utils.tensorboard.SummaryWriter is required, but found {type(writer)}.'

        for prof in self.profilers:
            prof.to_tensorboard(writer)

    def to_file(self, log_dir: Union[str, Path]):
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)

        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
        for prof in self.profilers:
            log_file = log_dir.joinpath(f'{prof.name}_rank_{gpc.get_global_rank()}.log')
            prof.to_file(log_file)

    def show(self):
        for prof in self.profilers:
            prof.show()
