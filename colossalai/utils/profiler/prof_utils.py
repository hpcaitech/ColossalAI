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

    ```python
        world_size = 4
        inputs = torch.randn(10, 10, dtype=torch.float32, device=get_current_device())
        outputs = torch.empty(world_size, 10, 10, dtype=torch.float32, device=get_current_device())
        outputs_list = list(torch.chunk(outputs, chunks=world_size, dim=0))

        cc_prof = CommProfiler()

        with ProfilerContext([cc_prof]) as prof:
            op = dist.all_reduce(inputs, async_op=True)
            dist.all_gather(outputs_list, inputs)
            op.wait()
            dist.reduce_scatter(inputs, outputs_list)
            dist.broadcast(inputs, 0)
            dist.reduce(inputs, 0)

        prof.show()
    ```
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
