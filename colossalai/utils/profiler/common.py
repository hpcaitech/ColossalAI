from pathlib import Path
from typing import List, Union

from colossalai.core import global_context as gpc

from ._base_profiler import BaseProfiler


class ProfilerContext(object):
    '''
    Profiler context manager

    Usage:
    from colossalai.utils.profiler import CommProfiler, profiler
    from torch.utils.tensorboard import SummaryWriter

    # to profile communication metrics
    with profiler(activities=[CommProfiler()]) as prof:
        train()
    writer = SummaryWriter('tb/path')
    prof.to_tensorboard(writer)
    prof.to_file('./prof_logs/')
    prof.show()
    '''

    def __init__(self, activities: List[BaseProfiler] = None):
        self.activities = list()
        if activities is not None:
            for a in activities:
                self.activities.append(a)

    def __enter__(self):
        for a in self.activities:
            a.enable()

    def __exit__(self):
        for a in self.activities:
            a.disable()

    def to_tensorboard(self, writer):
        from torch.utils.tensorboard import SummaryWriter

        assert isinstance(writer, SummaryWriter), \
            f'torch.utils.tensorboard.SummaryWriter is required, but found {type(writer)}.'

        for a in self.activities:
            a.to_tensorboard(writer)

    def to_file(self, log_dir: Union[str, Path]):
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)

        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
        for a in self.activities:
            name = a.__class__.__name__.lower()
            log_file = log_dir.joinpath(f'{name}_rank_{gpc.get_global_rank()}.log')
            a.to_file(log_file)

    def show(self):
        for a in self.activities:
            a.show()

    def get_last(self):
        res = dict()
        for a in self.activities:
            out = a.get_last()
            if out is not None:
                for k, v in out.items():
                    res[k] = v
        return res

    def get_avg(self):
        res = dict()
        for a in self.activities:
            out = a.get_avg()
            if out is not None:
                for k, v in out.items():
                    res[k] = v
        return res


def profiler(activities: List[BaseProfiler] = None) -> ProfilerContext:
    return ProfilerContext(activities)
