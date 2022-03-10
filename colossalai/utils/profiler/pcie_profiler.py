from pathlib import Path
from torch.autograd.profiler import profile
from .prof_utils import BaseProfiler, _format_time, _format_memory, _format_bandwith
from typing import List


def _get_size(dtype: str):
    if dtype == "fp16":
        return 2
    elif dtype == "fp32":
        return 4
    else:
        raise NotImplementedError


def _get_numel(my_list: List[int]) -> int:
    from functools import reduce
    from operator import mul
    return reduce(mul, my_list)


def _reduce_location(locations: List[str]) -> str:
    ret = []
    for lo in locations:
        ret.append(lo)
        ret.append("\n")
    return ''.join(ret)


class PcieEvent(object):
    """Pcie Event.
    """

    def __init__(self, count: int = 0, pcie_vol: int = 0, cuda_time: int = 0):
        self.count = count
        self.pcie_vol = pcie_vol
        self.cuda_time = cuda_time

    def add(self, rhs):
        self.count += rhs.count
        self.pcie_vol += rhs.pcie_vol
        self.cuda_time += rhs.cuda_time


class PcieProfiler(BaseProfiler):
    """Pcie profiler. Records all data transmission between CPU and GPU.

    TODO: Merge pcie profiler into communication profiler
    """

    def __init__(self,
                 dtype: str = "fp32",
                 depth: int = 1,
                 total_count: int = 0,
                 total_pcie_vol: int = 0,
                 total_cuda_time: int = 0):
        super().__init__(profiler_name="Pcie", priority=10)
        self.depth = depth
        self.data_size = _get_size(dtype)
        self.total_count = total_count
        self.total_pcie_vol = total_pcie_vol
        self.total_cuda_time = total_cuda_time

        self.ops_record = dict()
        self.profiler = None

    def enable(self):
        self.profiler = profile(enabled=True,
                                use_cuda=True,
                                use_cpu=True,
                                use_kineto=True,
                                record_shapes=True,
                                with_stack=True)
        self.profiler.__enter__()

    def disable(self):
        self.profiler.__exit__(None, None, None)

        if self.profiler.enabled:
            events = self.profiler.function_events
            for event in events:
                if event.name == "aten::_to_copy":
                    t_shape = event.input_shapes[0]
                    if len(t_shape) == 0 or event.cuda_time_total == 0:
                        continue
                    current_comm_event = PcieEvent(1, self.data_size * _get_numel(t_shape), event.cuda_time_total)
                    self.total_count += current_comm_event.count
                    self.total_pcie_vol += current_comm_event.pcie_vol
                    self.total_cuda_time += current_comm_event.cuda_time
                    code_location = _reduce_location(event.stack[:self.depth])
                    if code_location in self.ops_record:
                        self.ops_record[code_location].add(current_comm_event)
                    else:
                        self.ops_record[code_location] = current_comm_event

        self.profiler = None

    def to_tensorboard(self, writer):
        writer.add_text(tag="Data Transmission", text_string=self.result_list("\n\n"))

    def to_file(self, filename: Path):
        with open(filename, "w") as f:
            f.write(self.result_list())

    def show(self):
        print(self.result_list())

    def result_list(self, sep: str = "\n"):
        res = []

        def append(s: str):
            res.append(s)
            res.append(sep)

        append("Pcie profiling result:")
        append("total cuda time: {}".format(_format_time(self.total_cuda_time)))
        append("average bandwith: {}".format(_format_bandwith(self.total_pcie_vol, self.total_cuda_time)))
        append("total number of calls: {}".format(self.total_count))
        append("All events:\n----------------------------------------")

        show_list = sorted(self.ops_record.items(), key=lambda kv: -kv[1].cuda_time)
        for location, event in show_list:
            append(location)
            append("cuda time: {}".format(_format_time(event.cuda_time)))
            append("{:.1f}% of total pcie time".format(event.cuda_time / self.total_cuda_time * 100.0))
            append("pcie volme: {}".format(_format_memory(event.pcie_vol)))
            append("average bandwith: {}".format(_format_bandwith(event.pcie_vol, event.cuda_time)))
            append("number of calls: {}".format(event.count))
            append("----------------------------------------")

        return ''.join(res)
