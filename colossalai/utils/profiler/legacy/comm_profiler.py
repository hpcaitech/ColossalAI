import inspect
from pathlib import Path
from functools import partial
import torch
from torch.autograd.profiler import profile
import torch.distributed as dist
from torch.distributed import ReduceOp
from colossalai.utils import get_current_device
from .prof_utils import BaseProfiler, _format_time, _format_memory, _format_bandwidth
from typing import List, Optional


def _get_code_location(depth: int):
    ret = []
    length = min(len(inspect.stack()), depth + 1)
    for i in range(3, length):
        upper_frame = inspect.stack()[i]
        function_name = inspect.stack()[i - 1].function
        ret.append(upper_frame.filename)
        ret.append('(')
        ret.append(str(upper_frame.lineno))
        ret.append('): ')
        ret.append(function_name)
        if i != length - 1:
            ret.append('\n')

    return ''.join(ret)


torch_all_reduce = dist.all_reduce
torch_all_gather = dist.all_gather
torch_reduce_scatter = dist.reduce_scatter
torch_broadcast = dist.broadcast
torch_reduce = dist.reduce


class CommEvent(object):
    """Communication Event. Used for communication time and communication
    volume recording.
    """

    def __init__(self, count: int = 0, comm_vol: float = 0., cuda_time: int = 0):
        self.self_count = count
        self.self_comm_vol = comm_vol
        self.self_cuda_time = cuda_time

    def add(self, rhs):
        self.self_count += rhs.self_count
        self.self_comm_vol += rhs.self_comm_vol
        self.self_cuda_time += rhs.self_cuda_time


class CommProfiler(BaseProfiler):
    """Communication profiler. Records all communication events.
    """

    def __init__(self, depth: int = 0, total_count: int = 0, total_comm_vol: float = 0, total_cuda_time: int = 0):
        super().__init__(profiler_name="Collective_Communication", priority=0)
        self.depth = 3 + depth
        self.total_count = total_count
        self.total_comm_vol = total_comm_vol
        self.total_cuda_time = total_cuda_time

        self.ops_record = dict()
        self.profiler = None
        self.pending_op = None
        self.pending_metadata = None
        self.warn_flag = False

    def reset(self):
        self.total_count = 0
        self.total_comm_vol = 0
        self.total_cuda_time = 0

        self.ops_record = dict()
        self.profiler = None
        self.pending_op = None
        self.pending_metadata = None
        self.warn_flag = False

    def enable(self):
        dist.all_reduce = partial(all_reduce, profiler=self)
        dist.all_gather = partial(all_gather, profiler=self)
        dist.reduce_scatter = partial(reduce_scatter, profiler=self)
        dist.broadcast = partial(broadcast, profiler=self)
        dist.reduce = partial(reduce, profiler=self)

    def disable(self):
        dist.all_reduce = torch_all_reduce
        dist.all_gather = torch_all_gather
        dist.reduce_scatter = torch_reduce_scatter
        dist.broadcast = torch_broadcast
        dist.reduce = torch_reduce

    def to_tensorboard(self, writer):
        writer.add_text(tag="Collective Communication", text_string=self.result_str("\n\n"))

    def to_file(self, filename: Path):
        with open(filename, "w") as f:
            f.write(self.result_str())

    def show(self):
        print(self.result_str())

    def result_str(self, sep: str = "\n"):
        res = []

        def append(s: str = None):
            if s is not None:
                res.append(s)
            res.append(sep)

        if self.warn_flag:
            append("Warnning: there exists multiple communication operations in the same time. As a result, "
                   "the profiling result is not accurate.")

        if self.total_cuda_time == 0:
            return "No collective communication has been called yet!"

        append("Collective communication profiling result:")
        append("total cuda time: {}".format(_format_time(self.total_cuda_time)))
        append("average bandwidth: {}".format(_format_bandwidth(self.total_comm_vol, self.total_cuda_time)))
        append("total number of calls: {}".format(self.total_count))
        append("All events:")

        seperation = '-' * 74
        row_format = '{:^10}' + '{:^12}' * 2 + '{:^16}' + '{:^12}' * 2

        append(seperation)
        append(row_format.format('Location', 'GPU time', 'Percentage', 'Comm volume', 'Bandwidth', 'Num of calls'))
        append(seperation)

        show_list = sorted(self.ops_record.items(), key=lambda kv: -kv[1].self_cuda_time)
        for location, event in show_list:
            append(location)
            append(
                row_format.format('', _format_time(event.self_cuda_time),
                                  '{:.1f}%'.format(event.self_cuda_time / self.total_cuda_time * 100.0),
                                  _format_memory(event.self_comm_vol),
                                  _format_bandwidth(event.self_comm_vol, event.self_cuda_time), event.self_count))
            append()

        return ''.join(res)

    @property
    def has_aync_op(self):
        return self.pending_op is not None

    def activate_profiler(self, kn: str, vol: float):
        self.pending_metadata = (kn, _get_code_location(self.depth), vol)
        self.profiler = profile(enabled=True, use_cuda=True, use_cpu=True, use_kineto=True)
        self.profiler.__enter__()

    def close_profiler(self, group=None):
        assert self.profiler is not None, "There is no running dist op"
        kernel_name, code_location, vol = self.pending_metadata
        self.profiler.__exit__(None, None, None)

        if self.profiler.enabled and dist.get_world_size(group) > 1:
            assert_flag = 0
            current_comm_event = None
            events = self.profiler.function_events
            for event in events:
                if kernel_name in event.name:
                    assert assert_flag == 0, "Multiple dist ops has been called "
                    current_comm_event = CommEvent(1, vol, event.self_cuda_time_total)
                    assert_flag += 1

            assert current_comm_event is not None, "dist op has not been found"

            buffer = torch.tensor([current_comm_event.self_cuda_time], device=get_current_device())
            torch_all_reduce(buffer, op=ReduceOp.MIN, group=group)
            current_comm_event.self_cuda_time = buffer.item()

            self.total_count += current_comm_event.self_count
            self.total_comm_vol += current_comm_event.self_comm_vol
            self.total_cuda_time += current_comm_event.self_cuda_time
            if code_location in self.ops_record:
                self.ops_record[code_location].add(current_comm_event)
            else:
                self.ops_record[code_location] = current_comm_event

        self.profiler = None
        self.pending_op = None
        self.pending_metadata = None

    def wait_async_op(self):
        if self.pending_op is not None:
            op = self.pending_op
            op.wait()
            self.close_profiler()


class CommHandler(object):
    """Communication handler. A dummy handler to wait aync operations.
    """

    def __init__(self, profiler: CommProfiler):
        super().__init__()
        self.prof = profiler

    def wait(self):
        self.prof.wait_async_op()


def async_check(profiler: CommProfiler):
    if profiler.pending_op is not None:
        profiler.warn_flag = True
        profiler.wait_async_op()


def all_reduce(tensor: torch.Tensor,
               op: ReduceOp = ReduceOp.SUM,
               group=None,
               async_op: bool = False,
               profiler: CommProfiler = None) -> Optional[CommHandler]:
    async_check(profiler)

    comm_size = dist.get_world_size(group)
    correction = 2 * (comm_size - 1) / comm_size
    comm_vol = correction * tensor.element_size() * tensor.numel()
    profiler.activate_profiler("ncclKernel_AllReduce_", comm_vol)
    profiler.pending_op = torch_all_reduce(tensor, op, group, async_op)

    if async_op:
        return CommHandler(profiler)

    profiler.close_profiler(group)


def reduce_scatter(output: torch.Tensor,
                   input_list: List[torch.Tensor],
                   op: ReduceOp = ReduceOp.SUM,
                   group=None,
                   async_op: bool = False,
                   profiler: CommProfiler = None) -> Optional[CommHandler]:
    async_check(profiler)

    comm_size = dist.get_world_size(group)
    correction = (comm_size - 1) / comm_size
    comm_vol = 0
    for tensor in input_list:
        comm_vol += tensor.element_size() * tensor.numel()
    comm_vol *= correction
    profiler.activate_profiler("ncclKernel_ReduceScatter_", comm_vol)
    profiler.pending_op = torch_reduce_scatter(output, input_list, op, group, async_op)

    if async_op:
        return CommHandler(profiler)

    profiler.close_profiler(group)


def all_gather(tensor_list: List[torch.Tensor],
               tensor: torch.Tensor,
               group=None,
               async_op: bool = False,
               profiler: CommProfiler = None) -> Optional[CommHandler]:
    async_check(profiler)

    comm_size = dist.get_world_size(group)
    correction = (comm_size - 1) / comm_size
    comm_vol = 0
    for ten in tensor_list:
        comm_vol += ten.element_size() * ten.numel()
    comm_vol *= correction
    profiler.activate_profiler("ncclKernel_AllGather_", comm_vol)
    profiler.pending_op = torch_all_gather(tensor_list, tensor, group, async_op)

    if async_op:
        return CommHandler(profiler)

    profiler.close_profiler(group)


def broadcast(tensor: torch.Tensor,
              src: int,
              group=None,
              async_op: bool = False,
              profiler: CommProfiler = None) -> Optional[CommHandler]:
    async_check(profiler)

    comm_vol = 1.0 * tensor.element_size() * tensor.numel()
    profiler.activate_profiler("ncclKernel_Broadcast_", comm_vol)
    profiler.pending_op = torch_broadcast(tensor, src, group, async_op)

    if async_op:
        return CommHandler(profiler)

    profiler.close_profiler(group)


def reduce(tensor: torch.Tensor,
           dst: int,
           op: ReduceOp = ReduceOp.SUM,
           group=None,
           async_op: bool = False,
           profiler: CommProfiler = None) -> Optional[CommHandler]:
    async_check(profiler)

    comm_vol = 1.0 * tensor.element_size() * tensor.numel()
    profiler.activate_profiler("ncclKernel_Reduce_", comm_vol)
    profiler.pending_op = torch_reduce(tensor, dst, op, group, async_op)

    if async_op:
        return CommHandler(profiler)

    profiler.close_profiler(group)
