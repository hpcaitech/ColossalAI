import inspect
import torch
from torch.autograd.profiler import profile
import torch.distributed as dist
from torch.distributed import ReduceOp
from colossalai.utils import get_current_device
from typing import List, Optional


def _get_code_location(depth: int):
    ret = ""
    length = len(inspect.stack())
    for i in range(3, min(length, depth + 1)):
        upper_frame = inspect.stack()[i]
        function_name = inspect.stack()[i - 1].function
        info = upper_frame.filename + "(" + str(upper_frame.lineno) + "): " + function_name + "\n"
        ret += info

    return ret


# copied from high version pytorch to support low version
def _format_time(time_us):
    """Defines how to format time in FunctionEvent"""
    US_IN_SECOND = 1000.0 * 1000.0
    US_IN_MS = 1000.0
    if time_us >= US_IN_SECOND:
        return '{:.3f}s'.format(time_us / US_IN_SECOND)
    if time_us >= US_IN_MS:
        return '{:.3f}ms'.format(time_us / US_IN_MS)
    return '{:.3f}us'.format(time_us)


# copied from high version pytorch to support low version
def _format_memory(nbytes):
    """Returns a formatted memory size string"""
    KB = 1024
    MB = 1024 * KB
    GB = 1024 * MB
    if (abs(nbytes) >= GB):
        return '{:.2f} GB'.format(nbytes * 1.0 / GB)
    elif (abs(nbytes) >= MB):
        return '{:.2f} MB'.format(nbytes * 1.0 / MB)
    elif (abs(nbytes) >= KB):
        return '{:.2f} KB'.format(nbytes * 1.0 / KB)
    else:
        return str(nbytes) + ' b'


def _format_bandwith(volme: float, time_us: int):
    sec_div_mb = (1000.0 / 1024.0)**2
    mb_per_sec = volme / time_us * sec_div_mb

    if mb_per_sec >= 1024.0:
        return '{:.3f} GB/s'.format(mb_per_sec / 1024.0)
    else:
        return '{:.3f} MB/s'.format(mb_per_sec)


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


class CommProfiler(object):
    """Communication profiler. Records all communication events.
    """

    def __init__(self, total_count: int = 0, total_comm_vol: float = 0, total_cuda_time: int = 0, prof_depth: int = 3):
        super().__init__()
        self.total_count = total_count
        self.total_comm_vol = total_comm_vol
        self.total_cuda_time = total_cuda_time
        self.depth = prof_depth

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

    def show(self):
        if self.warn_flag:
            print("Warnning: there exists multiple communication operations in the same time.\n"
                  "As a result, the profiling result is not accurate.")
        print("Collective communication profiling result:",
              "total cuda time: {}".format(_format_time(self.total_cuda_time)),
              "average bandwith: {}".format(_format_bandwith(self.total_comm_vol, self.total_cuda_time)),
              "total number of calls: {}".format(self.total_count),
              "All events:",
              sep='\n')

        show_list = sorted(self.ops_record.items(), key=lambda kv: -kv[1].self_cuda_time)
        for location, event in show_list:
            print(location,
                  "self cuda time: {}".format(_format_time(event.self_cuda_time)),
                  "{:.1f}% of total communication time".format(event.self_cuda_time / self.total_cuda_time * 100.0),
                  "self communication volme: {}".format(_format_memory(event.self_comm_vol)),
                  "average bandwith: {}".format(_format_bandwith(event.self_comm_vol, event.self_cuda_time)),
                  "number of calls: {}".format(event.self_count),
                  "--------------------",
                  sep='\n')

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

        if self.profiler.enabled:
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

    def __init__(self):
        super().__init__()
        self.prof = COL_COMM_PROF

    def wait(self):
        self.prof.wait_async_op()


COL_COMM_PROF = CommProfiler()
torch_all_reduce = dist.all_reduce
torch_all_gather = dist.all_gather
torch_reduce_scatter = dist.reduce_scatter
torch_broadcast = dist.broadcast
torch_reduce = dist.reduce


def enable_communication_prof(depth: int = 0):
    COL_COMM_PROF.depth = 3 + depth
    dist.all_reduce = all_reduce
    dist.all_gather = all_gather
    dist.reduce_scatter = reduce_scatter
    dist.broadcast = broadcast
    dist.reduce = reduce


def communication_prof_show():
    COL_COMM_PROF.show()


def async_check():
    if COL_COMM_PROF.pending_op is not None:
        COL_COMM_PROF.warn_flag = True
        COL_COMM_PROF.wait_async_op()


def all_reduce(tensor: torch.Tensor,
               op: ReduceOp = ReduceOp.SUM,
               group=None,
               async_op: bool = False) -> Optional[CommHandler]:
    async_check()

    comm_size = dist.get_world_size(group)
    correction = 2 * (comm_size - 1) / comm_size
    comm_vol = correction * tensor.element_size() * tensor.numel()
    COL_COMM_PROF.activate_profiler("ncclKernel_AllReduce_", comm_vol)
    COL_COMM_PROF.pending_op = torch_all_reduce(tensor, op, group, async_op)

    if async_op:
        return CommHandler()

    COL_COMM_PROF.close_profiler(group)


def reduce_scatter(output: torch.Tensor,
                   input_list: List[torch.Tensor],
                   op: ReduceOp = ReduceOp.SUM,
                   group=None,
                   async_op: bool = False) -> Optional[CommHandler]:
    async_check()

    comm_size = dist.get_world_size(group)
    correction = (comm_size - 1) / comm_size
    comm_vol = 0
    for tensor in input_list:
        comm_vol += tensor.element_size() * tensor.numel()
    comm_vol *= correction
    COL_COMM_PROF.activate_profiler("ncclKernel_ReduceScatter_", comm_vol)
    COL_COMM_PROF.pending_op = torch_reduce_scatter(output, input_list, op, group, async_op)

    if async_op:
        return CommHandler()

    COL_COMM_PROF.close_profiler(group)


def all_gather(tensor_list: List[torch.Tensor],
               tensor: torch.Tensor,
               group=None,
               async_op: bool = False) -> Optional[CommHandler]:
    async_check()

    comm_size = dist.get_world_size(group)
    correction = (comm_size - 1) / comm_size
    comm_vol = 0
    for ten in tensor_list:
        comm_vol += ten.element_size() * ten.numel()
    comm_vol *= correction
    COL_COMM_PROF.activate_profiler("ncclKernel_AllGather_", comm_vol)
    COL_COMM_PROF.pending_op = torch_all_gather(tensor_list, tensor, group, async_op)

    if async_op:
        return CommHandler()

    COL_COMM_PROF.close_profiler(group)


def broadcast(tensor: torch.Tensor, src: int, group=None, async_op: bool = False) -> Optional[CommHandler]:
    async_check()

    comm_vol = 1.0 * tensor.element_size() * tensor.numel()
    COL_COMM_PROF.activate_profiler("ncclKernel_Broadcast_", comm_vol)
    COL_COMM_PROF.pending_op = torch_broadcast(tensor, src, group, async_op)

    if async_op:
        return CommHandler()

    COL_COMM_PROF.close_profiler(group)


def reduce(tensor: torch.Tensor,
           dst: int,
           op: ReduceOp = ReduceOp.SUM,
           group=None,
           async_op: bool = False) -> Optional[CommHandler]:
    async_check()

    comm_vol = 1.0 * tensor.element_size() * tensor.numel()
    COL_COMM_PROF.activate_profiler("ncclKernel_Reduce_", comm_vol)
    COL_COMM_PROF.pending_op = torch_reduce(tensor, dst, op, group, async_op)

    if async_op:
        return CommHandler()

    COL_COMM_PROF.close_profiler(group)
