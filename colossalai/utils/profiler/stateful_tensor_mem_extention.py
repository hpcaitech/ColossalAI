import os
import threading
import time
import torch
from enum import Enum
from typing import List
from colossalai.gemini.stateful_tensor import StatefulTensor
from colossalai.gemini.ophooks import BaseOpHook
from colossalai.engine import Engine
from colossalai.utils.profiler.extention import ProfilerExtension


class DeviceType(Enum):
    CPU = 0
    CUDA = 1


def get_timestamp_us():
    return int(time.time() * 1e6)


def generic_instant_event(name, pid, tid, timestamp, args):
    return {'ph': 'i', 's': 't', 'name': name, 'pid': pid, 'tid': tid, 'ts': timestamp, 'args': args}


class StatefulTensorMemoryEvent:
    EVENT_NAME = '[statefulTensorMemory]'

    def __init__(self, timestamp: int, device_type: DeviceType, bytes_: int) -> None:
        self.pid = os.getpid()
        self.tid = threading.get_ident()
        self.timestamp = timestamp
        self.device_type = device_type
        self.device_id = torch.cuda.current_device() if device_type == DeviceType.CUDA else -1
        self.bytes = bytes_

    def state_dict(self):
        return generic_instant_event(StatefulTensorMemoryEvent.EVENT_NAME, self.pid, self.tid, self.timestamp, {
            'Device Type': self.device_type.value,
            'Device Id': self.device_id,
            'Bytes': self.bytes
        })


class StatefulTensorMemoryTracer:

    def __init__(self) -> None:
        self.events: List[StatefulTensorMemoryEvent] = []
        self._tracing = False

    def sample(self):
        cuda_mem = StatefulTensor.GST_MGR.total_mem['cuda']
        cpu_mem = StatefulTensor.GST_MGR.total_mem['cpu']
        timestamp = get_timestamp_us()
        if self._tracing:
            self.events.append(StatefulTensorMemoryEvent(timestamp, DeviceType.CUDA, cuda_mem))
            self.events.append(StatefulTensorMemoryEvent(timestamp, DeviceType.CPU, cpu_mem))

    def start_trace(self):
        self.events.clear()
        self._tracing = True

    def stop_trace(self):
        self._tracing = False

    def state_dict(self):
        return [event.state_dict() for event in self.events]


class StatefulTensorMemoryTracerHook(BaseOpHook):

    def __init__(self, tracer: StatefulTensorMemoryTracer):
        super().__init__()
        self.tracer = tracer
        self._enable = False

    def pre_fwd_exec(self, module: torch.nn.Module, *args):
        if self._enable:
            self.tracer.sample()

    def post_fwd_exec(self, module: torch.nn.Module, *args):
        if self._enable:
            self.tracer.sample()

    def pre_bwd_exec(self, module: torch.nn.Module, input_, output):
        if self._enable:
            self.tracer.sample()

    def post_bwd_exec(self, module: torch.nn.Module, input_):
        if self._enable:
            self.tracer.sample()

    def post_iter(self):
        if self._enable:
            self.tracer.sample()

    def enable(self):
        self._enable = True

    def disable(self):
        self._enable = False


class StatefulTensorMemoryProfilerExtention(ProfilerExtension):

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.tracer = StatefulTensorMemoryTracer()
        self.hook = StatefulTensorMemoryTracerHook(self.tracer)
        self.hook_registered = False

    def prepare_trace(self):
        self.hook.enable()
        if not self.hook_registered:
            self.engine.add_hook(self.hook)
            self.hook_registered = True

    def start_trace(self):
        self.prepare_trace()
        self.tracer.start_trace()

    def stop_trace(self):
        self.tracer.stop_trace()
        self.hook.disable()
        if self.hook_registered:
            self.engine.remove_hook(self.hook)
            # remove_hook is not implemented now
            # FIXME(ver217): uncomment below line when remove_hook is implemented
            # self.hook_registered = False

    def extend_chrome_trace(self, trace: dict) -> dict:
        trace['traceEvents'].extend(self.tracer.state_dict())
        return trace
