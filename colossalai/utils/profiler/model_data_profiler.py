import os
import threading
import time
from enum import Enum
from typing import List
from colossalai.gemini.memory_tracer import GLOBAL_MODEL_DATA_TRACER
from colossalai.engine.ophooks import BaseOpHook
import torch
from colossalai.engine import Engine


class DeviceType(Enum):
    CPU = 0
    CUDA = 1


def get_timestamp_us():
    return int(time.time() * 1e6)


def generic_instant_event(name, pid, tid, timestamp, args):
    return {'ph': 'i', 's': 't', 'name': name, 'pid': pid, 'tid': tid, 'ts': timestamp, 'args': args}


class ModelDataEvent:
    EVENT_NAME = '[modelData]'

    def __init__(self, timestamp: int, device_type: DeviceType, bytes: int) -> None:
        self.pid = os.getpid()
        self.tid = threading.get_ident()
        self.timestamp = timestamp
        self.device_type = device_type
        self.device_id = torch.cuda.current_device() if device_type == DeviceType.CUDA else -1
        self.bytes = bytes

    def state_dict(self):
        return generic_instant_event(ModelDataEvent.EVENT_NAME, self.pid, self.tid, self.timestamp, {
            'Device Type': self.device_type.value,
            'Device Id': self.device_id,
            'Bytes': self.bytes
        })


class ModelDataTracer:

    def __init__(self) -> None:
        self.events: List[ModelDataEvent] = []
        self._tracing = False

    def sample(self):
        cuda_model_data, cpu_model_data = GLOBAL_MODEL_DATA_TRACER.both_mem_usage
        timestamp = get_timestamp_us()
        if self._tracing:
            self.events.append(ModelDataEvent(timestamp, DeviceType.CUDA, cuda_model_data))
            self.events.append(ModelDataEvent(timestamp, DeviceType.CPU, cpu_model_data))

    def start_trace(self):
        self._tracing = True

    def stop_trace(self):
        self._tracing = False

    def state_dict(self):
        return [event.state_dict() for event in self.events]


class ModelDataTracerHook(BaseOpHook):

    def __init__(self, tracer: ModelDataTracer):
        super().__init__()
        self.tracer = tracer

    def pre_fwd_exec(self, module: torch.nn.Module, *args):
        self.tracer.sample()

    def post_fwd_exec(self, module: torch.nn.Module, *args):
        self.tracer.sample()

    def pre_bwd_exec(self, module: torch.nn.Module, input, output):
        self.tracer.sample()

    def post_bwd_exec(self, module: torch.nn.Module, input):
        self.tracer.sample()

    def post_iter(self):
        self.tracer.sample()


class ModelDataProfiler:

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.tracer = ModelDataTracer()
        self.hook = ModelDataTracerHook(self.tracer)
        self.hook_registered = False

    def prepare_trace(self):
        if not self.hook_registered:
            self.engine.add_hook(self.hook)
            self.hook_registered = True

    def start_trace(self):
        self.prepare_trace()
        self.tracer.start_trace()

    def stop_trace(self):
        self.tracer.stop_trace()
        if self.hook_registered:
            self.engine.remove_hook(self.hook)
            self.hook_registered = False
