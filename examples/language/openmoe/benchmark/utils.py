import dataclasses
import time
from typing import Dict

import torch.distributed as dist
import torch.nn as nn
from colossalai.logging import DistributedLogger


def print_model_numel(logger: DistributedLogger,
                      model: nn.Module) -> None:
    B = 1024**3
    M = 1024**2
    K = 1024
    outputs = "Model param count: "
    model_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if model_param >= B:
        outputs += f'{model_param / B:.2f} B\n'
    elif model_param >= M:
        outputs += f'{model_param / M:.2f} M\n'
    elif model_param >= K:
        outputs += f'{model_param / K:.2f} K\n'
    else:
        outputs += f'{model_param}\n'
    logger.info(outputs, ranks=[0])


@dataclasses.dataclass
class TimingItem():
    last_time: float = 0.0
    total_time: float = 0.0
    count: float = 0

    def __str__(self) -> str:
        return f"average time: {self.total_time/self.count * 1000:.2f} ms"


class SimpleTimer():
    def __init__(self, warmup: int = 10) -> None:
        self.timing_items: Dict[str, TimingItem] = {}
        self.warmup = warmup

    def start(self, name: str):
        if name not in self.timing_items:
            self.timing_items[name] = TimingItem()
        self.timing_items[name].last_time = time.time()

    def stop(self, name: str):
        assert name in self.timing_items
        timing_item = self.timing_items[name]
        timing_item.total_time += time.time() - timing_item.last_time
        timing_item.count += 1
        if timing_item.count > self.warmup:
            timing_item.count = 0
            timing_item.total_time = 0.0

    def __repr__(self) -> str:
        result = "[Timer]:\n"
        for name, timing_item in self.timing_items.items():
            result += f"    {name}: {timing_item}\n"
        return result
