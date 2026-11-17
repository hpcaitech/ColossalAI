"""``Setup``: everything one measurement depends on, as one record. A scenario is a series of setups that differ in
one field (the sequence length, the device count, the plugin, the implementation, the optimizer)."""

from dataclasses import dataclass, fields
from typing import Any

from .model import DEFAULT_IMPLEMENTATION
from .optimizers import DEFAULT_OPTIMIZER, OPTIMIZERS
from .plugins import CONFIGS

WARMUP_STEPS = 5
TIMED_STEPS = 20


@dataclass(frozen=True)
class Setup:
    """What one measurement measures. Every field is recorded in the report."""

    entry: str
    size: str
    config: str
    seq_len: int
    micro_batch: int
    devices: int = 1
    implementation: str = DEFAULT_IMPLEMENTATION
    optimizer: str = DEFAULT_OPTIMIZER
    gradient_checkpointing: bool = False
    warmup_steps: int = WARMUP_STEPS
    timed_steps: int = TIMED_STEPS
    lr: float = 1e-5
    seed: int = 1024
    # steps run under the profiler after the timed ones, for the communication share; 0 skips profiling
    profile_steps: int = 0

    def __post_init__(self):
        if self.config not in CONFIGS:
            raise ValueError(f"unknown config {self.config!r}; known: {sorted(CONFIGS)}")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"unknown optimizer {self.optimizer!r}; known: {sorted(OPTIMIZERS)}")
        if self.devices < 1:
            raise ValueError(f"need at least one device, got {self.devices}")
        if self.timed_steps < 1 or self.warmup_steps < 0 or self.profile_steps < 0:
            raise ValueError("need at least one timed step and no negative warmup or profiled steps")


def parse_field(name: str, text: str) -> Any:
    """``text`` as the value of ``Setup.<name>``, typed like the field: for ``--compare name=text``."""
    types = {f.name: f.type for f in fields(Setup)}
    if name not in types:
        raise ValueError(f"Setup has no field {name!r}; fields: {list(types)}")
    kind = types[name]
    if kind is bool:
        if text.lower() not in ("true", "false"):
            raise ValueError(f"{name} is true or false, not {text!r}")
        return text.lower() == "true"
    return kind(text)
