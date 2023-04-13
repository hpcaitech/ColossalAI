import torch.distributed as dist
from typing import Any, Callable, Dict, List, Optional
from coati.models.bloom import BLOOMActor, BLOOMCritic
from coati.models.gpt import GPTActor, GPTCritic
from coati.models.opt import OPTActor, OPTCritic
from coati.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy
import torch
import os


def is_rank_0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


