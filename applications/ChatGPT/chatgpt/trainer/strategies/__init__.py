from .base import Strategy
# from .colossalai import ColossalAIStrategy
from .ddp import DDPStrategy
from .deepspeed import DeepspeedStrategy
from .naive import NaiveStrategy

__all__ = ['Strategy', 'NaiveStrategy', 'DDPStrategy', 'DeepspeedStrategy']
