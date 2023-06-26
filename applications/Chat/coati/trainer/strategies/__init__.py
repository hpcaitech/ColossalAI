from .base import Strategy
from .colossalai import ColossalAIStrategy
from .ddp import DDPStrategy

__all__ = ['Strategy', 'DDPStrategy', 'ColossalAIStrategy']
