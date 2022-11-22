from .data_parallel import ColoDDP, ZeroDDP
from .gemini_parallel import GeminiDDP

__all__ = ['ColoDDP', 'ZeroDDP', 'GeminiDDP']
