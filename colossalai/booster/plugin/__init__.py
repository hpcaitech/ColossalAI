from .gemini_plugin import GeminiPlugin
from .low_level_zero_plugin import LowLevelZeroPlugin
from .plugin_base import Plugin
from .torch_ddp_plugin import TorchDDPPlugin

__all__ = ['Plugin', 'TorchDDPPlugin', 'GeminiPlugin', 'LowLevelZeroPlugin']

import torch
from packaging import version

if version.parse(torch.__version__) >= version.parse('1.12.0'):
    from .torch_fsdp_plugin import TorchFSDPPlugin
    __all__.append('TorchFSDPPlugin')
