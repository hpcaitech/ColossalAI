from .activation_checkpoint import checkpoint
from .common import print_rank_0, sync_model_param_in_dp, is_dp_rank_0, is_tp_rank_0, is_no_pp_or_last_stage
from .cuda import get_current_device, synchronize, empty_cache, set_to_cuda
from .memory import report_memory_usage
from .timer import MultiTimer, Timer

_GLOBAL_MULTI_TIMER = MultiTimer(on=False)


def get_global_multitimer():
    return _GLOBAL_MULTI_TIMER


def set_global_multitimer_status(mode: bool):
    _GLOBAL_MULTI_TIMER.set_status(mode)


__all__ = ['checkpoint', 'print_rank_0', 'sync_model_param_in_dp', 'get_current_device',
           'synchronize', 'empty_cache', 'set_to_cuda', 'report_memory_usage', 'Timer', 'MultiTimer',
           'get_global_multitimer', 'set_global_multitimer_status',
           'is_dp_rank_0', 'is_tp_rank_0', 'is_no_pp_or_last_stage'
           ]
