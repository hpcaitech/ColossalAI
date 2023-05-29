import contextlib
from typing import Dict, Iterator, Tuple

import torch

from colossalai.elixir.tracer.utils import get_cuda_allocated, get_cuda_max_allocated

from .output_shape import addmm_output, bmm_output, check_cuda_mm, mm_output


@contextlib.contextmanager
def no_dispatch() -> Iterator[None]:
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard


def tensor_info(x: torch.Tensor):
    # returns the meta information used for CUDA kernels
    return (x.shape, x.stride(), x.layout, x.dtype)


def get_args_info(*args):
    # returns a tuple contains the meta information of all inputs
    # every argument is expected to be a tensor
    info_list = []
    for x in args:
        if isinstance(x, torch.Tensor):
            info_list.append(tensor_info(x))
    return tuple(info_list)


class OpCache(object):

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.temp_memory: Dict[Tuple, int] = dict()

    def reset(self):
        self.temp_memory.clear()

    def get(self, info):
        if info in self.temp_memory:
            return True, self.temp_memory[info]
        else:
            return False, None

    def add(self, info, memo):
        self.temp_memory[info] = memo

    def print(self):
        print(f'OpCache {self.name} information:')
        for k, v in self.temp_memory.items():
            print(f'key: {k}\ntemp_memo:{v}')


aten = torch.ops.aten
addmm_cache = OpCache('aten.addmm.default')
bmm_cache = OpCache('aten.bmm.default')
mm_cache = OpCache('aten.mm.default')

op_mapping = {
    aten.mm.default: {
        'cache': mm_cache,
        'output': mm_output
    },
    aten.addmm.default: {
        'cache': addmm_cache,
        'output': addmm_output
    },
    aten.bmm.default: {
        'cache': bmm_cache,
        'output': bmm_output
    }
}


def reset_caches():
    addmm_cache.reset()
    bmm_cache.reset()
    mm_cache.reset()


def fake_cuda_output(temp_memo, output_shape, dtype):
    ret = torch.empty(output_shape, dtype=dtype, device='cuda')
    sub = temp_memo - ret.numel() * ret.element_size()

    if sub > 0:
        # allocate a temp empty tensor block to simulate the computation in kernels
        temp = torch.empty(sub, dtype=torch.int8, device='cuda')
        # release this tensor block
        del temp

    return ret


def real_cuda_output(func, *args, **kwargs):
    cur_alc = get_cuda_allocated()
    # save the peak memory usage
    pre_max_alc = get_cuda_max_allocated()
    # the peak memory history is cleared here
    torch.cuda.reset_peak_memory_stats()

    with no_dispatch():
        ret = func(*args, **kwargs)

    max_alc = get_cuda_max_allocated()
    # calculate the temporary memory allocation
    temp_memo = max_alc - cur_alc

    return ret, temp_memo, pre_max_alc


def wrapped_mm_ops(func, *args, **kwargs):
    check_cuda_mm(*args)

    if func not in op_mapping:
        raise RuntimeError(f'Unsupported mm operation {func}')

    args_info = get_args_info(*args)
    cache = op_mapping[func]['cache']
    cached_flag, temp_memo = cache.get(args_info)

    if cached_flag:
        output_fn = op_mapping[func]['output']
        out_shape = output_fn(*args)
        ret = fake_cuda_output(temp_memo=temp_memo, output_shape=out_shape, dtype=args[0].dtype)
        return ret, 0
    else:
        ret, temp_memo, pre_max_alc = real_cuda_output(func, *args, **kwargs)
        cache.add(args_info, temp_memo)
        return ret, pre_max_alc
