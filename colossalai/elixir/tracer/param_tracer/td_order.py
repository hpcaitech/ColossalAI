import contextlib
import uuid
from typing import Callable, Dict, Iterator, List, Tuple, Union

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

from colossalai.elixir.tracer.ops import SameStorageAten
from colossalai.elixir.tracer.utils import meta_copy


@contextlib.contextmanager
def no_dispatch() -> Iterator[None]:
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard


def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


def register_storage(x):
    assert isinstance(x, nn.Parameter)
    assert x.data_ptr() == 0

    data_ptr = uuid.uuid1()
    x.data_ptr = lambda: data_ptr


class ATensor(torch.Tensor):
    elem: torch.Tensor

    __slots__ = ['elem']

    data_ptr_dict: Dict[int, Tuple[str, nn.Parameter]] = None
    order_list: List[Dict] = None

    @staticmethod
    def reset():
        ATensor.data_ptr_dict = dict()
        ATensor.order_list = list()

    @staticmethod
    def clear():
        ATensor.data_ptr_dict = None
        ATensor.order_list = None

    @staticmethod
    def add_data_ptr(name: str, param: nn.Parameter):
        data_ptr = param.data_ptr()
        if data_ptr not in ATensor.data_ptr_dict:
            ATensor.data_ptr_dict[data_ptr] = (name, param)
        else:
            name_in, param_in = ATensor.data_ptr_dict[data_ptr]
            if name != name_in or id(param) != id(param_in):
                raise RuntimeError('Got two different parameters with the same data ptr')

    @staticmethod
    def get_param(data_ptr: int):
        if data_ptr in ATensor.data_ptr_dict:
            return ATensor.data_ptr_dict.get(data_ptr)
        else:
            return None, None

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            elem.size(),
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
        # TODO: clone strides and storage aliasing
            dtype=elem.dtype,
            layout=elem.layout,
            device=elem.device,
            requires_grad=elem.requires_grad)
        r.elem = elem
        return r

    def __repr__(self):
        return f'ATensor({self.elem})'

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        step_dict = dict()

        def record_param(x):
            if isinstance(x, torch.Tensor):
                name, param = ATensor.get_param(x.data_ptr())
                if name is not None:
                    step_dict[name] = param

        def debug_tensor(x):
            if isinstance(x, torch.Tensor):
                print(type(x), x.shape, x.data_ptr(), id(x))
                if x.grad_fn:
                    print(x.grad_fn)

        tree_map(record_param, args)
        if len(step_dict) > 0:
            ATensor.order_list.append(step_dict)
        del step_dict

        def unwrap(x):
            return x.elem if isinstance(x, ATensor) else x

        def wrap(x):
            return ATensor(x) if isinstance(x, torch.Tensor) else x

        with no_dispatch():
            res = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
        outs = normalize_tuple(res)
        res = tree_map(wrap, outs)

        if func in SameStorageAten:
            for x in res:
                if isinstance(x, torch.Tensor):
                    x.data_ptr = args[0].data_ptr

        if len(res) == 1:
            return res[0]
        else:
            return res


def generate_td_order(model: nn.Module, inp: Union[torch.Tensor, Tuple], step_fn: Callable):
    ATensor.reset()

    def tensor_trans(t):
        meta_t = ATensor(t.data.to('meta'))
        if isinstance(t, nn.Parameter):
            meta_t = nn.Parameter(meta_t)
        return meta_t

    model = meta_copy(model, tensor_trans)
    for name, param in model.named_parameters():
        register_storage(param)
        ATensor.add_data_ptr(name, param)

    # convert all input data to meta_tensor
    if not isinstance(inp, tuple):
        inp = (inp,)
    inp = tree_map(lambda t: ATensor(torch.empty_like(t, device='meta', requires_grad=t.requires_grad)), inp)

    step_fn(model, inp)

    ret = ATensor.order_list
    ATensor.clear()

    return ret
