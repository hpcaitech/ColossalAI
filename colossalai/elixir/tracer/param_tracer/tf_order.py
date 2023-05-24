from typing import Callable, Dict, List

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.utils._pytree import tree_map

from colossalai.elixir.tensor import is_no_hook_op
from colossalai.elixir.tracer.utils import meta_copy
from colossalai.elixir.utils import no_dispatch, normalize_tuple

torch_checkpoint_function = torch.utils.checkpoint.checkpoint


def attach_checkpoint():
    default_in_checkpoint = False

    def inner_checkpoint_function(function, *args, use_reentrant: bool = True, **kwargs):
        nonlocal default_in_checkpoint
        prev_in_checkpoint = default_in_checkpoint
        default_in_checkpoint = True
        # record the step where going into checkpoint
        if not prev_in_checkpoint:
            Record.record_in_checkpoint()
        # use original torch checkpoint function
        global torch_checkpoint_function
        ret = torch_checkpoint_function(function, *args, use_reentrant=use_reentrant, **kwargs)
        # roll back
        default_in_checkpoint = prev_in_checkpoint
        if not default_in_checkpoint:
            Record.record_out_checkpoint()
        return ret

    torch.utils.checkpoint.checkpoint = inner_checkpoint_function


def release_checkpoint():
    global torch_checkpoint_function
    torch.utils.checkpoint.checkpoint = torch_checkpoint_function


class PostFwdPreBwd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, params, *args):
        ctx.params = params
        return args

    @staticmethod
    def backward(ctx, *grads):
        Record.record_params(ctx.params)
        return (None, *grads)


class Record(nn.Parameter):
    record_steps: List = None
    checkpoint_info: List = None
    in_checkpoint_step: int = -1

    def __new__(cls, elem):
        assert elem.device.type == 'meta', f'device type: {elem.device.type}'
        r = torch.Tensor._make_subclass(cls, elem)
        return r

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if is_no_hook_op(func):
            with torch._C.DisableTorchFunction():
                ret = func(*args, **kwargs)
            return ret

        params = list()

        def append_param(x):
            if isinstance(x, nn.Parameter):
                assert isinstance(x, Record)
                params.append(x)

        tree_map(append_param, args)
        tree_map(append_param, kwargs)
        Record.record_params(params)

        with torch._C.DisableTorchFunction():
            ret = normalize_tuple(func(*args, **kwargs))
            ret = PostFwdPreBwd.apply(params, *ret)

        def clone(t):
            if isinstance(t, torch.Tensor):
                t = t.clone()
            return t

        ret = tree_map(clone, ret)

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # notice: we should disable __torch_function__ here
        # otherwise, unexpected operations are called inside meta kernels
        with torch._C.DisableTorchFunction():
            with no_dispatch():
                return func(*args, **kwargs)

    @staticmethod
    def reset():
        Record.record_steps = list()
        Record.checkpoint_info = list()
        Record.in_checkpoint_step = -1

    @staticmethod
    def record_in_checkpoint():
        assert Record.in_checkpoint_step == -1
        Record.in_checkpoint_step = len(Record.record_steps)

    @staticmethod
    def record_out_checkpoint():
        assert Record.in_checkpoint_step != -1
        value_pair = (Record.in_checkpoint_step, len(Record.record_steps))
        Record.checkpoint_info.append(value_pair)
        Record.in_checkpoint_step = -1

    @staticmethod
    def steps():
        ret = dict(params_per_step=Record.record_steps, checkpoint_info=Record.checkpoint_info)
        Record.record_steps = None
        Record.checkpoint_info = None
        return ret

    @staticmethod
    def record_params(params):
        record_dict = {p.param_name for p in params}
        Record.record_steps.append(record_dict)


def generate_tf_order(model: nn.Module, inp: Dict, step_fn: Callable, dtype: torch.dtype = torch.float):
    assert isinstance(inp, dict), 'The example input should be a dictionary'

    Record.reset()

    def mtensor_trans(t: torch.Tensor):
        if t.is_floating_point():
            meta_dtype = dtype
        else:
            meta_dtype = t.dtype

        meta_t = torch.empty_like(t, dtype=meta_dtype, device='meta')
        if isinstance(t, nn.Parameter):
            meta_t = Record(meta_t)
            meta_t.requires_grad = t.requires_grad
        return meta_t

    model = meta_copy(model, mtensor_trans)
    for name, param in model.named_parameters():
        param.param_name = name

    def input_trans(t):
        if isinstance(t, torch.Tensor):
            if t.is_floating_point():
                meta_dtype = dtype
            else:
                meta_dtype = t.dtype

            meta_t = torch.empty_like(t, dtype=meta_dtype, device='meta', requires_grad=t.requires_grad)
            return meta_t
        return t

    inp = tree_map(input_trans, inp)
    attach_checkpoint()
    step_fn(model, inp)
    release_checkpoint()
    ret = Record.steps()
    return ret
