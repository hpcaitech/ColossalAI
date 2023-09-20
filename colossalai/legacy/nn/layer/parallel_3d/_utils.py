from collections import OrderedDict
from functools import partial

import torch
from torch import Tensor

from colossalai.legacy.constants import (
    INPUT_GROUP_3D,
    INPUT_X_WEIGHT_3D,
    OUTPUT_GROUP_3D,
    OUTPUT_X_WEIGHT_3D,
    WEIGHT_GROUP_3D,
)
from colossalai.legacy.core import global_context as gpc
from colossalai.legacy.global_variables import tensor_parallel_env as env


def get_depth_from_env() -> int:
    try:
        depth = env.depth_3d
        assert depth > 0, "DEPTH must be greater than zero"
        return depth

    except KeyError:
        raise EnvironmentError(
            "DEPTH is not found in the current environment, "
            "please make sure that you have used the correct process group initializer"
        )


def get_parallel_mode_from_env(group):
    assert group in [
        INPUT_GROUP_3D,
        WEIGHT_GROUP_3D,
        OUTPUT_GROUP_3D,
        INPUT_X_WEIGHT_3D,
        OUTPUT_X_WEIGHT_3D,
    ], f"{group} is not valid for 3D tensor parallelism."
    return getattr(env, group)


def swap_in_out_group():
    env.input_group_3d, env.output_group_3d = env.output_group_3d, env.input_group_3d
    env.input_x_weight_group_3d, env.output_x_weight_group_3d = (
        env.output_x_weight_group_3d,
        env.input_x_weight_group_3d,
    )


def dbg_check_shape(tensor: Tensor, shape: tuple):
    rank = gpc.get_global_rank()
    if rank == 0:
        print(tensor.shape)
    assert tensor.shape == shape, "{} does not match {}".format(tensor.shape, shape)


class AsyncGradientBucket(object):
    def __init__(self):
        self.bucket = OrderedDict()

    def __len__(self):
        return len(self.bucket)

    def push(self, async_op, grad_tensor, param_id):
        self.bucket[param_id] = tuple((async_op, grad_tensor))
        return torch.zeros_like(grad_tensor, dtype=grad_tensor.dtype, device=grad_tensor.device)

    def pop(self, param_id):
        grad = None
        if param_id in self.bucket:
            op, grad = self.bucket.pop(param_id)
            if op is not None:
                op.wait()
        return grad

    def synchronize(self, params):
        for p in params:
            i = id(p)
            if i in self.bucket:
                op, grad = self.bucket.pop(i)
                if op is not None:
                    op.wait()
                p.grad.add_(grad)


_async_grad_bucket = AsyncGradientBucket()


def push_async_grad(op, grad, param_id):
    return _async_grad_bucket.push(op, grad, param_id)


def pop_async_grad(param_id):
    return _async_grad_bucket.pop(param_id)


def _async_grad_hook(grad, param_id):
    grad.add_(pop_async_grad(param_id))
    return grad


def register_async_grad_hook(param):
    param.register_hook(partial(_async_grad_hook, param_id=id(param)))


def synchronize(params=list()):
    _async_grad_bucket.synchronize(params)
    torch.cuda.default_stream().synchronize()
    if len(_async_grad_bucket) > 0:
        raise RuntimeError(f"{len(_async_grad_bucket)} asynchronous gradient(s) not collected.")
