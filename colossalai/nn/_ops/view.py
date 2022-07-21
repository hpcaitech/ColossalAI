import math
import torch
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.tensor import ColoTensor, ColoTensorSpec, ReplicaSpec
from typing import Optional, Union


def _all_int(my_iter):
    return all(isinstance(i, int) for i in my_iter)


def _get_valid_shape(shape):
    if isinstance(shape, list):
        if _all_int(shape):
            return tuple(shape)
        else:
            raise RuntimeError("expects type(int) but finds an other type")
    elif isinstance(shape, tuple):
        if _all_int(shape):
            return shape
        else:
            return _get_valid_shape(shape[0])
    else:
        raise RuntimeError("expects an iterable array but finds '{}'".format(type(shape)))


def _shape_infer(org_sp, tgt_sp):
    cnt = 0
    pos = 0
    for idx, dim in enumerate(tgt_sp):
        if dim < -1:
            raise RuntimeError("invalid shape dimension {}".format(dim))
        elif dim == -1:
            cnt += 1
            pos = idx

    if cnt > 1:
        raise RuntimeError("only one dimension can be inferred")

    org_prod = math.prod(org_sp)
    tgt_prod = math.prod(tgt_sp)

    if cnt == 0:
        if org_prod != tgt_prod:
            raise RuntimeError("shape '{}' is invalid for input of size {}".format(tgt_sp, org_prod))
        else:
            return tgt_sp
    elif org_prod % tgt_prod != 0:
        raise RuntimeError("shape '{}' is invalid for input of size {}".format(tgt_sp, org_prod))

    infer_dim = -(org_prod // tgt_prod)
    return tgt_sp[: pos] + (infer_dim,) + tgt_sp[pos + 1:]


@colo_op_impl(torch.Tensor.view)
def colo_view(self: ColoTensor, *shape) -> 'ColoTensor':
    """Handles ``__torch_function__`` dispatch for ``torch.Tensor.view``.
    Changes the shape of the current tensor.
    """
    assert isinstance(self, ColoTensor)
    # apply original `view` function for replicated colo tensors
    if self.is_replicate():
        return self.view(*shape)

    cur_sp = self.size()
    org_sp = self.size_global()
    # parse the passed arguments
    tgt_sp = _get_valid_shape(shape)
    # get the correct shape from inference
    inf_sp = _shape_infer(org_sp, tgt_sp)

    if self.is_shard_1drow() and org_sp[0] == inf_sp[0]:
        new_shape = (cur_sp[0],) + tgt_sp[1:]
        res = self.view(*new_shape)
    elif self.is_shard_1dcol() and org_sp[-1] == inf_sp[-1]:
        new_shape = tgt_sp[:-1] + (cur_sp[-1],)
        res = self.view(*new_shape)
    else:
        replicated_t = self.redistribute(dist_spec=ReplicaSpec())
        return ColoTensor.from_torch_tensor(
            tensor=replicated_t.view(*shape),
            spec=ColoTensorSpec(self.get_process_group()))

    return ColoTensor.from_torch_tensor(
        tensor=res,
        spec=ColoTensorSpec(
            pg=self.get_process_group(),
            dist_attr=self.dist_spec))


@colo_op_impl(torch.Tensor.size)
def colo_size(self: ColoTensor, dim: Optional[int] = None) -> Union[torch.Size, int]:
    size = self.size_global()
    if dim is None:
        return size
    else:
        return size[dim]
