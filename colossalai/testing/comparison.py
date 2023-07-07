from typing import Any, List, OrderedDict

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.testing import assert_close
from torch.utils._pytree import tree_flatten


def assert_equal(a: Tensor, b: Tensor):
    assert torch.all(a == b), f'expected a and b to be equal but they are not, {a} vs {b}'


def assert_not_equal(a: Tensor, b: Tensor):
    assert not torch.all(a == b), f'expected a and b to be not equal but they are, {a} vs {b}'


def assert_close_loose(a: Tensor, b: Tensor, rtol: float = 1e-3, atol: float = 1e-3):
    assert_close(a,
                 b,
                 rtol=rtol,
                 atol=atol,
                 msg=f"Tensor not close, shape: {a.shape} vs {b.shape}, \
                                                   dtype: {a.dtype} vs {b.dtype}")


def assert_equal_in_group(tensor: Tensor, process_group: ProcessGroup = None):
    # all gather tensors from different ranks
    world_size = dist.get_world_size(process_group)
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor, group=process_group)

    # check if they are equal one by one
    for i in range(world_size - 1):
        a = tensor_list[i]
        b = tensor_list[i + 1]
        assert torch.all(a == b), f'expected tensors on rank {i} and {i + 1} to be equal but they are not, {a} vs {b}'


def check_state_dict_equal(d1: OrderedDict, d2: OrderedDict, ignore_device: bool = True):
    assert len(list(d1.keys())) == len(list(d2.keys())), \
           f"Number of keys unequal: {len(list(d1.keys()))} vs {len(list(d2.keys()))}"
    for k, v1 in d1.items():
        assert k in d2
        v2 = d2[k]
        if isinstance(v1, dict):
            assert isinstance(v2, dict)
            check_state_dict_equal(v1, v2, ignore_device)
        elif isinstance(v1, list):
            assert isinstance(v2, list)
            for v1_i, v2_i in zip(v1, v2):
                if isinstance(v1_i, torch.Tensor):
                    assert isinstance(v2_i, torch.Tensor)
                    if not ignore_device:
                        v1_i = v1_i.to("cpu")
                        v2_i = v2_i.to("cpu")
                    assert_close_loose(v1_i, v2_i)
                elif isinstance(v1_i, dict):
                    assert isinstance(v2_i, dict)
                    check_state_dict_equal(v1_i, v2_i, ignore_device)
                else:
                    assert v1_i == v2_i, f"{v1_i} not equals to {v2_i}"
        elif isinstance(v1, torch.Tensor):
            assert isinstance(v2, torch.Tensor)
            if not ignore_device:
                v1 = v1.to("cpu")
                v2 = v2.to("cpu")
            assert_close_loose(v1, v2)
        else:
            assert v1 == v2, f"{v1} not equals to {v2}"


def check_state_dict_equal_pytree(d1: OrderedDict, d2: OrderedDict, ignore_device: bool = True):
    flat_d1, _ = tree_flatten(d1)
    flat_d2, _ = tree_flatten(d2)
    assert len(flat_d1) == len(flat_d2)
    for v1, v2 in zip(flat_d1, flat_d2):
        if isinstance(v1, torch.Tensor):
            assert isinstance(v2, torch.Tensor)
            if not ignore_device:
                v1 = v1.to("cpu")
                v2 = v2.to("cpu")
            assert_close_loose(v1, v2)
        else:
            assert v1 == v2, f"{v1} not equals to {v2}"


def assert_hf_output_close(out1: Any,
                           out2: Any,
                           ignore_keys: List[str] = None,
                           track_name: str = "",
                           atol=1e-5,
                           rtol=1e-5):
    """
    Check if two outputs from huggingface are equal.

    Args:
        out1 (Any): the first output
        out2 (Any): the second output
        ignore_keys (List[str]): the keys to ignore when comparing two dicts
        track_name (str): the name of the value compared, used to track the path
    """
    if isinstance(out1, dict) and isinstance(out2, dict):
        # if two values are dict
        # we recursively check the keys
        assert set(out1.keys()) == set(out2.keys())
        for k in out1.keys():
            if ignore_keys is not None and k in ignore_keys:
                continue
            assert_hf_output_close(out1[k],
                                   out2[k],
                                   track_name=f"{track_name}.{k}",
                                   ignore_keys=ignore_keys,
                                   atol=atol,
                                   rtol=rtol)
    elif isinstance(out1, (list, tuple)) and isinstance(out2, (list, tuple)):
        # if two values are list
        # we recursively check the elements
        assert len(out1) == len(out2)
        for i in range(len(out1)):
            assert_hf_output_close(out1[i],
                                   out2[i],
                                   track_name=f"{track_name}.{i}",
                                   ignore_keys=ignore_keys,
                                   atol=atol,
                                   rtol=rtol)
    elif isinstance(out1, Tensor) and isinstance(out2, Tensor):
        if out1.shape != out2.shape:
            raise AssertionError(f"{track_name}: shape mismatch: {out1.shape} vs {out2.shape}")
        assert torch.allclose(
            out1, out2, atol=atol, rtol=rtol
        ), f"{track_name}: tensor value mismatch\nvalue 1: {out1}\nvalue 2: {out2}, \nmean error: {torch.abs(out1 - out2).mean()}"
    else:
        assert out1 == out2, f"{track_name}: value mismatch.\nout1: {out1}\nout2: {out2}"
