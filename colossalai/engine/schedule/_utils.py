#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Union, List

from torch import Tensor


def convert_to_fp16(data: Union[Tensor, List[Tensor]]):
    if isinstance(data, Tensor):
        ret = data.half()
    elif isinstance(data, (list, tuple)):
        ret = [val.half() for val in data]
    else:
        raise TypeError(f"Expected argument 'data' to be a Tensor or a list/tuple of Tensor, but got {type(data)}")
    return ret
