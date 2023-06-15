# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Type, TypeVar

import torch


def get_operator(library: str, name: str):
    def no_such_operator(*args, **kwargs):
        raise RuntimeError(
            f"No such operator {library}::{name} - did you forget to build xformers with `python setup.py develop`?"
        )

    try:
        return getattr(getattr(torch.ops, library), name)
    except (RuntimeError, AttributeError):
        return no_such_operator


def get_xformers_operator(name: str):
    return get_operator("xformers", name)


class BaseOperator:
    OPERATOR: Any
    NAME: str
    OPERATOR_CATEGORY: str

    @classmethod
    def is_available(cls) -> bool:
        if cls.OPERATOR is None or cls.OPERATOR.__name__ == "no_such_operator":
            return False
        return True

    @classmethod
    def operator_flop(cls, *inputs) -> int:
        """Calculate number of FLOP given inputs to `OPERATOR`"""
        return -1


OPERATORS_REGISTRY: List[Type[BaseOperator]] = []
FUNC_TO_XFORMERS_OPERATOR: Dict[Any, Type[BaseOperator]] = {}

ClsT = TypeVar("ClsT")


def register_operator(cls: ClsT) -> ClsT:
    global OPERATORS_REGISTRY, FUNC_TO_XFORMERS_OPERATOR
    OPERATORS_REGISTRY.append(cls)  # type: ignore
    FUNC_TO_XFORMERS_OPERATOR[cls.OPERATOR] = cls  # type: ignore
    return cls
