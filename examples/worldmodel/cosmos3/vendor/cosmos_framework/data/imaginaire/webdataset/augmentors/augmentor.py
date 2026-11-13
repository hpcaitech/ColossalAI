# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from collections.abc import Iterable
from typing import Any, Generator, Optional


class Augmentor:
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        r"""Base augmentor class

        Args:
            input_keys (list): List of input keys
            output_keys (list): List of output keys
            args (dict): Arguments associated with the augmentation
        """
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.args = args

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise ValueError("Augmentor not implemented")


class IterableAugmentor:
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        r"""Base augmentor class

        Args:
            input_keys (list): List of input keys
            output_keys (list): List of output keys
            args (dict): Arguments associated with the augmentation
        """
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.args = args
        self.is_generator = True

    def __call__(self, data: Iterable) -> Generator:
        r"""Example usage:

        for data_dict in data:
            # Do something to data_dict
            data_dict["input"] = data_dict["raw_sequence"][:, :-1]
            data_dict["target"] = data_dict["raw_sequence"][:, 1:]
            # Skip sample if needed
            if data_dict["input"].shape[1] < 64:
                continue
            # Construct a generator
            yield data_dict
        """
        raise ValueError("Augmentor not implemented")
