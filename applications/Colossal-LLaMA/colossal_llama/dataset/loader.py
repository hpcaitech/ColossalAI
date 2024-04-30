#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from datasets import dataset_dict, load_from_disk
from torch.utils.data import ConcatDataset, Dataset, DistributedSampler
from transformers.tokenization_utils import PreTrainedTokenizer

DatasetType = Union[Dataset, ConcatDataset, dataset_dict.Dataset]
PathType = Union[str, os.PathLike]


def load_tokenized_dataset(
    dataset_paths: Union[PathType, List[PathType]], mode: str = "train"
) -> Optional[DatasetType]:
    """
    Load pre-tokenized dataset.
    Each instance of dataset is a dictionary with
    `{'input_ids': List[int], 'labels': List[int], sequence: str}` format.
    """
    mode_map = {"train": "train", "dev": "validation", "test": "test"}
    assert mode in tuple(mode_map), f"Unsupported mode {mode}, it must be in {tuple(mode_map)}"

    if isinstance(dataset_paths, (str, os.PathLike)):
        dataset_paths = [dataset_paths]

    datasets = []  # `List[datasets.dataset_dict.Dataset]`
    for ds_path in dataset_paths:
        ds_path = os.path.abspath(ds_path)
        assert os.path.exists(ds_path), f"Not existed file path {ds_path}"
        ds_dict = load_from_disk(dataset_path=ds_path, keep_in_memory=False)
        if isinstance(ds_dict, HFDataset):
            datasets.append(ds_dict)
        else:
            if mode_map[mode] in ds_dict:
                datasets.append(ds_dict[mode_map[mode]])
    if len(datasets) == 0:
        return None
    if len(datasets) == 1:
        return datasets.pop()
    return ConcatDataset(datasets=datasets)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """
    Collate instances for supervised dataset.
    Each instance is a tokenized dictionary with fields
    `input_ids`(List[int]), `labels`(List[int]) and `sequence`(str).
    """

    tokenizer: PreTrainedTokenizer
    max_length: int = 4096
    ignore_index: int = -100
    padding: str = "max_length"

    def __call__(self, instances: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        """

        Args:
            instances (`Sequence[Dict[str, List[int]]]`):
                Mini-batch samples, each sample is stored in an individual dictionary.

        Returns:
            (`Dict[str, torch.Tensor]`): Contains the following `torch.Tensor`:
                `input_ids`: `torch.Tensor` of shape (bsz, max_len);
                `attention_mask`: `torch.BoolTensor` of shape (bsz, max_len);
                `labels`: `torch.Tensor` of shape (bsz, max_len), which contains `IGNORE_INDEX`.
        """
        assert isinstance(self.tokenizer.pad_token_id, int) and self.tokenizer.pad_token_id >= 0, (
            f"`{self.tokenizer.__class__.__name__}.pad_token_id` must be a valid non-negative integer index value, "
            f"but now `{self.tokenizer.pad_token_id}`"
        )

        # `List[torch.Tensor]`
        batch_input_ids = [
            (
                torch.LongTensor(instance["input_ids"][: self.max_length])
                if len(instance["input_ids"]) > self.max_length
                else torch.LongTensor(instance["input_ids"])
            )
            for instance in instances
        ]
        batch_labels = [
            (
                torch.LongTensor(instance["labels"][: self.max_length])
                if len(instance["labels"]) > self.max_length
                else torch.LongTensor(instance["labels"])
            )
            for instance in instances
        ]

        if self.tokenizer.padding_side == "right":
            input_ids = torch.nn.utils.rnn.pad_sequence(
                sequences=batch_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )  # (bsz, max_len)
            labels = torch.nn.utils.rnn.pad_sequence(
                sequences=batch_labels,
                batch_first=True,
                padding_value=self.ignore_index,
            )  # (bsz, max_len)
            if self.padding == "max_length":
                # pad to max
                to_pad = self.max_length - input_ids.size(1)
                input_ids = F.pad(input_ids, (0, to_pad), value=self.tokenizer.pad_token_id)
                labels = F.pad(labels, (0, to_pad), value=self.ignore_index)
        elif self.tokenizer.padding_side == "left":
            reversed_input_ids = [seq.flip(dims=(0,)) for seq in batch_input_ids]
            reversed_input_ids = torch.nn.utils.rnn.pad_sequence(
                sequences=reversed_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )  # (bsz, max_len)
            input_ids = torch.flip(reversed_input_ids, dims=(1,))  # (bsz, max_len)
            reversed_labels = [seq.flip(dims=(0,)) for seq in batch_labels]
            reversed_labels = torch.nn.utils.rnn.pad_sequence(
                sequences=reversed_labels,
                batch_first=True,
                padding_value=self.ignore_index,
            )  # (bsz, max_len)
            labels = torch.flip(reversed_labels, dims=(1,))  # (bsz, max_len)
        else:
            raise RuntimeError(
                f"`{self.tokenizer.__class__.__name__}.padding_side` can only be `left` or `right`, "
                f"but now `{self.tokenizer.padding_side}`"
            )

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)  # `torch.BoolTensor`, (bsz, max_len)

        return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


class StatefulDistributedSampler(DistributedSampler):
    """
    Stateful distributed sampler for multi-stage training.
    """

    def __init__(
        self,
        dataset: DatasetType,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.start_index = 0

    def __iter__(self) -> Iterator:
        iterator = super().__iter__()
        indices = list(iterator)
        indices = indices[self.start_index :]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def set_start_index(self, start_index: int) -> None:
        self.start_index = start_index
