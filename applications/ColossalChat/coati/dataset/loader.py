#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataloader for sft, dpo, ppo
"""

import math
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from coati.dataset.utils import chuncate_sequence, pad_to_max_len
from datasets import Dataset as HFDataset
from datasets import dataset_dict, load_from_disk
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _get_default_group
from torch.utils.data import ConcatDataset, DataLoader, Dataset, DistributedSampler
from transformers.tokenization_utils import PreTrainedTokenizer

DatasetType = Union[Dataset, ConcatDataset, dataset_dict.Dataset]
PathType = Union[str, os.PathLike]


def load_tokenized_dataset(
    dataset_paths: Union[PathType, List[PathType]], mode: str = "train", **kwargs
) -> Optional[DatasetType]:
    """
    Load pre-tokenized dataset.
    Each instance of dataset is a dictionary with
    `{'input_ids': List[int], 'labels': List[int], sequence: str}` format.
    """
    mode_map = kwargs.get("mode_map", {"train": "train", "dev": "validation", "test": "test"})
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
            torch.LongTensor(instance["input_ids"][: self.max_length])
            if len(instance["input_ids"]) > self.max_length
            else torch.LongTensor(instance["input_ids"])
            for instance in instances
        ]
        batch_labels = [
            torch.LongTensor(instance["labels"][: self.max_length])
            if len(instance["labels"]) > self.max_length
            else torch.LongTensor(instance["labels"])
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


@dataclass
class DataCollatorForPromptDataset(DataCollatorForSupervisedDataset):
    def __call__(self, instances: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        """

        Args:
            instances (`Sequence[Dict[str, List[int]]]`):
                Mini-batch samples, each sample is stored in an individual dictionary.

        Returns:
            (`Dict[str, torch.Tensor]`): Contains the following `torch.Tensor`:
                `input_ids`: `torch.Tensor` of shape (bsz, max_len);
                `attention_mask`: `torch.BoolTensor` of shape (bsz, max_len);
        """
        instances = [{"input_ids": ins["input_ids"], "labels": ins["input_ids"]} for ins in instances]
        ret = super().__call__(instances=instances)
        input_ids = F.pad(
            ret["input_ids"], (self.max_length - ret["input_ids"].size(1), 0), value=self.tokenizer.pad_token_id
        )
        attention_mask = F.pad(ret["attention_mask"], (self.max_length - ret["attention_mask"].size(1), 0), value=False)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


@dataclass
class DataCollatorForPreferenceDataset(object):
    """
    Collate instances for supervised dataset.
    Each instance is a tokenized dictionary with fields
    `input_ids`(List[int]), `labels`(List[int]) and `sequence`(str).
    """

    tokenizer: PreTrainedTokenizer
    max_length: int = 4096

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

        (
            chosen_input_ids,
            chosen_loss_mask,  # [batch_size * seq_len]
            reject_input_ids,
            reject_loss_mask,
        ) = (
            chuncate_sequence([ins["chosen_input_ids"] for ins in instances], self.max_length, torch.int64),
            chuncate_sequence([ins["chosen_loss_mask"] for ins in instances], self.max_length, torch.bool),
            chuncate_sequence([ins["rejected_input_ids"] for ins in instances], self.max_length, torch.int64),
            chuncate_sequence([ins["rejected_loss_mask"] for ins in instances], self.max_length, torch.bool),
        )

        padding_side = self.tokenizer.padding_side
        chosen_attention_mask = [torch.ones_like(seq).bool() for seq in chosen_input_ids]
        reject_attention_mask = [torch.ones_like(seq).bool() for seq in reject_input_ids]

        (
            chosen_input_ids,
            chosen_attention_mask,
            chosen_loss_mask,
            reject_input_ids,
            reject_attention_mask,
            reject_loss_mask,
        ) = (
            pad_to_max_len(chosen_input_ids, self.max_length, self.tokenizer.pad_token_id, padding_side=padding_side),
            pad_to_max_len(chosen_attention_mask, self.max_length, False, padding_side=padding_side),
            pad_to_max_len(chosen_loss_mask, self.max_length, False, padding_side=padding_side),
            pad_to_max_len(reject_input_ids, self.max_length, self.tokenizer.pad_token_id, padding_side=padding_side),
            pad_to_max_len(reject_attention_mask, self.max_length, False, padding_side=padding_side),
            pad_to_max_len(reject_loss_mask, self.max_length, False, padding_side=padding_side),
        )

        return dict(
            chosen_input_ids=chosen_input_ids,
            chosen_attention_mask=chosen_attention_mask,
            chosen_loss_mask=chosen_loss_mask,
            reject_input_ids=reject_input_ids,
            reject_attention_mask=reject_attention_mask,
            reject_loss_mask=reject_loss_mask,
        )


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
        use_tp: Optional[bool] = False,
    ) -> None:
        if not use_tp:
            super().__init__(
                dataset=dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle,
                seed=seed,
                drop_last=drop_last,
            )
        else:
            # adapted from https://github.com/pytorch/pytorch/blob/4979f9c0d72490970e2019bb1d2284f83d93f76b/torch/utils/data/distributed.py#L62
            # TODO: support tp_group>1. will fix it later
            num_replicas = 1
            if rank is None:
                rank = dist.get_rank()
            if rank < 0:
                raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, 0]")
            self.dataset = dataset
            self.num_replicas = num_replicas
            self.rank = rank
            self.epoch = 0
            self.drop_last = drop_last
            # If the dataset length is evenly divisible by # of replicas, then there
            # is no need to drop any data, since the dataset will be split equally.
            if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
                # Split to nearest available length that is evenly divisible.
                # This is to ensure each rank receives the same amount of data when
                # using this Sampler.
                self.num_samples = math.ceil(
                    (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
                )
            else:
                self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
            self.total_size = self.num_samples * self.num_replicas
            self.shuffle = shuffle
            self.seed = seed
        self.start_index = 0
        self.use_tp = use_tp

    def __iter__(self) -> Iterator:
        if self.use_tp:
            # TODO Add support for tp_group not equal to 1
            pass
            # adpated from https://github.com/pytorch/pytorch/blob/4979f9c0d72490970e2019bb1d2284f83d93f76b/torch/utils/data/distributed.py#L96
            if self.shuffle:
                # deterministically shuffle based on epoch and seed
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
            else:
                indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

            if not self.drop_last:
                # add extra samples to make it evenly divisible
                padding_size = self.total_size - len(indices)
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
            else:
                # remove tail of data to make it evenly divisible.
                indices = indices[: self.total_size]
            assert len(indices) == self.total_size

            # subsample
            indices = indices[
                : self.total_size : self.num_replicas
            ]  # num_replicas=tp_group=1, we only support tp_group==1 for now
            assert len(indices) == self.num_samples

            return iter(indices)

        else:
            iterator = super().__iter__()
            indices = list(iterator)
            indices = indices[self.start_index :]
            return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def set_start_index(self, start_index: int) -> None:
        self.start_index = start_index


def setup_distributed_dataloader(
    dataset: DatasetType,
    batch_size: int = 1,
    shuffle: bool = False,
    seed: int = 1024,
    drop_last: bool = False,
    pin_memory: bool = False,
    num_workers: int = 0,
    collate_fn: Callable[[Sequence[Dict[str, Union[str, List[int]]]]], Dict[str, torch.Tensor]] = None,
    process_group: Optional[ProcessGroup] = None,
    use_tp: Optional[bool] = False,
    **kwargs,
) -> DataLoader:
    """
    Setup dataloader for distributed training.
    """
    _kwargs = kwargs.copy()
    process_group = process_group or _get_default_group()
    sampler = StatefulDistributedSampler(
        dataset=dataset,
        num_replicas=process_group.size() if not use_tp else 1,
        rank=process_group.rank(),
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
        use_tp=use_tp,
    )

    # Deterministic dataloader
    def seed_worker(worker_id: int) -> None:
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        **_kwargs,
    )
