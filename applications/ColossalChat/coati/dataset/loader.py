#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataloader for sft, dpo, ppo
"""

import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from coati.dataset.utils import chuncate_sequence, pad_to_max_len
from datasets import Dataset as HFDataset
from datasets import dataset_dict, load_from_disk
from torch.utils.data import ConcatDataset, Dataset, DistributedSampler
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
    if not dataset_paths:
        return None
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


@dataclass
class DataCollatorForKTODataset(object):
    """
    Collate instances for kto dataset.
    Each input instance is a tokenized dictionary with fields
    `prompt`(List[int]), `completion`(List[int]) and `label`(bool).
    Each output instance is a tokenized dictionary with fields
    `kl_input_ids`(List[int]), `kl_attention_mask`(List[int]) and `kl_loss_mask`(List[int]).
    `input_ids`(List[int]), `attention_mask`(List[int]), `loss_mask`(List[int]) and `label`(bool).
    """

    tokenizer: PreTrainedTokenizer
    max_length: int = 4096
    ignore_index: int = -100

    def __call__(self, instances: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        """

        Args:
            instances (`Sequence[Dict[str, List[int]]]`):
                Mini-batch samples, each sample is stored in an individual dictionary contains the following fields:
                `prompt`(List[int]), `completion`(List[int]) and `label`(bool, if the sample is desirable or not).

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
        # prepare the preference data
        prompt = [torch.LongTensor(instance["prompt"]) for instance in instances]
        prompt_zeros = [torch.zeros_like(t) for t in prompt]
        completion = [torch.LongTensor(instance["completion"]) for instance in instances]
        completion_ones = [torch.ones_like(t) for t in completion]
        label = [torch.tensor(instance["label"], dtype=torch.bool) for instance in instances]
        input_ids = [torch.cat([prompt[i], completion[i]], dim=-1) for i in range(len(instances))]
        loss_mask = [torch.cat([prompt_zeros[i], completion_ones[i]], dim=-1) for i in range(len(instances))]
        # right padding
        input_ids = torch.nn.utils.rnn.pad_sequence(
            sequences=input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )  # (bsz, max_len)
        loss_mask = torch.nn.utils.rnn.pad_sequence(
            sequences=loss_mask, batch_first=True, padding_value=0
        )  # (bsz, max_len)
        to_pad = self.max_length - input_ids.size(1)
        input_ids = F.pad(input_ids, (0, to_pad), value=self.tokenizer.pad_token_id)
        loss_mask = F.pad(loss_mask, (0, to_pad), value=0)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)  # `torch.BoolTensor`, (bsz, max_len)

        # prepare kt data
        kl_completion = completion[::-1]  # y'
        kl_completion_ones = [torch.ones_like(t) for t in kl_completion]
        kl_input_ids = [torch.cat([prompt[i], kl_completion[i]], dim=-1) for i in range(len(instances))]
        kl_loss_mask = [torch.cat([prompt_zeros[i], kl_completion_ones[i]], dim=-1) for i in range(len(instances))]
        # right padding
        kl_input_ids = torch.nn.utils.rnn.pad_sequence(
            sequences=kl_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )  # (bsz, max_len)
        kl_loss_mask = torch.nn.utils.rnn.pad_sequence(
            sequences=kl_loss_mask, batch_first=True, padding_value=0
        )  # (bsz, max_len)
        to_pad = self.max_length - kl_input_ids.size(1)
        kl_input_ids = F.pad(kl_input_ids, (0, to_pad), value=self.tokenizer.pad_token_id)
        kl_loss_mask = F.pad(kl_loss_mask, (0, to_pad), value=0)
        kl_attention_mask = kl_input_ids.ne(self.tokenizer.pad_token_id)  # `torch.BoolTensor`, (bsz, max_len)
        data_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "label": torch.stack(label),
            "kl_input_ids": kl_input_ids,
            "kl_attention_mask": kl_attention_mask,
            "kl_loss_mask": kl_loss_mask,
        }
        return data_dict


class StatefulDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.start_index: int = 0

    def __iter__(self) -> Iterator:
        iterator = super().__iter__()
        indices = list(iterator)
        indices = indices[self.start_index :]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def set_start_index(self, start_index: int) -> None:
        self.start_index = start_index
