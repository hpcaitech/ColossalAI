#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
from typing import Dict, Optional, Sequence, Tuple

import torch
from coati.models.chatglm.chatglm_tokenizer import ChatGLMTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from colossalai.cluster import DistCoordinator

from .utils import is_rank_0, jload, read_string_by_schema

IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def _preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preprocess the data by tokenizing."""
    sequences = [s + t for s, t in zip(sources, targets)]
    sequences_token = tokenizer(
        sequences, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    sources_token = tokenizer(
        sources, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
    )

    assert sequences_token["attention_mask"].dim() == 2, "seq2seq model should be preprocessed differently"
    labels = copy.deepcopy(sequences_token["input_ids"])
    for i in range(labels.shape[0]):
        source_len = sources_token["attention_mask"][i].sum().item()
        pad_len = max_length - sequences_token["attention_mask"][i].sum().item()
        if tokenizer.padding_side == "right":
            # |prompt|completion|eos|pad|
            labels[i][:source_len] = IGNORE_INDEX
            labels[i][-pad_len:] = IGNORE_INDEX
        elif tokenizer.padding_side == "left":
            # |pad|prompt|completion|eos|
            labels[i][: pad_len + source_len] = IGNORE_INDEX
        else:
            raise RuntimeError()

    return sequences_token["input_ids"], labels, sequences_token["attention_mask"]


def _preprocess_chatglm(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Preprocess the data by tokenizing.
    None for attention mask, ChatGLM will calculate attention mask according to input ids
    """

    labels = []
    input_ids = []
    for source, target in zip(sources, targets):
        source_id = tokenizer.encode(text=source, add_special_tokens=False)
        target_id = tokenizer.encode(text=target, add_special_tokens=False)
        input_id = tokenizer.build_inputs_with_special_tokens(source_id, target_id)
        # truncate
        sp_token_list = [tokenizer.gmask_token_id, tokenizer.bos_token_id]
        truncate_length = max(0, len(input_id) - max_length)
        input_id = input_id[truncate_length:]
        if truncate_length == len(source_id) + 1:
            input_id = sp_token_list + input_id[1:]
        elif truncate_length > len(source_id) + 1:
            input_id = sp_token_list + input_id[2:]

        context_length = input_id.index(tokenizer.bos_token_id)
        mask_position = context_length - 1
        label = [IGNORE_INDEX] * context_length + input_id[mask_position + 1 :]

        pad_len = max_length - len(input_id)
        input_id = input_id + [tokenizer.pad_token_id] * pad_len
        input_ids.append(input_id)
        labels.append(label + [IGNORE_INDEX] * pad_len)
    return torch.tensor(input_ids), torch.tensor(labels), None


class SFTDataset(Dataset):
    """
    Dataset for sft model

    Args:
        dataset: dataset for supervised model
        tokenizer: tokenizer for supervised model
        max_length: max length of input
        dataset_schema: schema for reading the dataset. cascaded feild names seperated by '.'.
             e.g. person.name.first will access data['person']['name']['first']
    """

    def __init__(
        self,
        dataset: Dict,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        dataset_schema: Dict[str, str] = {"prompt": "prompt", "completion": "completion"},
    ) -> None:
        super().__init__()
        self.input_ids = []
        self.coordinator = DistCoordinator()

        sources = [read_string_by_schema(data, dataset_schema["prompt"]) for data in dataset]
        targets = [
            read_string_by_schema(data, dataset_schema["completion"]) + tokenizer.eos_token
            for data in tqdm(dataset, disable=not is_rank_0())
        ]

        self.coordinator.print_on_master("Tokenizing inputs... This may take some time...")
        if isinstance(tokenizer, ChatGLMTokenizer):
            self.input_ids, self.labels, self.attention_mask = _preprocess_chatglm(
                sources, targets, tokenizer, max_length
            )
        else:
            self.input_ids, self.labels, self.attention_mask = _preprocess(sources, targets, tokenizer, max_length)

        self.coordinator.print_on_master("Loaded dataset.")

    def __len__(self):
        length = self.input_ids.shape[0]
        return length

    def __getitem__(self, idx):
        if self.attention_mask is not None:
            return dict(input_ids=self.input_ids[idx], labels=self.labels[idx], attention_mask=self.attention_mask[idx])
        else:
            return dict(input_ids=self.input_ids[idx], labels=self.labels[idx])


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.

    Args:
        dataset: dataset for supervised model
        tokenizer: tokenizer for supervised model
        max_datasets_size: number of examples to use from the dataset
        max_length: max length of input
        prompt_dict: prompts for the dataset used to format prompt
        dataset_schema: schema for reading the dataset. cascaded feild names seperated by '.'.
             e.g. person.name.first will access data['person']['name']['first']
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_datasets_size: Optional[int] = None,
        max_length: int = 512,
        prompt_dict: Optional[Dict[str, str]] = PROMPT_DICT,
        split: str = "train",
        dataset_schema: Dict[str, str] = {"instruction": "instruction", "input": "input", "output": "output"},
    ):
        super().__init__()
        self.coordinator = DistCoordinator()
        self.coordinator.print_on_master("Loading data...")
        try:
            dataset = load_dataset(data_path)
            list_data_dict = list(dataset[split])
        except FileNotFoundError:
            list_data_dict = jload(data_path)
        self.coordinator.print_on_master(f"Loaded {len(list_data_dict)} examples.")

        if max_datasets_size is not None:
            self.coordinator.print_on_master(f"Limiting dataset to {max_datasets_size} examples.")
            list_data_dict = list_data_dict[:max_datasets_size]

        self.coordinator.print_on_master("Formatting inputs...")
        prompt_input, prompt_no_input = prompt_dict["prompt_input"], prompt_dict["prompt_no_input"]
        list_data_dict = [
            {k: read_string_by_schema(example, dataset_schema[k]) for k in dataset_schema} for example in list_data_dict
        ]
        sources = [
            prompt_input.format_map(example) if example["input"] != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [example["output"] + tokenizer.eos_token for example in list_data_dict]
        self.coordinator.print_on_master("Tokenizing inputs... This may take some time...")
        if isinstance(tokenizer, ChatGLMTokenizer):
            self.input_ids, self.labels, self.attention_mask = _preprocess_chatglm(
                sources, targets, tokenizer, max_length
            )
        else:
            self.input_ids, self.labels, self.attention_mask = _preprocess(sources, targets, tokenizer, max_length)

        self.coordinator.print_on_master("Loaded dataset.")

    def __len__(self):
        length = self.input_ids.shape[0]
        return length

    def __getitem__(self, idx):
        if self.attention_mask is not None:
            return dict(input_ids=self.input_ids[idx], labels=self.labels[idx], attention_mask=self.attention_mask[idx])
        else:
            return dict(input_ids=self.input_ids[idx], labels=self.labels[idx])
