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
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import torch
import torch.distributed as dist
import transformers
from torch.utils.data import Dataset
from tqdm import tqdm

from colossalai.logging import get_dist_logger

from .conversation import default_conversation
from .utils import is_rank_0, jload

# The following is a template prompt for a 4-round conversation.
"""
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.

Human: xxx</s>Assistant: xxx</s>Human: xxx</s>Assistant: xxx</s>Human: xxx</s>Assistant: xxx</s>Human: xxx</s>Assistant: xxx</s>
"""
# Please note that we only calculate loss on assistant's answer tokens.

logger = get_dist_logger()

IGNORE_INDEX = -100
DEFAULT_EOS_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input":
        ("Below is an instruction that describes a task, paired with an input that provides further context. "
         "Write a response that appropriately completes the request.\n\n"
         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"),
    "prompt_no_input": ("Below is an instruction that describes a task. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\n{instruction}\n\n### Response:"),
}


class SFTDataset(Dataset):
    """
    Dataset for sft model

    Args:
        dataset: dataset for supervised model
        tokenizer: tokenizer for supervised model
        max_length: max length of input
    """

    def __init__(self, dataset, tokenizer: Callable, max_length: int = 512) -> None:
        super().__init__()
        self.input_ids = []

        for data in tqdm(dataset, disable=not is_rank_0()):
            prompt = data['prompt'] + data['completion'] + tokenizer.eos_token
            prompt_token = tokenizer(prompt,
                                     max_length=max_length,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")

            self.input_ids.append(prompt_token['input_ids'][0])
        self.labels = copy.deepcopy(self.input_ids)

    def __len__(self):
        length = len(self.input_ids)
        return length

    def __getitem__(self, idx):
        return dict(input_ids=self.input_ids[idx], labels=self.labels[idx])


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int) -> Dict[str, torch.Tensor]:
    """Tokenize a list of strings."""
    tokenized_list = tokenizer(strings, return_tensors="pt", padding="longest", max_length=max_length, truncation=True)
    input_ids = labels = tokenized_list["input_ids"]
    input_ids_lens = labels_lens = \
        tokenized_list["input_ids"].ne(tokenizer.pad_token_id).sum(dim=-1)
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer, max_length) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def preprocess_conversation(sources: List[List[Dict]], tokenizer: transformers.PreTrainedTokenizer,
                            max_length: int) -> Dict:
    """Preprocess the conversation data by tokenizing."""
    conversations = []
    intermediates = []
    for source in sources:
        header = f"{default_conversation.system}"
        conversation, intermediate = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
        intermediates.append(intermediate)

    conversations_tokenized = _tokenize_fn(conversations, tokenizer, max_length)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)

    assert len(targets) == len(intermediates)
    for target, inters in zip(targets, intermediates):
        mask = torch.zeros_like(target, dtype=torch.bool)
        for inter in inters:
            tokenized = _tokenize_fn(inter, tokenizer, max_length)

            start_idx = tokenized["input_ids"][0].size(0) - 1
            end_idx = tokenized["input_ids"][1].size(0)

            mask[start_idx:end_idx] = True
        target[~mask] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def _add_speaker_and_signal(header: str,
                            source: List[Dict],
                            get_conversation: bool = True) -> Tuple[str, List[List[str]]]:
    END_SIGNAL = DEFAULT_EOS_TOKEN
    conversation = header
    intermediate = []
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = default_conversation.roles[1]
        else:
            from_str = 'unknown'

        value = from_str + ": " + sentence["value"] + END_SIGNAL
        if sentence["from"].lower() == "gpt":
            start = conversation + from_str + ": "
            end = conversation + value
            intermediate.append([start, end])
        if get_conversation:
            conversation += value
    return conversation, intermediate


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_datasets_size: int = None,
                 max_length: int = 512):
        super(SupervisedDataset, self).__init__()
        logger.info("Loading data...")
        list_data_dict = jload(data_path)
        logger.info(f"Loaded {len(list_data_dict)} examples.")

        if max_datasets_size is not None:
            logger.info(f"Limiting dataset to {max_datasets_size} examples.")
            list_data_dict = list_data_dict[:max_datasets_size]

        logger.info("Formatting inputs...")
        if "conversations" not in list_data_dict[0]:
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
            sources = [
                prompt_input.format_map(example)
                if example.get("input", "") != "" else prompt_no_input.format_map(example) for example in list_data_dict
            ]
            targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

            if is_rank_0():
                logger.info("Tokenizing inputs... This may take some time...")

            data_dict = preprocess(sources, targets, tokenizer, max_length)
        else:
            if is_rank_0():
                logger.info("Tokenizing inputs... This may take some time...")

            sources = [conv["conversations"] for conv in list_data_dict]
            data_dict = preprocess_conversation(sources, tokenizer, max_length)

        if is_rank_0():
            logger.info("Tokenizing finish.")

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                                    batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
