#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Splicing multiple pre-tokenized sequence data points
"""

import bisect
import random
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

from colossal_llama2.utils.conversation import Conversation, default_conversation
from datasets import dataset_dict
from torch.utils.data import ConcatDataset, Dataset, IterableDataset
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

IGNORE_INDEX = -100

DSType = Union[Dataset, ConcatDataset, dataset_dict.Dataset]


def supervised_tokenize_pretrain(
    data_point: Dict[str, str], tokenizer: LlamaTokenizer, ignore_index: int = None, max_length: int = 4096
) -> Dict[str, Union[int, str, List[int]]]:
    """
    A tokenization function to tokenize an original pretraining data point as following:
        {"source": "", "target": "Beijing, the capital of the People's Republic of China, ...", "category": "geography"}
    """
    assert tokenizer.add_bos_token is False and tokenizer.add_eos_token is False, (
        "Initially set `tokenizer.add_bos_token` and `tokenizer.add_eos_token` to False, "
        "add <bos> and <eos> manually later"
    )
    if ignore_index is None:
        ignore_index = IGNORE_INDEX

    source_text = data_point["source"]  # `str`
    target_text = data_point["target"]  # `str`
    is_null_source = len(source_text) == 0

    source_text = tokenizer.bos_token + source_text
    target_text += tokenizer.eos_token
    sequence_text = source_text + target_text

    tokenized = tokenizer([source_text, sequence_text])["input_ids"]
    sequence_input_ids = tokenized[1]
    sequence_labels = deepcopy(sequence_input_ids)

    source_length = len(tokenized[0])
    if not is_null_source:
        sequence_labels[:source_length] = [ignore_index for _ in range(source_length)]

    # sequence truncation.
    if len(sequence_input_ids) > max_length:
        sequence_input_ids = sequence_input_ids[:max_length]
        sequence_labels = sequence_labels[:max_length]

    return dict(
        input_ids=sequence_input_ids,
        labels=sequence_labels,
        seq_length=len(sequence_input_ids),
        seq_category=data_point["category"],
    )


def supervised_tokenize_sft(
    data_point: Dict[str, str],
    tokenizer: LlamaTokenizer,
    conversation_template: Conversation = default_conversation,
    ignore_index: int = None,
    max_length: int = 4096,
) -> Dict[str, Union[int, str, List[int]]]:
    """
    A tokenization function to tokenize an original pretraining data point as following:
        {"messages": [{"from": "human", "content": "xxx"}, {"from": "assistant", "content": "xxx"}]}
    """
    assert tokenizer.add_bos_token is False and tokenizer.add_eos_token is False, (
        "Initially set `tokenizer.add_bos_token` and `tokenizer.add_eos_token` to False, "
        "add <bos> and <eos> manually later"
    )

    assert (
        tokenizer.bos_token == conversation_template.seps[0] and tokenizer.eos_token == conversation_template.seps[1]
    ), "`bos_token` and `eos_token` should be the same with `conversation_template.seps`."

    if ignore_index is None:
        ignore_index = IGNORE_INDEX

    messages = data_point["messages"]
    template = deepcopy(conversation_template)
    template.messages = []

    for mess in messages:
        from_str = mess["from"]
        if from_str.lower() == "human":
            from_str = template.roles[0]
        elif from_str.lower() == "assistant":
            from_str = template.roles[1]
        else:
            raise ValueError(f"Unsupported role {from_str.lower()}")

        template.append_message(from_str, mess["content"])

    if len(template.messages) % 2 != 0:
        template.messages = template.messages[0:-1]

    # `target_turn_index` is the number of turns which exceeds `max_length - 1` for the first time.
    turns = [i for i in range(1, len(messages) // 2 + 1)]
    target_turn_index = bisect.bisect_right(
        turns,
        max_length - 1,
        key=lambda x: len(tokenizer([template.get_prompt(2 * x)], add_special_tokens=False)["input_ids"][0]),
    )

    # The tokenized length for first turn already exceeds `max_length - 1`.
    if target_turn_index - 1 < 0:
        return dict(
            input_ids=None,
            labels=None,
            inputs_decode=None,
            labels_decode=None,
            seq_length=None,
            seq_category=None,
        )

    target_turn = turns[target_turn_index - 1]
    prompt = template.get_prompt(2 * target_turn)
    tokenized = tokenizer([prompt], add_special_tokens=False)["input_ids"][0]

    # Uncomment this to check whether `bisect_right` is right.
    # if 2 * target_turn < len(template.messages):
    #     length_to_next_turn = len(tokenizer([template.get_prompt(2*target_turn+2)], add_special_tokens=False)["input_ids"][0])
    #     assert length_to_next_turn > max_length - 1, print(f"The length of the prompt until the next turn after tokenization is {length_to_next_turn}, which is smaller than {max_length - 1}")

    template.messages = template.messages[0 : 2 * target_turn]

    starts = []
    ends = []
    gpt_bos = False if template.messages[0][0] == template.roles[0] else True
    gpt_eos = False if template.messages[0][0] == template.roles[0] else True

    for i, token_id in enumerate(tokenized):
        if token_id == tokenizer.bos_token_id:
            if gpt_bos:
                starts.append(i)
            gpt_bos = not gpt_bos
        elif token_id == tokenizer.eos_token_id:
            if gpt_eos:
                ends.append(i)
            gpt_eos = not gpt_eos

    if len(starts) != target_turn or len(ends) != target_turn:
        print(
            "Please check whether the tokenizer add additional `bos_token` and `eos_token`.\n\nOr the original message contains `bos_token` or `eos_token`."
        )
        return dict(
            input_ids=None,
            labels=None,
            inputs_decode=None,
            labels_decode=None,
            seq_length=None,
            seq_category=None,
        )

    tokenized = [tokenizer.bos_token_id] + tokenized
    labels = [ignore_index] * len(tokenized)
    for start, end in zip(starts, ends):
        labels[start + 1 : end + 2] = tokenized[start + 1 : end + 2]

    labels_decode = deepcopy(labels)
    for i, z in enumerate(labels_decode):
        if z == ignore_index:
            labels_decode[i] = tokenizer.unk_token_id

    # `inputs_decode` and `labels decode` can be used to check whether the tokenization method is true.
    return dict(
        input_ids=tokenized,
        labels=labels,
        inputs_decode=tokenizer.decode(tokenized),
        labels_decode=tokenizer.decode(labels_decode),
        seq_length=len(tokenized),
        seq_category=data_point["category"] if "category" in data_point else "None",
    )


def generate_loss_mask(template: Conversation, tokenizer: Any, context_len: int):
    target_turn = int(len(template.messages) / 2)
    prompt = template.get_prompt(2 * target_turn)
    tokenized = tokenizer([prompt], add_special_tokens=False)
    input_ids = tokenized["input_ids"][0]
    attention_mask = tokenized["attention_mask"][0]
    starts = []
    ends = []
    gpt_bos = False if template.messages[0][0] == template.roles[0] else True
    gpt_eos = False if template.messages[0][0] == template.roles[0] else True

    for i, token_id in enumerate(input_ids):
        if token_id == tokenizer.bos_token_id:
            if gpt_bos:
                starts.append(i)
            gpt_bos = not gpt_bos
        elif token_id == tokenizer.eos_token_id:
            if gpt_eos:
                ends.append(i)
            gpt_eos = not gpt_eos

    if len(starts) != target_turn or len(ends) != target_turn:
        print(
            "Please check whether the tokenizer add additional `bos_token` and `eos_token`.\n\nOr the original message contains `bos_token` or `eos_token`."
        )
        return dict(input_ids=None, attention_mask=None, loss_mask=None)

    input_ids = [tokenizer.bos_token_id] + input_ids
    attention_mask = [1] + attention_mask
    loss_mask = [0 for _ in range(len(input_ids))]
    starts = starts[context_len:]
    ends = ends[context_len:]
    for start, end in zip(starts, ends):
        for i in range(start + 1, end + 2):
            loss_mask[i] = 1 if attention_mask[i] else 0

    return {"input_ids": input_ids, "attention_mask": attention_mask, "loss_mask": loss_mask}


def tokenize_rlhf(
    data_point: Dict[str, str],
    tokenizer: LlamaTokenizer,
    conversation_template: Conversation = default_conversation,
    ignore_index: int = None,
    max_length: int = 4096,
) -> Dict[str, Union[int, str, List[int]]]:
    """
    A tokenization function to tokenize an original pretraining data point as following:
        {"context": [{"from": "human", "content": "xxx"}, {"from": "assistant", "content": "xxx"}],
        "chosen": {"from": "assistant", "content": "xxx"}, "rejected": {"from": "assistant", "content": "xxx"}}
    """
    assert tokenizer.add_bos_token is False and tokenizer.add_eos_token is False, (
        "Initially set `tokenizer.add_bos_token` and `tokenizer.add_eos_token` to False, "
        "add <bos> and <eos> manually later"
    )

    assert (
        tokenizer.bos_token == conversation_template.seps[0] and tokenizer.eos_token == conversation_template.seps[1]
    ), "`bos_token` and `eos_token` should be the same with `conversation_template.seps`."

    if ignore_index is None:
        ignore_index = IGNORE_INDEX

    context = data_point["context"]
    template = deepcopy(conversation_template)
    template.messages = []

    for mess in context:
        from_str = mess["from"]
        if from_str.lower() == "human":
            from_str = template.roles[0]
        elif from_str.lower() == "assistant":
            from_str = template.roles[1]
        else:
            raise ValueError(f"Unsupported role {from_str.lower()}")

        if len(template.messages) > 0 and from_str == template.messages[-1][0]:
            template.messages[-1][1] = str(template.messages[-1][1] + mess["content"])
        else:
            template.append_message(from_str, mess["content"])

    if len(template.messages) % 2 != 1:
        print("Please make sure leading context is started and ended with a line from human")
        print(template.messages)
        return dict(
            chosen_input_ids=None,
            chosen_attention_mask=None,
            chosen_loss_mask=None,
            rejected_input_ids=None,
            rejected_attention_mask=None,
            rejected_loss_mask=None,
        )
    round_of_context = int((len(template.messages) - 1) / 2)

    assert context[-1]["from"].lower() == "human", "The last message in context should be from human."
    chosen = deepcopy(template)
    rejected = deepcopy(template)

    for round in range(len(data_point["chosen"])):
        from_str = data_point["chosen"][round]["from"]
        if from_str.lower() == "human":
            from_str = template.roles[0]
        elif from_str.lower() == "assistant":
            from_str = template.roles[1]
        else:
            raise ValueError(f"Unsupported role {from_str.lower()}")
        chosen.append_message(from_str, data_point["chosen"][round]["content"])

    for round in range(len(data_point["rejected"])):
        from_str = data_point["rejected"][round]["from"]
        if from_str.lower() == "human":
            from_str = template.roles[0]
        elif from_str.lower() == "assistant":
            from_str = template.roles[1]
        else:
            raise ValueError(f"Unsupported role {from_str.lower()}")
        rejected.append_message(from_str, data_point["rejected"][round]["content"])

    (
        chosen_input_ids,
        chosen_attention_mask,
        chosen_loss_mask,
        rejected_input_ids,
        rejected_attention_mask,
        rejected_loss_mask,
    ) = (None, None, None, None, None, None)
    if (
        len(tokenizer([chosen.get_prompt(len(chosen.messages))], add_special_tokens=False)["input_ids"][0])
        <= max_length - 1
        and len(tokenizer([rejected.get_prompt(len(rejected.messages))], add_special_tokens=False)["input_ids"][0])
        <= max_length - 1
    ):
        chosen_data_packed = generate_loss_mask(chosen, tokenizer, round_of_context)
        (chosen_input_ids, chosen_attention_mask, chosen_loss_mask) = (
            chosen_data_packed["input_ids"],
            chosen_data_packed["attention_mask"],
            chosen_data_packed["loss_mask"],
        )

        rejected_data_packed = generate_loss_mask(rejected, tokenizer, round_of_context)
        (rejected_input_ids, rejected_attention_mask, rejected_loss_mask) = (
            rejected_data_packed["input_ids"],
            rejected_data_packed["attention_mask"],
            rejected_data_packed["loss_mask"],
        )

        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "chosen_loss_mask": chosen_loss_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "rejected_loss_mask": rejected_loss_mask,
        }
    else:
        return dict(
            chosen_input_ids=None,
            chosen_attention_mask=None,
            chosen_loss_mask=None,
            rejected_input_ids=None,
            rejected_attention_mask=None,
            rejected_loss_mask=None,
        )


class ClosedToConstantLengthSplicedDataset(IterableDataset):
    """
    Define an iterable dataset that returns a (close to) constant length data point spliced from multiple
    original independent (pre-tokenized) data points.
    """

    def __init__(
        self,
        dataset: DSType,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 4096,
        num_packed_sequences: int = 8,
        fetch_sequence_func: Callable[[Any], Tuple[List[int], List[int]]] = None,
        input_ids_field: str = "input_ids",
        labels_field: str = "labels",
        infinite: bool = False,
        shuffle: bool = True,
        error_strict: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length
        self.infinite = infinite
        self.max_buffer_size = max_length * num_packed_sequences  # e.g., 4096 * 16
        self.shuffle = shuffle

        # Callable[[Dict[str, Any]], Tuple[List[int], List[int]]],
        # A function that fetch sequence input_ids and labels from the original data point
        if fetch_sequence_func is None:
            self.fetch_sequence_func = lambda data_point: (data_point[input_ids_field], data_point[labels_field])
        else:
            self.fetch_sequence_func = fetch_sequence_func
        self.input_ids_field = input_ids_field
        self.labels_field = labels_field

        self.error_strict = error_strict
        self.current_size = 0  # `int`, current packed data size.

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterable[Dict[str, List[int]]]:
        iterator = iter(self.dataset)
        more_data_points = True
        while more_data_points is True:
            buffer, buffer_len = [], 0
            while True:
                # ending condition.
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    # `Tuple[List[int], List[int]]`
                    seq_input_ids, seq_labels = self.fetch_sequence_func(next(iterator))
                    buffer.append({self.input_ids_field: seq_input_ids, self.labels_field: seq_labels})
                    buffer_len += len(buffer[-1][self.input_ids_field])
                except StopIteration:
                    if self.infinite is True:
                        iterator = iter(self.dataset)
                        warnings.warn("The dataset reached end and the iterator is reset to the start.")
                    else:
                        more_data_points = False
                        break
            examples = []  # `List[Dict[str, List[int]]]`, save buffered spliced data points.
            spliced_input_ids, spliced_labels = [], []  # `List[int]`, `List[int]`
            for i, data_point in enumerate(buffer):
                # TODO(2023-09-18) check errors for each unspliced tokenized data point
                seq_input_ids = data_point[self.input_ids_field]
                seq_labels = data_point[self.labels_field]
                # Handle special case:
                # If the length of an original data point (i.e., input_ids length of a data point before splicing)
                # exceeds `max_length`, truncate it.
                if len(seq_input_ids) > self.max_length:
                    truncated_seq_input_ids = seq_input_ids[: self.max_length]
                    truncated_label_ids = seq_labels[: self.max_length]
                    if set(truncated_label_ids) == {IGNORE_INDEX}:
                        if self.error_strict is True:
                            raise ValueError(
                                f"Find an out-of-bounds length({len(seq_input_ids)}) data point "
                                f"with all label values as {IGNORE_INDEX}."
                            )
                        else:
                            warnings.warn(f"Filter an error truncated data point (labels all {IGNORE_INDEX})")
                            continue  # Skip the current error data point.
                    spliced_data_point = {
                        self.input_ids_field: truncated_seq_input_ids,
                        self.labels_field: truncated_label_ids,
                    }
                    examples.append(spliced_data_point)
                    warnings.warn("Find a data point to be truncated.")
                    continue

                # Pre action judgment.
                if len(spliced_input_ids) + len(seq_input_ids) > self.max_length:
                    spliced_data_point = {
                        self.input_ids_field: spliced_input_ids,
                        self.labels_field: spliced_labels,
                    }  # `Dict[str, List[int]]`
                    # Update.
                    spliced_input_ids, spliced_labels = [], []
                    spliced_input_ids.extend(seq_input_ids)
                    spliced_labels.extend(seq_labels)
                    examples.append(spliced_data_point)
                else:
                    spliced_input_ids.extend(seq_input_ids)
                    spliced_labels.extend(seq_labels)
            # For residual spliced data point at the end of the data set
            if self.infinite is False and more_data_points is False and len(spliced_input_ids) > 0:
                examples.append({self.input_ids_field: spliced_input_ids, self.labels_field: spliced_labels})
            if self.shuffle:
                random.shuffle(examples)
            for spliced_data_point in examples:
                # TODO(2023-09-18): check errors for each spliced tokenized data point.
                self.current_size += 1
                yield spliced_data_point
