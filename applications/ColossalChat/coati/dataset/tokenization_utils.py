#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tokenization utils for constructing dataset for ppo, dpo, sft, rm
"""

import warnings
from copy import deepcopy
from typing import Any, Dict, List, Union

from coati.dataset.conversation import Conversation
from coati.dataset.utils import split_templated_prompt_into_chunks, tokenize_and_concatenate
from datasets import dataset_dict
from torch.utils.data import ConcatDataset, Dataset
from transformers import PreTrainedTokenizer

from colossalai.logging import get_dist_logger

logger = get_dist_logger()

IGNORE_INDEX = -100

DSType = Union[Dataset, ConcatDataset, dataset_dict.Dataset]


def supervised_tokenize_sft(
    data_point: Dict[str, str],
    tokenizer: PreTrainedTokenizer,
    conversation_template: Conversation = None,
    ignore_index: int = None,
    max_length: int = 4096,
) -> Dict[str, Union[int, str, List[int]]]:
    """
    A tokenization function to tokenize an original pretraining data point as following
         and calculate corresponding labels for sft training:
        "Something here can be system message[user_line_start]User line[User line end][Assistant line start]Assistant line[Assistant line end]...[Assistant line end]Something here"
                                            ^
                                end_of_system_line_position

    Args:
        data_point: the data point of the following format
            {"messages": [{"from": "human", "content": "xxx"}, {"from": "assistant", "content": "xxx"}]}
        tokenizer: the tokenizer whose
        conversation_template: the conversation template to apply
        ignore_index: the ignore index when calculate loss during training
        max_length: the maximum context length
    """

    if ignore_index is None:
        ignore_index = IGNORE_INDEX

    messages = data_point["messages"]
    template = deepcopy(conversation_template)
    template.messages = []

    for mess in messages:
        from_str = mess["from"]
        if from_str.lower() == "human":
            from_str = "user"
        elif from_str.lower() == "assistant":
            from_str = "assistant"
        else:
            raise ValueError(f"Unsupported role {from_str.lower()}")

        template.append_message(from_str, mess["content"])

    if len(template.messages) % 2 != 0:
        template.messages = template.messages[0:-1]

    # `target_turn_index` is the number of turns which exceeds `max_length - 1` for the first time.
    turns = [i for i in range(1, len(messages) // 2 + 1)]

    lo, hi = 0, len(turns)
    while lo < hi:
        mid = (lo + hi) // 2
        prompt = template.get_prompt(2 * turns[mid] - 1)
        chunks, require_loss = split_templated_prompt_into_chunks(
            template.messages[: 2 * turns[mid] - 1], prompt, conversation_template.end_of_assistant
        )
        tokenized, starts, ends = tokenize_and_concatenate(tokenizer, chunks, require_loss)
        if max_length - 1 < len(tokenized):
            hi = mid
        else:
            lo = mid + 1
    target_turn_index = lo

    # The tokenized length for first turn already exceeds `max_length - 1`.
    if target_turn_index - 1 < 0:
        warnings.warn("The tokenized length for first turn already exceeds `max_length - 1`.")
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
    chunks, require_loss = split_templated_prompt_into_chunks(
        template.messages[: 2 * target_turn], prompt, conversation_template.end_of_assistant
    )
    tokenized, starts, ends = tokenize_and_concatenate(tokenizer, chunks, require_loss)

    labels = [ignore_index] * len(tokenized)
    for start, end in zip(starts, ends):
        if end == len(tokenized):
            tokenized = tokenized + [tokenizer.eos_token_id]
            labels = labels + [ignore_index]
        labels[start:end] = tokenized[start:end]

    # truncate the sequence at the last token that requires loss calculation
    to_truncate_len = 0
    for i in range(len(tokenized) - 1, -1, -1):
        if labels[i] == ignore_index:
            to_truncate_len += 1
        else:
            break
    to_truncate_len = max(len(tokenized) - max_length, to_truncate_len)
    tokenized = tokenized[: len(tokenized) - to_truncate_len]
    labels = labels[: len(labels) - to_truncate_len]

    if tokenizer.bos_token_id is not None:
        if tokenized[0] != tokenizer.bos_token_id:
            tokenized = [tokenizer.bos_token_id] + tokenized
            labels = [ignore_index] + labels

    if tokenizer.eos_token_id is not None:
        # Force to add eos token at the end of the tokenized sequence
        if tokenized[-1] != tokenizer.eos_token_id:
            tokenized = tokenized + [tokenizer.eos_token_id]
            labels = labels + [tokenizer.eos_token_id]
        else:
            labels[-1] = tokenizer.eos_token_id

    # For some model without bos/eos may raise the following errors
    inputs_decode = tokenizer.decode(tokenized)
    start = 0
    end = 0
    label_decode = []
    for i in range(len(labels)):
        if labels[i] == ignore_index:
            if start != end:
                label_decode.append(tokenizer.decode(labels[start + 1 : i], skip_special_tokens=False))
            start = i
            end = i
        else:
            end = i
            if i == len(labels) - 1:
                label_decode.append(tokenizer.decode(labels[start + 1 :], skip_special_tokens=False))

    # Check if all labels are ignored, this may happen when the tokenized length is too long
    if labels.count(ignore_index) == len(labels):
        return dict(
            input_ids=None,
            labels=None,
            inputs_decode=None,
            labels_decode=None,
            seq_length=None,
            seq_category=None,
        )

    return dict(
        input_ids=tokenized,
        labels=labels,
        inputs_decode=inputs_decode,
        labels_decode=label_decode,
        seq_length=len(tokenized),
        seq_category=data_point["category"] if "category" in data_point else "None",
    )


def tokenize_prompt_dataset(
    data_point: Dict[str, str],
    tokenizer: PreTrainedTokenizer,
    conversation_template: Conversation = None,
    ignore_index: int = None,
    max_length: int = 4096,
) -> Dict[str, Union[int, str, List[int]]]:
    """
    A tokenization function to tokenize an original pretraining data point as following for ppo training:
        "Something here can be system message[user_line_start]User line[User line end][Assistant line start]Assistant line[Assistant line end]...[Assistant line start]"
    Args:
        data_point: the data point of the following format
            {"messages": [{"from": "human", "content": "xxx"}, {"from": "assistant", "content": "xxx"}]}
        tokenizer: the tokenizer whose
        conversation_template: the conversation template to apply
        ignore_index: the ignore index when calculate loss during training
        max_length: the maximum context length
    """
    if ignore_index is None:
        ignore_index = IGNORE_INDEX

    messages = data_point["messages"]
    template = deepcopy(conversation_template)
    template.messages = []

    for mess in messages:
        from_str = mess["from"]
        if from_str.lower() == "human":
            from_str = "user"
        elif from_str.lower() == "assistant":
            from_str = "assistant"
        else:
            raise ValueError(f"Unsupported role {from_str.lower()}")

        template.append_message(from_str, mess["content"])

    # `target_turn_index` is the number of turns which exceeds `max_length - 1` for the first time.
    target_turn = len(template.messages)
    if target_turn % 2 != 1:
        # exclude the answer if provided. keep only the prompt
        target_turn = target_turn - 1

    # Prepare data
    prompt = template.get_prompt(target_turn, add_generation_prompt=True)
    chunks, require_loss = split_templated_prompt_into_chunks(
        template.messages[:target_turn], prompt, conversation_template.end_of_assistant
    )
    tokenized, starts, ends = tokenize_and_concatenate(tokenizer, chunks, require_loss)
    if tokenizer.bos_token_id is not None:
        if tokenized[0] != tokenizer.bos_token_id:
            tokenized = [tokenizer.bos_token_id] + tokenized

    # Skip overlength data
    if max_length - 1 < len(tokenized):
        return dict(
            input_ids=None,
            inputs_decode=None,
            seq_length=None,
            seq_category=None,
        )

    # `inputs_decode` can be used to check whether the tokenization method is true.
    return dict(
        input_ids=tokenized,
        inputs_decode=tokenizer.decode(tokenized),
        seq_length=len(tokenized),
        seq_category=data_point["category"] if "category" in data_point else "None",
    )


def apply_rlhf_data_format(
    template: Conversation, tokenizer: Any, context_len: int, mask_out_target_assistant_line_end=False
):
    target_turn = int(len(template.messages) / 2)
    prompt = template.get_prompt(target_turn * 2)
    chunks, require_loss = split_templated_prompt_into_chunks(
        template.messages[: 2 * target_turn], prompt, template.end_of_assistant
    )
    tokenized, starts, ends = tokenize_and_concatenate(tokenizer, chunks, require_loss)
    loss_mask = [0] * len(tokenized)
    mask_token = tokenizer.eos_token_id or tokenizer.pad_token_id
    if mask_token is None:
        mask_token = 1  # If the tokenizer doesn't have eos_token or pad_token: Qwen

    label_decode = []
    for start, end in zip(starts[-1:], ends[-1:]):
        # only the last round (chosen/rejected) counts
        if end == len(tokenized):
            tokenized = tokenized + [tokenizer.eos_token_id]
            loss_mask = loss_mask + [1]
        loss_mask[start:end] = [1] * len(loss_mask[start:end])
        label_decode.append(tokenizer.decode(tokenized[start:end], skip_special_tokens=False))
    if tokenizer.bos_token_id is not None:
        if tokenized[0] != tokenizer.bos_token_id:
            tokenized = [tokenizer.bos_token_id] + tokenized
            loss_mask = [0] + loss_mask

    if tokenizer.eos_token_id is not None:
        # Force to add eos token at the end of the tokenized sequence
        if tokenized[-1] != tokenizer.eos_token_id:
            tokenized = tokenized + [tokenizer.eos_token_id]
            loss_mask = loss_mask + [1]
        else:
            loss_mask[-1] = 1

    return {"input_ids": tokenized, "loss_mask": loss_mask, "label_decode": label_decode}


def tokenize_rlhf(
    data_point: Dict[str, str],
    tokenizer: PreTrainedTokenizer,
    conversation_template: Conversation = None,
    ignore_index: int = None,
    max_length: int = 4096,
) -> Dict[str, Union[int, str, List[int]]]:
    """
    A tokenization function to tokenize an original pretraining data point as following:
        {"context": [{"from": "human", "content": "xxx"}, {"from": "assistant", "content": "xxx"}],
        "chosen": {"from": "assistant", "content": "xxx"}, "rejected": {"from": "assistant", "content": "xxx"}}
    """
    if ignore_index is None:
        ignore_index = IGNORE_INDEX

    context = data_point["context"]
    template = deepcopy(conversation_template)
    template.clear()

    for mess in context:
        from_str = mess["from"]
        if from_str.lower() == "human":
            from_str = "user"
        elif from_str.lower() == "assistant":
            from_str = "assistant"
        else:
            raise ValueError(f"Unsupported role {from_str.lower()}")

        if len(template.messages) > 0 and from_str == template.messages[-1]["role"]:
            # Concate adjacent message from the same role
            template.messages[-1]["content"] = str(template.messages[-1]["content"] + " " + mess["content"])
        else:
            template.append_message(from_str, mess["content"])

    if len(template.messages) % 2 != 1:
        warnings.warn(
            "Please make sure leading context starts and ends with a line from human\nLeading context: "
            + str(template.messages)
        )
        return dict(
            chosen_input_ids=None,
            chosen_loss_mask=None,
            chosen_label_decode=None,
            rejected_input_ids=None,
            rejected_loss_mask=None,
            rejected_label_decode=None,
        )
    round_of_context = int((len(template.messages) - 1) / 2)

    assert context[-1]["from"].lower() == "human", "The last message in context should be from human."
    chosen = deepcopy(template)
    rejected = deepcopy(template)

    for round in range(len(data_point["chosen"])):
        from_str = data_point["chosen"][round]["from"]
        if from_str.lower() == "human":
            from_str = "user"
        elif from_str.lower() == "assistant":
            from_str = "assistant"
        else:
            raise ValueError(f"Unsupported role {from_str.lower()}")
        chosen.append_message(from_str, data_point["chosen"][round]["content"])

    for round in range(len(data_point["rejected"])):
        from_str = data_point["rejected"][round]["from"]
        if from_str.lower() == "human":
            from_str = "user"
        elif from_str.lower() == "assistant":
            from_str = "assistant"
        else:
            raise ValueError(f"Unsupported role {from_str.lower()}")
        rejected.append_message(from_str, data_point["rejected"][round]["content"])

    (
        chosen_input_ids,
        chosen_loss_mask,
        chosen_label_decode,
        rejected_input_ids,
        rejected_loss_mask,
        rejected_label_decode,
    ) = (None, None, None, None, None, None)

    chosen_data_packed = apply_rlhf_data_format(chosen, tokenizer, round_of_context)
    (chosen_input_ids, chosen_loss_mask, chosen_label_decode) = (
        chosen_data_packed["input_ids"],
        chosen_data_packed["loss_mask"],
        chosen_data_packed["label_decode"],
    )

    rejected_data_packed = apply_rlhf_data_format(
        rejected, tokenizer, round_of_context, mask_out_target_assistant_line_end=True
    )
    (rejected_input_ids, rejected_loss_mask, rejected_label_decode) = (
        rejected_data_packed["input_ids"],
        rejected_data_packed["loss_mask"],
        rejected_data_packed["label_decode"],
    )

    if len(chosen_input_ids) > max_length or len(rejected_input_ids) > max_length:
        return dict(
            chosen_input_ids=None,
            chosen_loss_mask=None,
            chosen_label_decode=None,
            rejected_input_ids=None,
            rejected_loss_mask=None,
            rejected_label_decode=None,
        )
    # Check if loss mask is all 0s (no loss), this may happen when the tokenized length is too long
    if chosen_loss_mask[1:].count(1) == 0 or rejected_loss_mask[1:].count(1) == 0:
        return dict(
            chosen_input_ids=None,
            chosen_loss_mask=None,
            chosen_label_decode=None,
            rejected_input_ids=None,
            rejected_loss_mask=None,
            rejected_label_decode=None,
        )

    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_loss_mask": chosen_loss_mask,
        "chosen_label_decode": chosen_label_decode,
        "rejected_input_ids": rejected_input_ids,
        "rejected_loss_mask": rejected_loss_mask,
        "rejected_label_decode": rejected_label_decode,
    }


def tokenize_kto(
    data_point: Dict[str, str],
    tokenizer: PreTrainedTokenizer,
    conversation_template: Conversation = None,
    ignore_index: int = None,
    max_length: int = 4096,
) -> Dict[str, Union[int, str, List[int]]]:
    """
    Tokenize a dataset for KTO training
    The raw input data is conversation that have the following format
    {
        "prompt": [{"from": "human", "content": "xxx"}...],
        "completion": {"from": "assistant", "content": "xxx"},
        "label": true/false
    }
    It returns three fields
    The context, which contain the query and the assistant start,
    the completion, which only contains the assistance's answer,
    and a binary label, which indicates if the sample is prefered or not
    """
    if ignore_index is None:
        ignore_index = IGNORE_INDEX

    prompt = data_point["prompt"]
    completion = data_point["completion"]
    template = deepcopy(conversation_template)
    template.clear()

    if prompt[0].get("from", None) != "human":
        raise ValueError("conversation should start with human")
    if completion.get("from", None) != "assistant":
        raise ValueError("conversation should end with assistant")

    for mess in prompt:
        if mess.get("from", None) == "human":
            template.append_message("user", mess["content"])
        elif mess.get("from", None) == "assistant":
            template.append_message("assistant", mess["content"])
        else:
            raise ValueError(f"Unsupported role {mess.get('from', None)}")
    generation_prompt = template.get_prompt(len(prompt), add_generation_prompt=True)
    template.append_message("assistant", completion["content"])
    full_prompt = template.get_prompt(len(prompt) + 1, add_generation_prompt=False)
    tokenized_full_prompt = tokenizer(full_prompt, add_special_tokens=False)["input_ids"]
    if len(tokenized_full_prompt) + 1 > max_length:
        return dict(prompt=None, completion=None, label=None, input_id_decode=None, completion_decode=None)
    tokenized_generation_prompt = tokenizer(generation_prompt, add_special_tokens=False)["input_ids"]
    tokenized_completion = tokenized_full_prompt[len(tokenized_generation_prompt) :]
    tokenized_completion = deepcopy(tokenized_completion)
    if tokenizer.bos_token_id is not None and tokenized_generation_prompt[0] != tokenizer.bos_token_id:
        tokenized_generation_prompt = [tokenizer.bos_token_id] + tokenized_generation_prompt
    decoded_full_prompt = tokenizer.decode(tokenized_full_prompt, skip_special_tokens=False)
    decoded_completion = tokenizer.decode(tokenized_completion, skip_special_tokens=False)

    return {
        "prompt": tokenized_generation_prompt,
        "completion": tokenized_completion,
        "label": data_point["label"],
        "input_id_decode": decoded_full_prompt,
        "completion_decode": decoded_completion,
    }
