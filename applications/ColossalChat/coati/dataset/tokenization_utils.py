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


def tokenize_sft(
    data_point: Dict[str, str],
    tokenizer: PreTrainedTokenizer,
    conversation_template: Conversation = None,
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
            {"messages": [{"from": "user", "content": "xxx"}, {"from": "assistant", "content": "xxx"}]}
        tokenizer: the tokenizer whose
        conversation_template: the conversation template to apply
        ignore_index: the ignore index when calculate loss during training
        max_length: the maximum context length
    """

    ignore_index = IGNORE_INDEX

    messages = data_point["messages"]
    template = deepcopy(conversation_template)

    if messages[0]["from"] == "system":
        template.system_message = str(messages[0]["content"])
        messages.pop(0)
    template.messages = []
    for idx, mess in enumerate(messages):
        if mess["from"] != template.roles[idx % 2]:
            raise ValueError(
                f"Message should iterate between user and assistant and starts with a \
                             line from the user. Got the following data:\n{messages}"
            )
        template.append_message(mess["from"], mess["content"])

    if len(template.messages) % 2 != 0:
        # Force to end with assistant response
        template.messages = template.messages[0:-1]

    # tokenize and calculate masked labels -100 for positions corresponding to non-assistant lines
    prompt = template.get_prompt()
    chunks, require_loss = split_templated_prompt_into_chunks(
        template.messages, prompt, conversation_template.end_of_assistant
    )
    tokenized, starts, ends = tokenize_and_concatenate(tokenizer, chunks, require_loss, max_length=max_length)
    if tokenized is None:
        return dict(
            input_ids=None,
            labels=None,
            inputs_decode=None,
            labels_decode=None,
            seq_length=None,
            seq_category=None,
        )

    labels = [ignore_index] * len(tokenized)
    for start, end in zip(starts, ends):
        labels[start:end] = tokenized[start:end]

    if tokenizer.bos_token_id is not None:
        # Force to add bos token at the beginning of the tokenized sequence if the input ids doesn;t starts with bos
        if tokenized[0] != tokenizer.bos_token_id:
            # Some chat templates already include bos token
            tokenized = [tokenizer.bos_token_id] + tokenized
            labels = [-100] + labels

    # log decoded inputs and labels for debugging
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


def tokenize_prompt(
    data_point: Dict[str, str],
    tokenizer: PreTrainedTokenizer,
    conversation_template: Conversation = None,
    max_length: int = 4096,
) -> Dict[str, Union[int, str, List[int]]]:
    """
    A tokenization function to tokenize an original pretraining data point as following for ppo training:
        "Something here can be system message[user_line_start]User line[User line end][Assistant line start]Assistant line[Assistant line end]...[Assistant line start]"
    Args:
        data_point: the data point of the following format
            {"messages": [{"from": "user", "content": "xxx"}, {"from": "assistant", "content": "xxx"}]}
        tokenizer: the tokenizer whose
        conversation_template: the conversation template to apply
        ignore_index: the ignore index when calculate loss during training
        max_length: the maximum context length
    """

    messages = data_point["messages"]
    template = deepcopy(conversation_template)
    template.messages = []

    if messages[0]["from"] == "system":
        template.system_message = str(messages[0]["content"])
        messages.pop(0)

    for idx, mess in enumerate(messages):
        if mess["from"] != template.roles[idx % 2]:
            raise ValueError(
                f"Message should iterate between user and assistant and starts with a line from the user. Got the following data:\n{messages}"
            )
        template.append_message(mess["from"], mess["content"])

    # `target_turn_index` is the number of turns which exceeds `max_length - 1` for the first time.
    if len(template.messages) % 2 != 1:
        # exclude the answer if provided. keep only the prompt
        template.messages = template.messages[:-1]

    # Prepare data
    prompt = template.get_prompt(length=len(template.messages), add_generation_prompt=True)
    tokenized = tokenizer([prompt], add_special_tokens=False)["input_ids"][0]

    if tokenizer.bos_token_id is not None:
        if tokenized[0] != tokenizer.bos_token_id:
            tokenized = [tokenizer.bos_token_id] + tokenized

    if len(tokenized) > max_length:
        return dict(
            input_ids=None,
            inputs_decode=None,
            seq_length=None,
            seq_category=None,
        )

    # `inputs_decode` can be used to check whether the tokenization method is true.
    return dict(
        input_ids=tokenized,
        inputs_decode=prompt,
        seq_length=len(tokenized),
        seq_category=data_point["category"] if "category" in data_point else "None",
    )


def apply_rlhf_data_format(template: Conversation, tokenizer: Any):
    target_turn = int(len(template.messages) / 2)
    prompt = template.get_prompt(target_turn * 2)
    chunks, require_loss = split_templated_prompt_into_chunks(
        template.messages[: 2 * target_turn], prompt, template.end_of_assistant
    )
    # no truncation applied
    tokenized, starts, ends = tokenize_and_concatenate(tokenizer, chunks, require_loss, max_length=None)

    loss_mask = [0] * len(tokenized)
    label_decode = []
    # only the last round (chosen/rejected) is used to calculate loss
    for i in range(starts[-1], ends[-1]):
        loss_mask[i] = 1
    label_decode.append(tokenizer.decode(tokenized[starts[-1] : ends[-1]], skip_special_tokens=False))
    if tokenizer.bos_token_id is not None:
        if tokenized[0] != tokenizer.bos_token_id:
            tokenized = [tokenizer.bos_token_id] + tokenized
            loss_mask = [0] + loss_mask

    return {"input_ids": tokenized, "loss_mask": loss_mask, "label_decode": label_decode}


def tokenize_rlhf(
    data_point: Dict[str, str],
    tokenizer: PreTrainedTokenizer,
    conversation_template: Conversation = None,
    max_length: int = 4096,
) -> Dict[str, Union[int, str, List[int]]]:
    """
    A tokenization function to tokenize an original pretraining data point as following:
        {"context": [{"from": "user", "content": "xxx"}, {"from": "assistant", "content": "xxx"}],
        "chosen": {"from": "assistant", "content": "xxx"}, "rejected": {"from": "assistant", "content": "xxx"}}
    """

    context = data_point["context"]
    template = deepcopy(conversation_template)
    template.clear()

    if context[0]["from"] == "system":
        template.system_message = str(context[0]["content"])
        context.pop(0)

    for idx, mess in enumerate(context):
        if mess["from"] != template.roles[idx % 2]:
            raise ValueError(
                f"Message should iterate between user and assistant and starts with a \
                             line from the user. Got the following data:\n{context}"
            )
        template.append_message(mess["from"], mess["content"])

    if len(template.messages) % 2 != 1:
        warnings.warn(
            "Please make sure leading context starts and ends with a line from user\nLeading context: "
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

    assert context[-1]["from"].lower() == template.roles[0], "The last message in context should be from user."
    chosen = deepcopy(template)
    rejected = deepcopy(template)
    chosen_continuation = data_point["chosen"]
    rejected_continuation = data_point["rejected"]
    for round in range(len(chosen_continuation)):
        if chosen_continuation[round]["from"] != template.roles[(round + 1) % 2]:
            raise ValueError(
                f"Message should iterate between user and assistant and starts with a \
                             line from the user. Got the following data:\n{chosen_continuation}"
            )
        chosen.append_message(chosen_continuation[round]["from"], chosen_continuation[round]["content"])

    for round in range(len(rejected_continuation)):
        if rejected_continuation[round]["from"] != template.roles[(round + 1) % 2]:
            raise ValueError(
                f"Message should iterate between user and assistant and starts with a \
                             line from the user. Got the following data:\n{rejected_continuation}"
            )
        rejected.append_message(rejected_continuation[round]["from"], rejected_continuation[round]["content"])

    (
        chosen_input_ids,
        chosen_loss_mask,
        chosen_label_decode,
        rejected_input_ids,
        rejected_loss_mask,
        rejected_label_decode,
    ) = (None, None, None, None, None, None)

    chosen_data_packed = apply_rlhf_data_format(chosen, tokenizer)
    (chosen_input_ids, chosen_loss_mask, chosen_label_decode) = (
        chosen_data_packed["input_ids"],
        chosen_data_packed["loss_mask"],
        chosen_data_packed["label_decode"],
    )

    rejected_data_packed = apply_rlhf_data_format(rejected, tokenizer)
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
    if chosen_loss_mask.count(1) == 0 or rejected_loss_mask.count(1) == 0:
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
    max_length: int = 4096,
) -> Dict[str, Union[int, str, List[int]]]:
    """
    Tokenize a dataset for KTO training
    The raw input data is conversation that have the following format
    {
        "prompt": [{"from": "user", "content": "xxx"}...],
        "completion": {"from": "assistant", "content": "xxx"},
        "label": true/false
    }
    It returns three fields
    The context, which contain the query and the assistant start,
    the completion, which only contains the assistance's answer,
    and a binary label, which indicates if the sample is prefered or not
    """
    prompt = data_point["prompt"]
    completion = data_point["completion"]
    template = deepcopy(conversation_template)
    template.clear()

    if prompt[0]["from"] == "system":
        template.system_message = str(prompt[0]["content"])
        prompt.pop(0)

    if prompt[0].get("from", None) != "user":
        raise ValueError("conversation should start with user")
    if completion.get("from", None) != "assistant":
        raise ValueError("conversation should end with assistant")

    for mess in prompt:
        if mess.get("from", None) == "user":
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
