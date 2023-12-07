#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tokenization utils for constructing dataset for ppo, dpo, sft, rm
"""

import warnings
from copy import deepcopy
from typing import Any, Dict, List, Union

from coati.dataset.conversation import Conversation, default_conversation
from datasets import dataset_dict
from torch.utils.data import ConcatDataset, Dataset
from transformers import PreTrainedTokenizer

from colossalai.logging import get_dist_logger

logger = get_dist_logger()

IGNORE_INDEX = -100

DSType = Union[Dataset, ConcatDataset, dataset_dict.Dataset]


def supervised_tokenize_pretrain(
    data_point: Dict[str, str], tokenizer: PreTrainedTokenizer, ignore_index: int = None, max_length: int = 4096
) -> Dict[str, Union[int, str, List[int]]]:
    """
    A tokenization function to tokenize an original pretraining data point as following:
        {"source": "", "target": "Beijing, the capital of the People's Republic of China, ...", "category": "geography"}
    """
    # assert tokenizer.add_bos_token is False and tokenizer.add_eos_token is False, (
    #     "Initially set `tokenizer.add_bos_token` and `tokenizer.add_eos_token` to False, "
    #     "add <bos> and <eos> manually later"
    # )
    if ignore_index is None:
        ignore_index = IGNORE_INDEX

    source_text = data_point["source"]  # `str`
    target_text = data_point["target"]  # `str`
    is_null_source = len(source_text) == 0

    source_text = tokenizer.bos_token + source_text
    target_text += " " + tokenizer.eos_token
    sequence_text = source_text + target_text

    tokenized = tokenizer([source_text, sequence_text], add_special_tokens=False)["input_ids"]
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
        seq_category=data_point["category"] if "category" in data_point else "None",
    )


def supervised_tokenize_sft(
    data_point: Dict[str, str],
    tokenizer: PreTrainedTokenizer,
    conversation_template: Conversation = default_conversation,
    ignore_index: int = None,
    max_length: int = 4096,
) -> Dict[str, Union[int, str, List[int]]]:
    """
    A tokenization function to tokenize an original pretraining data point as following:
        {"messages": [{"from": "human", "content": "xxx"}, {"from": "assistant", "content": "xxx"}]}
    """
    # assert tokenizer.add_bos_token is False and tokenizer.add_eos_token is False, (
    #     "Initially set `tokenizer.add_bos_token` and `tokenizer.add_eos_token` to False, "
    #     "add <bos> and <eos> manually later"
    # )

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

    lo, hi = 0, len(turns)
    while lo < hi:
        mid = (lo + hi) // 2
        if max_length - 1 < len(
            tokenizer([template.get_prompt(2 * turns[mid] - 1)], add_special_tokens=False)["input_ids"][0]
        ):
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
    tokenized = tokenizer([prompt], add_special_tokens=False)["input_ids"][0]
    template.messages = template.messages[0 : 2 * target_turn]

    starts = []
    ends = []
    expect_bos = True
    gpt_bos = False if template.messages[0][0] == template.roles[0] else True
    gpt_eos = False if template.messages[0][0] == template.roles[0] else True

    for i, token_id in enumerate(tokenized):
        if token_id == tokenizer.bos_token_id and expect_bos:
            if gpt_bos:
                starts.append(i)
            gpt_bos = not gpt_bos
            expect_bos = not expect_bos
            continue
        if token_id == tokenizer.eos_token_id and not expect_bos:
            if gpt_eos:
                ends.append(i)
            gpt_eos = not gpt_eos
            expect_bos = not expect_bos

    if len(starts) != target_turn or len(ends) != target_turn:
        logger.info(
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
            labels_decode[i] = tokenizer.eos_token_id

    # `inputs_decode` and `labels_decode` can be used to check whether the tokenization method is true.
    return dict(
        input_ids=tokenized,
        labels=labels,
        inputs_decode=tokenizer.decode(tokenized),
        labels_decode=tokenizer.decode(labels_decode),
        seq_length=len(tokenized),
        seq_category=data_point["category"] if "category" in data_point else "None",
    )


def tokenize_prompt_dataset(
    data_point: Dict[str, str],
    tokenizer: PreTrainedTokenizer,
    conversation_template: Conversation = default_conversation,
    ignore_index: int = None,
    max_length: int = 4096,
) -> Dict[str, Union[int, str, List[int]]]:
    """
    A tokenization function to tokenize an original pretraining data point as following:
        {"messages": [{"from": "human", "content": "xxx"}, {"from": "assistant", "content": "xxx"}]}
    """

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

    if len(template.messages) % 2 != 1:
        # exclude the answer if provided. keep only the prompt
        template.messages = template.messages[0:-1]

    # `target_turn_index` is the number of turns which exceeds `max_length - 1` for the first time.
    turns = [i for i in range(1, (len(messages) + 1) // 2 + 1)]

    lo, hi = 0, len(turns)
    while lo < hi:
        mid = (lo + hi) // 2
        if max_length - 1 < len(
            tokenizer([template.get_prompt(2 * turns[mid] - 1)], add_special_tokens=False)["input_ids"][0]
        ):
            hi = mid
        else:
            lo = mid + 1
    target_turn_index = lo

    # The tokenized length for first turn already exceeds `max_length - 1`.
    if target_turn_index - 1 < 0:
        warnings.warn("The tokenized length for first turn already exceeds `max_length - 1`.")
        return dict(
            input_ids=None,
            inputs_decode=None,
            seq_length=None,
            seq_category=None,
        )

    target_turn = turns[target_turn_index - 1]
    prompt = template.get_prompt(2 * target_turn - 1) + "Assistant: <s>"
    tokenized = tokenizer([prompt], add_special_tokens=False)["input_ids"][0]
    template.messages = template.messages[0 : 2 * target_turn - 1]
    tokenized = [tokenizer.bos_token_id] + tokenized

    # `inputs_decode` and `labels_decode` can be used to check whether the tokenization method is true.
    return dict(
        input_ids=tokenized,
        inputs_decode=tokenizer.decode(tokenized),
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
    expect_bos = True
    gpt_bos = False if template.messages[0][0] == template.roles[0] else True
    gpt_eos = False if template.messages[0][0] == template.roles[0] else True

    for i, token_id in enumerate(input_ids):
        if token_id == tokenizer.bos_token_id and expect_bos:
            if gpt_bos:
                starts.append(i)
            gpt_bos = not gpt_bos
            expect_bos = not expect_bos
            continue
        if token_id == tokenizer.eos_token_id and not expect_bos:
            if gpt_eos:
                ends.append(i)
            gpt_eos = not gpt_eos
            expect_bos = not expect_bos

    if len(starts) != target_turn or len(ends) != target_turn:
        warnings.warn(
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
    tokenizer: PreTrainedTokenizer,
    conversation_template: Conversation = default_conversation,
    ignore_index: int = None,
    max_length: int = 4096,
) -> Dict[str, Union[int, str, List[int]]]:
    """
    A tokenization function to tokenize an original pretraining data point as following:
        {"context": [{"from": "human", "content": "xxx"}, {"from": "assistant", "content": "xxx"}],
        "chosen": {"from": "assistant", "content": "xxx"}, "rejected": {"from": "assistant", "content": "xxx"}}
    """
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
        warnings.warn(
            "Please make sure leading context is started and ended with a line from human" + str(template.messages)
        )
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
