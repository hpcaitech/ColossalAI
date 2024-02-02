#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tokenization utils for constructing dataset for ppo, dpo, sft, rm
"""

import warnings
from copy import deepcopy
from typing import Any, Dict, List, Union

from coati.dataset.conversation import Conversation
from coati.dataset.utils import find_first_occurrence_subsequence, find_round_starts_and_ends
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
    prompt, seps_info = template.get_prompt(2 * target_turn, get_seps_info=True)
    
    seps_order = seps_info['seps_order']
    end_of_system_line_position = seps_info['end_of_system_line_position']
    tokenized = tokenizer([prompt], add_special_tokens=False)["input_ids"][0]

    starts, ends = find_round_starts_and_ends(tokenizer, template, prompt, tokenized, seps_order, end_of_system_line_position)

    if len(starts) != target_turn*2 or len(ends) != target_turn*2:
        tokens = tokenizer.convert_ids_to_tokens(tokenized, skip_special_tokens=False)
        corresponding_str = [tokenizer.convert_tokens_to_string([token]) for token in tokens]
        token_str_mapping = [(tokenized[i], s) for i, s in enumerate(corresponding_str)]
        raise ValueError(f"Please check whether the sequence control seperators are configed correctly \"{tokenizer.decode(getattr(template, sep_name), skip_special_tokens=False)}\" \
            in the prompt {prompt}. Please manually set sequence control tokens if this message continue to occur constantly.\nToken mapping:\n{token_str_mapping}\nCurrent Setting:\n{str(template)}")
        return dict(
            input_ids=None,
            labels=None,
            inputs_decode=None,
            labels_decode=None,
            seq_length=None,
            seq_category=None,
        )
    target_turns = []
    last_sep = None
    cnt = 0
    while len(seps_order)>0:
        turn1 = seps_order.pop(0)
        turn2 = seps_order.pop(0)
        assert turn1.endswith('start') and turn2.endswith('end')
        assert turn1.replace('start','end')==turn2
        if turn1.startswith('assistant'):
            target_turns.append(cnt)
        cnt += 1

    starts=[starts[i] for i in target_turns]
    ends=[ends[i] for i in target_turns]

    labels = [ignore_index] * len(tokenized)
    for start, end in zip(starts, ends):
        labels[start: end] = tokenized[start: end]

    labels_decode = deepcopy(labels)
    if tokenizer.eos_token_id is not None:
        for i, z in enumerate(labels_decode):
            if z == ignore_index:
                labels_decode[i] = tokenizer.eos_token_id
    else:
        # If the tokenizer doesn't have eos_token or pad_token: Qwen
        for i, z in enumerate(labels_decode):
            if z == ignore_index:
                labels_decode[i] = 1  # Label decode is for debugging only, it is not used in training
 
    
    if tokenizer.bos_token_id is not None:
        tokenized = [tokenizer.bos_token_id] + tokenized
        labels = [ignore_index] + labels
        label_decode = [tokenizer.eos_token_id or 1] + labels_decode

    # For some model without bos/eos may raise the following errors
    try:
        inputs_decode = tokenizer.decode(tokenized)
    except TypeError as e:
        raise TypeError(str(e)+f'\nUnable to decode input_ids: {tokenized}')

    try:
        labels_decode = tokenizer.decode(labels_decode)
    except TypeError as e:
        raise TypeError(str(e)+f'\nUnable to decode labels: {labels_decode}')


    
    return dict(
        input_ids=tokenized,
        labels=labels,
        inputs_decode=inputs_decode,
        labels_decode=labels_decode,
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
    tokenized = tokenizer([prompt], add_special_tokens=False)["input_ids"][0] 
    if tokenizer.bos_token_id is not None:
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


def apply_rlhf_data_format(template: Conversation, tokenizer: Any, context_len: int, mask_out_target_assistant_line_end=False):
    target_turn = int(len(template.messages)/2)
    prompt, seps_info = template.get_prompt(target_turn * 2, get_seps_info=True)
    seps_order = seps_info['seps_order']
    end_of_system_line_position = seps_info['end_of_system_line_position']
    tokenized = tokenizer([prompt], add_special_tokens=False)["input_ids"][0]

    # Find start index and end index of each dialogue
    starts, ends = find_round_starts_and_ends(tokenizer, template, prompt, tokenized, seps_order, end_of_system_line_position)

    if len(starts) != target_turn*2 or len(ends) != target_turn*2:
        tokens = tokenizer.convert_ids_to_tokens(tokenized, skip_special_tokens=False)
        corresponding_str = [tokenizer.convert_tokens_to_string([token]) for token in tokens]
        token_str_mapping = [(tokenized[i], s) for i, s in enumerate(corresponding_str)]
        raise ValueError(f"Please check whether the sequence control seperators are configed correctly \"{tokenizer.decode(getattr(template, sep_name), skip_special_tokens=False)}\" \
            in the prompt {prompt}. Please manually set sequence control tokens if this message continue to occur constantly.\nToken mapping:\n{token_str_mapping}\nCurrent Setting:\n{str(template)}")
        return dict(input_ids=None, loss_mask=None, label_decode=None)

    target_turns = []
    last_sep = None
    cnt = 0
    while len(seps_order)>0:
        turn1 = seps_order.pop(0)
        turn2 = seps_order.pop(0)
        assert turn1.endswith('start') and turn2.endswith('end')
        assert turn1.replace('start','end')==turn2
        if turn1.startswith('assistant'):
            target_turns.append(cnt)
        cnt += 1

    starts=[starts[i] for i in target_turns][context_len:]
    ends=[ends[i] for i in target_turns][context_len:]
    if mask_out_target_assistant_line_end:
        ends[-1] = ends[-1]-len(template.assistant_line_end)

    loss_mask = [0] * len(tokenized)
    mask_token = tokenizer.eos_token_id or tokenizer.pad_token_id
    if mask_token is None:
        mask_token = 1 # If the tokenizer doesn't have eos_token or pad_token: Qwen

    label_decode = [mask_token] * len(tokenized)
    for start, end in zip(starts, ends):
        for i in range(start, end):
            loss_mask[i] = 1
            label_decode[i] = tokenized[i]
    label_decode = tokenizer.decode(label_decode, skip_special_tokens=False)
    if tokenizer.bos_token_id is not None:
        tokenized = [tokenizer.bos_token_id] + tokenized
        loss_mask = [0] + loss_mask
        label_decode = [mask_token] + label_decode
    
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
            from_str = 'user'
        elif from_str.lower() == "assistant":
            from_str = 'assistant'
        else:
            raise ValueError(f"Unsupported role {from_str.lower()}")

        if len(template.messages) > 0 and from_str == template.messages[-1]['role']:
            # Concate adjacent message from the same role
            template.messages[-1]['content'] = str(template.messages[-1]['content'] + ' ' + mess["content"])
        else:
            template.append_message(from_str, mess["content"])

    if len(template.messages) % 2 != 1:
        warnings.warn(
            "Please make sure leading context starts and ends with a line from human\nLeading context: " + str(template.messages)
        )
        return dict(
            chosen_input_ids=None,
            chosen_loss_mask=None,
            chosen_label_decode=None,
            rejected_input_ids=None,
            rejected_loss_mask=None,
            rejected_label_decode=None
        )
    round_of_context = int((len(template.messages) - 1) / 2)

    assert context[-1]["from"].lower() == "human", "The last message in context should be from human."
    chosen = deepcopy(template)
    rejected = deepcopy(template)

    for round in range(len(data_point["chosen"])):
        from_str = data_point["chosen"][round]["from"]
        if from_str.lower() == "human":
            from_str = 'user'
        elif from_str.lower() == "assistant":
            from_str = 'assistant'
        else:
            raise ValueError(f"Unsupported role {from_str.lower()}")
        chosen.append_message(from_str, data_point["chosen"][round]["content"])

    for round in range(len(data_point["rejected"])):
        from_str = data_point["rejected"][round]["from"]
        if from_str.lower() == "human":
            from_str = 'user'
        elif from_str.lower() == "assistant":
            from_str = 'assistant'
        else:
            raise ValueError(f"Unsupported role {from_str.lower()}")
        rejected.append_message(from_str, data_point["rejected"][round]["content"])

    (
        chosen_input_ids,
        chosen_loss_mask,
        chosen_label_decode,
        rejected_input_ids,
        rejected_loss_mask,
        rejected_label_decode
    ) = (None, None, None, None, None, None)
    if (
        len(tokenizer([chosen.get_prompt(len(chosen.messages))], add_special_tokens=False)["input_ids"][0])
        <= max_length - 1
        and len(tokenizer([rejected.get_prompt(len(rejected.messages))], add_special_tokens=False)["input_ids"][0])
        <= max_length - 1
    ):
        chosen_data_packed = apply_rlhf_data_format(chosen, tokenizer, round_of_context)
        (chosen_input_ids, chosen_loss_mask, chosen_label_decode) = (
            chosen_data_packed["input_ids"],
            chosen_data_packed["loss_mask"],
            chosen_data_packed["label_decode"]
        )

        rejected_data_packed = apply_rlhf_data_format(rejected, tokenizer, round_of_context, 
            mask_out_target_assistant_line_end=True)
        (rejected_input_ids, rejected_loss_mask, rejected_label_decode) = (
            rejected_data_packed["input_ids"],
            rejected_data_packed["loss_mask"],
            rejected_data_packed["label_decode"]
        )

        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_loss_mask": chosen_loss_mask,
            "chosen_label_decode": chosen_label_decode,
            "rejected_input_ids": rejected_input_ids,
            "rejected_loss_mask": rejected_loss_mask,
            "rejected_label_decode": rejected_label_decode
        }
    else:
        return dict(
            chosen_input_ids=None,
            chosen_loss_mask=None,
            chosen_label_decode=None,
            rejected_input_ids=None,
            rejected_loss_mask=None,
            rejected_label_decode=None
        )
