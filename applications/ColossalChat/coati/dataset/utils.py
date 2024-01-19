import io
import json
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import PreTrainedTokenizer


def is_rank_0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def read_string_by_schema(data: Dict[str, Any], schema: str) -> str:
    """
    Read a feild of the dataset be schema
    Args:
        data: Dict[str, Any]
        schema: cascaded feild names seperated by '.'. e.g. person.name.first will access data['person']['name']['first']
    """
    keys = schema.split(".")
    result = data
    for key in keys:
        result = result.get(key, None)
        if result is None:
            return ""
    assert isinstance(result, str), f"dataset element is not a string: {result}"
    return result


def pad_to_max_len(
    sequence: List[torch.Tensor], max_length: int, padding_value: int, batch_first: bool = True, padding_side="left"
):
    """
    Args:
        sequence: a batch of tensor of shape [batch_size, seq_len] if batch_first==True
    """
    if padding_side == "left":
        reversed_sequence = [seq.flip(dims=(0,)) for seq in sequence]
        padded = torch.nn.utils.rnn.pad_sequence(
            sequences=reversed_sequence, batch_first=batch_first, padding_value=padding_value
        )
        to_pad = max_length - padded.size(1)
        padded = F.pad(padded, (0, to_pad), value=padding_value)
        return torch.flip(padded, dims=(1,))
    elif padding_side == "right":
        padded = torch.nn.utils.rnn.pad_sequence(
            sequences=sequence, batch_first=batch_first, padding_value=padding_value
        )
        to_pad = max_length - padded.size(1)
        return F.pad(padded, (0, to_pad), value=padding_value)
    else:
        raise RuntimeError(f"`padding_side` can only be `left` or `right`, " f"but now `{padding_side}`")


def chuncate_sequence(sequence: List[torch.Tensor], max_length: int, dtype: Any):
    """
    Args:
        sequence: a batch of tensor of shape [batch_size, seq_len] if batch_first==True
    """
    return [
        torch.Tensor(seq[:max_length]).to(dtype) if len(seq) > max_length else torch.Tensor(seq).to(dtype)
        for seq in sequence
    ]

def find_first_occurrence_subsequence(seq: Union[torch.Tensor, List[Any]], 
    subseq: Union[torch.Tensor, List[Any]], start_index: int=0) -> int:
    if not subseq:
        return 0
    for i in range(start_index, len(seq)-len(subseq)+1):
        if seq[i:i+len(subseq)] == subseq:
            return i
    return -1

def find_all_occurrence_subsequence(seq: Union[torch.Tensor, List[Any]], 
    subseq: Union[torch.Tensor, List[Any]]) -> List[int]:
    if not subseq:
        return list(range(len(seq)))
    result = []
    for i in range(len(seq)-len(subseq)+1):
        if seq[i:i+len(subseq)] == subseq:
            result.append(i)
    return result


def find_subsequences_that_concatenate_to_target_string(sequence: List[str], target: str, depth: int=20) -> Tuple[int, int]:
    """
    Args:
        target: a string
    Returns:
        start end index of the subsequence
    """
    sequence = [s.replace(' ','') for s in sequence]
    target = target.replace(' ','')
    all_occurances = []
    for i in range(len(sequence)):
        for j in range(i+1, min(len(sequence), i+depth)):
            if ''.join(sequence[i:j]) == target:
                all_occurances.append([i, j])
    return all_occurances

def longest_common_sublist(lists):
    # Function to find all sublists of a list
    def find_sublists(lst):
        sublists = []
        for i in range(len(lst)):
            for j in range(i + 1, len(lst) + 1):
                sublists.append(lst[i:j])
        return sublists

    # Find all sublists for the first list
    common_sublists = find_sublists(lists[0])

    # Iterate over the rest of the lists
    for lst in lists[1:]:
        # Find sublists for the current list
        lst_sublists = find_sublists(lst)
        # Keep only those sublists that are common with the previous lists
        common_sublists = [sublist for sublist in common_sublists if sublist in lst_sublists]

    # Find the longest common sublist
    if common_sublists:
        return max(common_sublists, key=len)
    else:
        return []

def find_corresponding_tokens_in_tokenized_prompt(prompt: str, tokenizer: PreTrainedTokenizer, target: str) -> List[int]:
    if target == "":
        return []
    tokenized = tokenizer([prompt], add_special_tokens=False)["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(tokenized, skip_special_tokens=False)
    corresponding_str = [tokenizer.convert_tokens_to_string([token]) for token in tokens]
    token_str_mapping = [(tokenized[i], s) for i, s in enumerate(corresponding_str)]
    all_occurances_of_target_tokens = find_subsequences_that_concatenate_to_target_string(corresponding_str, target)

    # If there are multiple occurance of the target, target tokens are the longest common substring
    ret = longest_common_sublist([tokenized[occurance[0]:occurance[1]] for occurance in all_occurances_of_target_tokens])
    if len(ret)==0:
        return None # fail
    return ret

def find_sep_tokens(prompt: str, tokenizer: PreTrainedTokenizer, sep_name: str, sep_str: str, conversation_template_config: Dict) -> List[int]:
    tokens = find_corresponding_tokens_in_tokenized_prompt(prompt, tokenizer, sep_str)
    if tokens is not None:
        return tokens
    else:
        tokenized = tokenizer([prompt], add_special_tokens=False)["input_ids"][0]
        tokens = tokenizer.convert_ids_to_tokens(tokenized, skip_special_tokens=False)
        corresponding_str = [tokenizer.convert_tokens_to_string([token]) for token in tokens]
        token_str_mapping = [(tokenized[i], s) for i, s in enumerate(corresponding_str)]
        raise ValueError(f"Unable to set the {sep_name} seperator automatically, Please config it manually, \nPrompt: {prompt}\nToken mapping:\n{token_str_mapping}\nCurrent Setting:\n{str(conversation_template_config)}")
        
def find_round_starts_and_ends(tokenizer: PreTrainedTokenizer, template: Any, prompt: str, tokenized: List[int],
        seps_order: List[str], end_of_system_line_position: int):
    '''
    Searching for the starts and ends indices from the end_of_system_line_position
    Args:
        tokenizer: the tokenizer to use
        template: the conversation template
        seps_orders: list of seperator names
        end_of_system_line_position: the search where start from this index. After that index, we search for the pattern iteratively:
            human_line_start -> human_line_end -> assistant_line_start -> assistant_line_end ...
    '''
    starts = [0]
    ends = [0]
    offset = max(end_of_system_line_position, 0)
    for sep_name in seps_order:
        sep_ids = getattr(template, sep_name)
        if len(sep_ids)==0:
            # Line starts right after the previous seqence control token
            # e.g. llama 
            # <s>[INST] what are some pranks with a pen I can do? [/INST] Are you looking for practical joke ideas? </s>
            if "start" in sep_name:
                starts.append(offset)
            elif "end" in sep_name:
                ends.append(offset)
            continue
        start_of_sep = find_first_occurrence_subsequence(tokenized, sep_ids, offset)
        if start_of_sep==-1:
            tokens = tokenizer.convert_ids_to_tokens(tokenized, skip_special_tokens=False)
            corresponding_str = [tokenizer.convert_tokens_to_string([token]) for token in tokens]
            tokens = tokenizer.convert_ids_to_tokens(tokenized, skip_special_tokens=False)
            corresponding_str = [tokenizer.convert_tokens_to_string([token]) for token in tokens]
            token_str_mapping = [(tokenized[i], s) for i, s in enumerate(corresponding_str)]
            raise ValueError(f"Please check whether the message contains the {sep_name} seperator \"{tokenizer.decode(getattr(template, sep_name), skip_special_tokens=False)}\" \
                in the prompt {prompt}. Please manually set sequence control tokens if this message continue to occur constantly.\nToken mapping:\n{token_str_mapping}\nCurrent Setting:\n{str(template)}")
        if 'start' in sep_name:
            starts.append(start_of_sep + len(sep_ids))
        elif 'end' in sep_name:
            ends.append(start_of_sep + len(sep_ids))
        offset = start_of_sep + len(sep_ids)
    starts = starts[1:]
    ends = ends[1:]
    return starts, ends