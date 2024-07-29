import io
import json
from typing import Any, Dict, List

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


def find_first_occurrence_subsequence(seq: torch.Tensor, subseq: torch.Tensor, start_index: int = 0) -> int:
    if subseq is None:
        return 0
    for i in range(start_index, len(seq) - len(subseq) + 1):
        if torch.all(seq[i : i + len(subseq)] == subseq):
            return i
    return -1


def tokenize_and_concatenate(
    tokenizer: PreTrainedTokenizer,
    text: List[str],
    require_loss: List[bool],
    max_length: int,
    discard_non_loss_tokens_at_tail: bool = True,
):
    """
    Tokenizes a list of texts using the provided tokenizer and concatenates the tokenized outputs.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenization.
        text (List[str]): The list of texts to tokenize.
        require_loss (List[bool]): A list of boolean values indicating whether each text requires loss calculation.
        max_length: used to truncate the input ids
        discard_non_loss_tokens_at_tail: whether to discard the non-loss tokens at the tail

    if the first round has already exeeded max length
    - if the user query already exeeded max length, discard the sample
    - if only the first assistant response exeeded max length, truncate the response to fit the max length
    else keep the first several complete rounds of the conversations until max length is reached

    Returns:
        Tuple[List[int], List[int], List[int]]: A tuple containing the concatenated tokenized input ids,
        the start positions of loss spans, and the end positions of loss spans.
    """
    input_ids = []
    loss_starts = []
    loss_ends = []
    for s, r in zip(text, require_loss):
        tokenized = tokenizer(s, add_special_tokens=False)["input_ids"]
        if not max_length or len(input_ids) + len(tokenized) <= max_length or len(loss_ends) == 0:
            if r:
                loss_starts.append(len(input_ids))
                loss_ends.append(len(input_ids) + len(tokenized))
            input_ids.extend(tokenized)
    if max_length and loss_starts[0] >= max_length:
        return None, None, None
    if discard_non_loss_tokens_at_tail:
        input_ids = input_ids[: loss_ends[-1]]
    if max_length:
        input_ids = input_ids[:max_length]
        loss_ends[-1] = min(max_length, loss_ends[-1])
    return input_ids, loss_starts, loss_ends


def split_templated_prompt_into_chunks(messages: List[Dict[str, str]], prompt: str, end_of_assistant: str):
    # Seperate templated prompt into chunks by human/assistant's lines, prepare data for tokenize_and_concatenate
    start_idx = 0
    chunks = []
    require_loss = []
    for line in messages:
        content_length = len(line["content"])
        first_occur = prompt.find(line["content"], start_idx)
        if line["role"].lower() == "assistant" and end_of_assistant in prompt[first_occur + content_length :]:
            content_length = (
                prompt.find(end_of_assistant, first_occur + content_length) + len(end_of_assistant) - first_occur
            )
        # if the tokenized content start with a leading space, we want to keep it in loss calculation
        # e.g., Assistant: I am saying...
        # if the tokenized content doesn't start with a leading space, we only need to keep the content in loss calculation
        # e.g.,
        # Assistant:   # '\n' as line breaker
        # I am saying...
        if prompt[first_occur - 1] != " ":
            chunks.append(prompt[start_idx:first_occur])
            chunks.append(prompt[first_occur : first_occur + content_length])
        else:
            chunks.append(prompt[start_idx : first_occur - 1])
            chunks.append(prompt[first_occur - 1 : first_occur + content_length])
        start_idx = first_occur + content_length
        if line["role"].lower() == "assistant":
            require_loss.append(False)
            require_loss.append(True)
        else:
            require_loss.append(False)
            require_loss.append(False)
    chunks.append(prompt[start_idx:])
    require_loss.append(False)
    return chunks, require_loss
