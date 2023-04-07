import re
from threading import Lock
from typing import Any, Callable, Generator, List, Optional
import json
import jieba

import torch
import torch.distributed as dist
import torch.nn as nn
from pydantic import BaseModel, Field

try:
    from transformers.generation_logits_process import (
        LogitsProcessorList,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )
except ImportError:
    from transformers.generation import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper


def prepare_logits_processor(top_k: Optional[int] = None,
                             top_p: Optional[float] = None,
                             temperature: Optional[float] = None) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature is not None and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if top_k is not None and top_k != 0:
        processor_list.append(TopKLogitsWarper(top_k))
    if top_p is not None and top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    return processor_list


def _is_sequence_finished(unfinished_sequences: torch.Tensor) -> bool:
    if dist.is_initialized() and dist.get_world_size() > 1:
        # consider DP
        unfinished_sequences = unfinished_sequences.clone()
        dist.all_reduce(unfinished_sequences)
    return unfinished_sequences.max() == 0


def sample_streamingly(model: nn.Module,
                       input_ids: torch.Tensor,
                       max_generate_tokens: int,
                       early_stopping: bool = False,
                       eos_token_id: Optional[int] = None,
                       pad_token_id: Optional[int] = None,
                       top_k: Optional[int] = None,
                       top_p: Optional[float] = None,
                       temperature: Optional[float] = None,
                       prepare_inputs_fn: Optional[Callable[[torch.Tensor, Any], dict]] = None,
                       update_model_kwargs_fn: Optional[Callable[[dict, Any], dict]] = None,
                       **model_kwargs) -> Generator:

    logits_processor = prepare_logits_processor(top_k, top_p, temperature)
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

    for _ in range(max_generate_tokens):
        model_inputs = prepare_inputs_fn(input_ids, **model_kwargs) if prepare_inputs_fn is not None else {
            'input_ids': input_ids
        }
        outputs = model(**model_inputs)

        next_token_logits = outputs['logits'][:, -1, :]
        # pre-process distribution
        next_token_logits = logits_processor(input_ids, next_token_logits)
        # sample
        probs = torch.softmax(next_token_logits, dim=-1, dtype=torch.float)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        yield next_tokens

        # update generated ids, model inputs for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if update_model_kwargs_fn is not None:
            model_kwargs = update_model_kwargs_fn(outputs, **model_kwargs)

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

        # stop when each sentence is finished if early_stopping=True
        if early_stopping and _is_sequence_finished(unfinished_sequences):
            break


def update_model_kwargs_fn(outputs: dict, **model_kwargs) -> dict:
    if "past_key_values" in outputs:
        model_kwargs["past"] = outputs["past_key_values"]
    else:
        model_kwargs["past"] = None

    # update token_type_ids with last value
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

    # update attention mask
    if "attention_mask" in model_kwargs:
        attention_mask = model_kwargs["attention_mask"]
        model_kwargs["attention_mask"] = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

    return model_kwargs


class Dialogue(BaseModel):
    instruction: str = Field(min_length=1, example='Count up from 1 to 500.')
    response: str = Field(example='')


def _format_dialogue(instruction: str, response: str = ''):
    return f'\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}'


STOP_PAT = re.compile(r'(###|instruction:).*', flags=(re.I | re.S))


class ChatPromptProcessor:
    SAFE_RESPONSE = 'The input/response contains inappropriate content, please rephrase your prompt.'

    def __init__(self, tokenizer, context: str, max_len: int = 2048, censored_words: List[str]=[]):
        self.tokenizer = tokenizer
        self.context = context
        self.max_len = max_len
        self.censored_words = set([word.lower() for word in censored_words])
        # These will be initialized after the first call of preprocess_prompt()
        self.context_len: Optional[int] = None
        self.dialogue_placeholder_len: Optional[int] = None

    def preprocess_prompt(self, history: List[Dialogue], max_new_tokens: int) -> str:
        if self.context_len is None:
            self.context_len = len(self.tokenizer(self.context)['input_ids'])
        if self.dialogue_placeholder_len is None:
            self.dialogue_placeholder_len = len(
                self.tokenizer(_format_dialogue(''), add_special_tokens=False)['input_ids'])
        prompt = self.context
        # the last dialogue must be in the prompt
        last_dialogue = history.pop()
        # the response of the last dialogue is empty
        assert last_dialogue.response == ''
        if len(self.tokenizer(_format_dialogue(last_dialogue.instruction), add_special_tokens=False)
               ['input_ids']) + max_new_tokens + self.context_len >= self.max_len:
            # to avoid truncate placeholder, apply truncate to the original instruction
            instruction_truncated = self.tokenizer(last_dialogue.instruction,
                                                   add_special_tokens=False,
                                                   truncation=True,
                                                   max_length=(self.max_len - max_new_tokens - self.context_len -
                                                               self.dialogue_placeholder_len))['input_ids']
            instruction_truncated = self.tokenizer.decode(instruction_truncated).lstrip()
            prompt += _format_dialogue(instruction_truncated)
            return prompt

        res_len = self.max_len - max_new_tokens - len(self.tokenizer(prompt)['input_ids'])

        rows = []
        for dialogue in history[::-1]:
            text = _format_dialogue(dialogue.instruction, dialogue.response)
            cur_len = len(self.tokenizer(text, add_special_tokens=False)['input_ids'])
            if res_len - cur_len < 0:
                break
            res_len -= cur_len
            rows.insert(0, text)
        prompt += ''.join(rows) + _format_dialogue(last_dialogue.instruction)
        return prompt

    def postprocess_output(self, output: str) -> str:
        output = STOP_PAT.sub('', output)
        return output.strip()

    def has_censored_words(self, text: str) -> bool:
        if len(self.censored_words) == 0:
            return False
        intersection = set(jieba.cut(text.lower())) & self.censored_words
        return len(intersection) > 0

class LockedIterator:

    def __init__(self, it, lock: Lock) -> None:
        self.lock = lock
        self.it = iter(it)

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)

def load_json(path: str):
    with open(path) as f:
        return json.load(f)