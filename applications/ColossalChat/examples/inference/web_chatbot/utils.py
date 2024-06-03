import copy
import json
from threading import Lock
from typing import List

import jieba
import torch
from coati.dataset.conversation import default_conversation
from pydantic import BaseModel, Field


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
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
        )

    return model_kwargs


class Dialogue(BaseModel):
    instruction: str = Field(min_length=1, example="Count up from 1 to 500.")
    response: str = Field(example="")


class ChatPromptProcessor:
    SAFE_RESPONSE = "The input/response contains inappropriate content, please rephrase your prompt."

    def __init__(self, censored_words: List[str] = []):
        self.censored_words = set([word.lower() for word in censored_words])
        self.conv = copy.deepcopy(default_conversation)

    def preprocess_prompt(self, history: List[Dialogue]) -> str:
        self.conv.clear()
        for round in history:
            self.conv.append_message(self.conv.roles[0], round.instruction)
            if len(round.instruction) > 0:
                self.conv.append_message(self.conv.roles[1], round.response)
        return self.conv.get_prompt()

    def postprocess_output(self, output: str) -> str:
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
