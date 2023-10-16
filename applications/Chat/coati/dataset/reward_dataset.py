from typing import Callable

from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import is_rank_0


# Dahoas/rm-static
class RmStaticDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
        special_token: special token at the end of sentence
    """

    def __init__(self, dataset, tokenizer: Callable, max_length: int, special_token=None) -> None:
        super().__init__()
        self.end_token = tokenizer.eos_token if special_token is None else special_token

        chosen = [data["prompt"] + data["chosen"] + self.end_token for data in tqdm(dataset, disable=not is_rank_0())]
        chosen_token = tokenizer(
            chosen, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        self.chosen = {"input_ids": chosen_token["input_ids"], "attention_mask": chosen_token["attention_mask"]}

        reject = [data["prompt"] + data["rejected"] + self.end_token for data in tqdm(dataset, disable=not is_rank_0())]
        reject_token = tokenizer(
            reject, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        self.reject = {"input_ids": reject_token["input_ids"], "attention_mask": reject_token["attention_mask"]}

    def __len__(self):
        length = self.chosen["input_ids"].shape[0]
        return length

    def __getitem__(self, idx):
        return (
            self.chosen["input_ids"][idx],
            self.chosen["attention_mask"][idx],
            self.reject["input_ids"][idx],
            self.reject["attention_mask"][idx],
        )


# Anthropic/hh-rlhf
class HhRlhfDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
        special_token: special token at the end of sentence
    """

    def __init__(self, dataset, tokenizer: Callable, max_length: int, special_token=None) -> None:
        super().__init__()
        self.end_token = tokenizer.eos_token if special_token is None else special_token

        chosen = [data["chosen"] + self.end_token for data in tqdm(dataset, disable=not is_rank_0())]
        chosen_token = tokenizer(
            chosen, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        self.chosen = {"input_ids": chosen_token["input_ids"], "attention_mask": chosen_token["attention_mask"]}

        reject = [data["rejected"] + self.end_token for data in tqdm(dataset, disable=not is_rank_0())]
        reject_token = tokenizer(
            reject, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        self.reject = {"input_ids": reject_token["input_ids"], "attention_mask": reject_token["attention_mask"]}

    def __len__(self):
        length = self.chosen["input_ids"].shape[0]
        return length

    def __getitem__(self, idx):
        return (
            self.chosen["input_ids"][idx],
            self.chosen["attention_mask"][idx],
            self.reject["input_ids"][idx],
            self.reject["attention_mask"][idx],
        )
