from typing import Callable
import random
from torch.utils.data import Dataset
import torch.distributed as dist
from tqdm import tqdm
import torch

from .utils import is_rank_0


class SFTDataset(Dataset):
    """
    Dataset for sft model

    Args:
        dataset: dataset for supervised model
        tokenizer: tokenizer for supervised model
        max_length: max length of input
    """

    def __init__(self, dataset, tokenizer: Callable, max_length: int=512) -> None:
        super().__init__()
        self.prompts = []

        for data in tqdm(dataset, disable=not is_rank_0()):
            prompt = data['prompt'] + data['completion'] + "<|endoftext|>"
            prompt_token = tokenizer(prompt,
                                     max_length=max_length,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")

            self.prompts.append(prompt_token)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]
