from typing import Callable

from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import is_rank_0


class RewardDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
    """

    def __init__(self, dataset, tokenizer: Callable, max_length: int) -> None:
        super().__init__()
        self.chosen = []
        self.reject = []
        for data in tqdm(dataset, disable=not is_rank_0()):
            prompt = data['prompt']

            chosen = prompt + data['chosen'] + "<|endoftext|>"
            chosen_token = tokenizer(chosen,
                                     max_length=max_length,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")
            self.chosen.append({
                "input_ids": chosen_token['input_ids'],
                "attention_mask": chosen_token['attention_mask']
            })

            reject = prompt + data['rejected'] + "<|endoftext|>"
            reject_token = tokenizer(reject,
                                     max_length=max_length,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")
            self.reject.append({
                "input_ids": reject_token['input_ids'],
                "attention_mask": reject_token['attention_mask']
            })

    def __len__(self):
        length = len(self.chosen)
        return length

    def __getitem__(self, idx):
        return self.chosen[idx]["input_ids"], self.chosen[idx]["attention_mask"], self.reject[idx][
            "input_ids"], self.reject[idx]["attention_mask"]
