from typing import Callable, Dict

from torch.utils.data import Dataset
from tqdm import tqdm

from colossalai.logging import get_dist_logger

from .utils import is_rank_0, read_string_by_schema

logger = get_dist_logger()


class PreferenceDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
        special_token: special token at the end of sentence
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        special_token=None,
        verbose=True,
        dataset_schema: Dict[str, str] = {"prompt": "", "chosen": "chosen", "rejected": "rejected"},
    ) -> None:
        super().__init__()
        self.end_token = tokenizer.eos_token if special_token is None else special_token
        chosen = [
            read_string_by_schema(data, dataset_schema["prompt"])
            + read_string_by_schema(data, dataset_schema["chosen"])
            + self.end_token
            for data in tqdm(dataset, disable=not is_rank_0())
        ]
        chosen_token = tokenizer(
            chosen, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        self.chosen = {"input_ids": chosen_token["input_ids"], "attention_mask": chosen_token["attention_mask"]}

        reject = [
            read_string_by_schema(data, dataset_schema["prompt"])
            + read_string_by_schema(data, dataset_schema["rejected"])
            + self.end_token
            for data in tqdm(dataset, disable=not is_rank_0())
        ]
        reject_token = tokenizer(
            reject, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        self.reject = {"input_ids": reject_token["input_ids"], "attention_mask": reject_token["attention_mask"]}
        self.verbose = verbose
        if self.verbose:
            logger.info(
                "Display the first two item in the preference dataset, to disable this message, set verbose=False in the PreferenceDataset constructor"
            )
            logger.info("chosen: ", chosen[:2])
            logger.info("reject: ", reject[:2])

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
