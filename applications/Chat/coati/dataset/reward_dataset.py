from typing import Callable, Dict

from torch.utils.data import Dataset
from tqdm import tqdm

from colossalai.logging import get_dist_logger

from .utils import is_rank_0, read_string_by_schema


class PreferenceDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
        special_token: special token at the end of sentence
        dataset_schema: schema for reading the dataset. cascaded feild names seperated by '.'.
             e.g. person.name.first will access data['person']['name']['first']
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        special_token=None,
        dataset_schema: Dict[str, str] = {"prompt": "", "chosen": "chosen", "rejected": "rejected"},
    ) -> None:
        super().__init__()
        self.end_token = tokenizer.eos_token if special_token is None else special_token
        chosen = [
            (read_string_by_schema(data, dataset_schema["prompt"]) if "prompt" in dataset_schema else "")
            + (read_string_by_schema(data, dataset_schema["chosen"]) if "chosen" in dataset_schema else "")
            + self.end_token
            for data in tqdm(dataset, disable=not is_rank_0())
        ]
        self.logger = get_dist_logger()
        self.logger.info("Tokenizing inputs... This may take some time...")
        chosen_token = tokenizer(
            chosen, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        self.chosen = {"input_ids": chosen_token["input_ids"], "attention_mask": chosen_token["attention_mask"]}

        reject = [
            (read_string_by_schema(data, dataset_schema["prompt"]) if "prompt" in dataset_schema else "")
            + (read_string_by_schema(data, dataset_schema["rejected"]) if "rejected" in dataset_schema else "")
            + self.end_token
            for data in tqdm(dataset, disable=not is_rank_0())
        ]
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
