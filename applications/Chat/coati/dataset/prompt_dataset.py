from collections import defaultdict
from typing import Dict

import torch
import transformers
from torch.utils.data import Dataset

from colossalai.logging import get_dist_logger

from .utils import jload, read_string_by_schema


class PromptDataset(Dataset):
    """Dataset for supervised fine-tuning.

    Args:
        dataset: dataset for supervised model
        tokenizer: tokenizer for supervised model
        max_datasets_size: number of examples to use from the dataset
        max_length: max length of input
        verbose: whether to display the first two item in the dataset
        dataset_schema: schema for reading the dataset. cascaded feild names seperated by '.'.
             e.g. person.name.first will access data['person']['name']['first']
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        max_datasets_size: int = None,
        max_length: int = 96,
        dataset_schema: Dict[str, str] = {"instruction": "instruction"},
    ):
        super(PromptDataset, self).__init__()
        self.keyed_prompt = defaultdict(list)
        self.logger = get_dist_logger()
        self.logger.info("Loading data...")
        list_data_dict = jload(data_path)
        self.logger.info(f"Loaded {len(list_data_dict)} examples.")

        if max_datasets_size is not None:
            self.logger.info(f"Limiting dataset to {max_datasets_size} examples.")
            list_data_dict = list_data_dict[:max_datasets_size]

        instructions = [
            read_string_by_schema(data_dict, dataset_schema["instruction"]) + "\n" for data_dict in list_data_dict
        ]
        tokens = tokenizer(
            instructions, return_tensors="pt", max_length=max_length, padding="max_length", truncation=True
        )
        for k, tensor in tokens.items():
            self.keyed_prompt[k] = tensor.to(torch.cuda.current_device()).unbind()

    def __len__(self):
        return len(self.keyed_prompt["input_ids"])

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return {k: v[i] for k, v in self.keyed_prompt.items()}
