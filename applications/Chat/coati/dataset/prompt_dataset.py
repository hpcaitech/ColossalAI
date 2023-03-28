import copy
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence

import torch
import torch.distributed as dist
import transformers
from torch.utils.data import Dataset
from tqdm import tqdm

from colossalai.logging import get_dist_logger

from .utils import is_rank_0, jload

logger = get_dist_logger()


class PromptDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, max_datasets_size: int = None):
        super(PromptDataset, self).__init__()
        self.prompt = []
        logger.info("Loading data...")
        list_data_dict = jload(data_path)
        logger.info(f"Loaded {len(list_data_dict)} examples.")

        if max_datasets_size is not None:
            logger.info(f"Limiting dataset to {max_datasets_size} examples.")
            list_data_dict = list_data_dict[:max_datasets_size]

        for data_dict in list_data_dict:
            token = tokenizer(data_dict["instruction"],
                              return_tensors='pt',
                              max_length=96,
                              padding='max_length',
                              truncation=True)
            for idx in token['input_ids']:
                self.prompt.append(idx.to(torch.cuda.current_device()))

    def __len__(self):
        return len(self.prompt)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.prompt[i]
