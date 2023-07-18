import copy
import random
from collections import defaultdict
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

    def __init__(self,
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_datasets_size: int = None,
                 max_length: int = 96):
        super(PromptDataset, self).__init__()
        self.keyed_prompt = defaultdict(list)
        logger.info("Loading data...")
        list_data_dict = jload(data_path)
        logger.info(f"Loaded {len(list_data_dict)} examples.")

        if max_datasets_size is not None:
            logger.info(f"Limiting dataset to {max_datasets_size} examples.")
            list_data_dict = list_data_dict[:max_datasets_size]

        instructions = [data_dict["instruction"] for data_dict in list_data_dict]
        tokens = tokenizer(instructions,
                           return_tensors='pt',
                           max_length=max_length,
                           padding='max_length',
                           truncation=True)
        for k, tensor in tokens.items():
            self.keyed_prompt[k] = tensor.to(torch.cuda.current_device()).unbind()

    def __len__(self):
        return len(self.keyed_prompt["input_ids"])

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return {k: v[i] for k, v in self.keyed_prompt.items()}
