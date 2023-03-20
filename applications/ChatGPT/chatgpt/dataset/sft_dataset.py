from typing import Callable
import random
from torch.utils.data import Dataset
import torch.distributed as dist
from torch.utils.data import IterableDataset
from tqdm import tqdm
import torch

from .utils import is_rank_0


class SFTDataset(Dataset):
    """
    Dataset for sft model

    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
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
            self.prompts.append({
                "input_ids": prompt_token['input_ids'],
                "attention_mask": prompt_token['attention_mask']
            })

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]["input_ids"], self.prompts[idx]["attention_mask"]


class DistributedSampler:
    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers

        return dict(rank=self.rank,
                    worker_size=self.world_size,
                    worker_id=self.worker_id,
                    num_workers=self.num_workers)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def sample(self, data):
        data = list(range(len(data)))
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            data = data[self.rank::self.world_size]
        data = data[self.worker_id::self.num_workers]
        return data


class SFTDistributedDataset(IterableDataset):
    def __init__(self, dataset, tokenizer: Callable,max_length=512, batch_size=16, shuffle=True, partition=True):
        self.prompts = dataset
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.sampler = DistributedSampler(shuffle, partition)
        self.batch_size = batch_size

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def batch(self):
        buf = []
        sampler_info = self.sampler.update()
        indexes = self.sampler.sample(self.prompts)
        for index in indexes:
            data = self.prompts[index]
            prompt = data['prompt'] + data['completion'] + "<|endoftext|>"
            buf.append(prompt)
            if len(buf) >= self.batch_size:
                yield buf
                buf = []
        if len(buf) > 0:
            yield buf

    def __iter__(self):
        for data in self.batch():
            assert isinstance(data, list)
            prompt_token = self.tokenizer(data,
                                          max_length=self.max_length,
                                          padding="max_length",
                                          truncation=True,
                                          return_tensors="pt")
            input_ids = prompt_token['input_ids']
            attention_mask = prompt_token['attention_mask']
            yield input_ids, attention_mask
