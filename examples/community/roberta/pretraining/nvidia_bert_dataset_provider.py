import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import torch
import torch.distributed as dist
from bert_dataset_provider import BertDatasetProviderInterface
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


# Workaround because python functions are not picklable
class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


def create_pretraining_dataset(
    input_file, max_predictions_per_seq, num_workers, train_batch_size, worker_init, data_sampler
):
    train_data = pretraining_dataset(input_file=input_file, max_predictions_per_seq=max_predictions_per_seq)
    train_dataloader = DataLoader(
        train_data,
        sampler=data_sampler(train_data),
        batch_size=train_batch_size,
        num_workers=num_workers,
        worker_init_fn=worker_init,
        pin_memory=True,
    )
    return train_dataloader, len(train_data)


class pretraining_dataset(Dataset):
    def __init__(self, input_file, max_predictions_per_seq):
        self.input_file = input_file
        self.max_predictions_per_seq = max_predictions_per_seq
        f = h5py.File(input_file, "r")
        keys = ["input_ids", "input_mask", "segment_ids", "masked_lm_positions"]
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.inputs[0])

    def __getitem__(self, index):
        [input_ids, input_mask, segment_ids, masked_lm_labels] = [
            (
                torch.from_numpy(input[index].astype(np.int64))
                if indice < 5
                else torch.from_numpy(np.asarray(input[index].astype(np.int64)))
            )
            for indice, input in enumerate(self.inputs)
        ]

        return [input_ids, input_mask, segment_ids, masked_lm_labels]


class NvidiaBertDatasetProvider(BertDatasetProviderInterface):
    def __init__(self, args, evaluate=False):
        self.num_workers = args.num_workers
        self.max_seq_length = args.max_seq_length
        self.max_predictions_per_seq = args.max_predictions_per_seq

        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        if not evaluate:
            self.train_micro_batch_size_per_gpu = args.train_micro_batch_size_per_gpu
        else:
            self.train_micro_batch_size_per_gpu = args.eval_micro_batch_size_per_gpu
        self.logger = args.logger

        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Initialize dataset files
        if not evaluate:
            self.dataset_files = [
                os.path.join(args.data_path_prefix, f)
                for f in os.listdir(args.data_path_prefix)
                if os.path.isfile(os.path.join(args.data_path_prefix, f)) and "h5" in f
            ]
        else:
            self.dataset_files = [
                os.path.join(args.eval_data_path_prefix, f)
                for f in os.listdir(args.eval_data_path_prefix)
                if os.path.isfile(os.path.join(args.eval_data_path_prefix, f)) and "h5" in f
            ]

        self.dataset_files.sort()
        # random.shuffle(self.dataset_files)
        self.num_files = len(self.dataset_files)
        # self.data_sampler = RandomSampler
        self.data_sampler = DistributedSampler

        self.worker_init = WorkerInitObj(args.seed + args.local_rank)
        self.dataset_future = None
        self.pool = ProcessPoolExecutor(1)
        self.data_file = None
        self.shuffle = True

        if self.global_rank == 0:
            self.logger.info(f"NvidiaBertDatasetProvider - Initialization: num_files = {self.num_files}")

    def get_shard(self, index):
        start = time.time()
        if self.dataset_future is None:
            self.data_file = self._get_shard_file(index)
            self.train_dataloader, sample_count = create_pretraining_dataset(
                input_file=self.data_file,
                max_predictions_per_seq=self.max_predictions_per_seq,
                num_workers=self.num_workers,
                train_batch_size=self.train_micro_batch_size_per_gpu,
                worker_init=self.worker_init,
                data_sampler=self.data_sampler,
            )
        else:
            self.train_dataloader, sample_count = self.dataset_future.result(timeout=None)

        self.logger.info(
            f"Data Loading Completed for Pretraining Data from {self.data_file} with {sample_count} samples took {time.time()-start:.2f}s."
        )

        return self.train_dataloader, sample_count

    def release_shard(self):
        del self.train_dataloader
        self.pool.shutdown()

    def prefetch_shard(self, index):
        self.data_file = self._get_shard_file(index)
        self.dataset_future = self.pool.submit(
            create_pretraining_dataset,
            self.data_file,
            self.max_predictions_per_seq,
            self.num_workers,
            self.train_micro_batch_size_per_gpu,
            self.worker_init,
            self.data_sampler,
        )

    def get_batch(self, batch_iter):
        return batch_iter

    def prefetch_batch(self):
        pass

    def _get_shard_file(self, shard_index):
        file_index = self._get_shard_file_index(shard_index, self.global_rank)
        return self.dataset_files[file_index]

    def _get_shard_file_index(self, shard_index, global_rank):
        # if dist.is_initialized() and self.world_size > self.num_files:
        #     remainder = self.world_size % self.num_files
        #     file_index = (shard_index * self.world_size) + global_rank + (
        #         remainder * shard_index)
        # else:
        #     file_index = shard_index * self.world_size + global_rank

        return shard_index % self.num_files

    def shuffle_dataset(self, epoch):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(self.num_files, generator=g).tolist()
            new_dataset = [self.dataset_files[i] for i in indices]
            self.dataset_files = new_dataset
