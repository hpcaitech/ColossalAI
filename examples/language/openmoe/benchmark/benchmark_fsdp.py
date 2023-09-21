import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from model.modeling_openmoe import LlamaConfig, OpenMoeForCausalLM
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers.models.llama import LlamaConfig

from colossalai.moe.manager import MOE_MANAGER


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "14523"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


class RandomDataset(Dataset):

    def __init__(self, num_samples: int = 1000, max_length: int = 2048, vocab_size: int = 32000):
        self.num_samples = num_samples
        self.max_length = max_length
        self.input_ids = torch.randint(0, vocab_size, (num_samples, max_length))
        self.attention_mask = torch.ones_like(self.input_ids)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx],
        }


def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()

    for idx, data in enumerate(train_loader):

        input_ids, attention_mask, labels = (
            data["input_ids"].cuda(),
            data["attention_mask"].cuda(),
            data["labels"].cuda(),
        )

        optimizer.zero_grad()
        output = model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            chunk_head=False,
        )
        loss = output["loss"]
        loss.backward()
        optimizer.step()


def fsdp_main(rank, world_size, args):
    # 每个进程都要setup一下
    setup(rank, world_size)
    MOE_MANAGER.setup(seed=42, parallel=None, use_kernel_optim=False)

    dataset1 = RandomDataset()
    sampler1 = DistributedSampler(dataset1, rank=rank, num_replicas=world_size, shuffle=True)

    train_kwargs = {"batch_size": args.batch_size, "sampler": sampler1}
    cuda_kwargs = {"num_workers": 2, "shuffle": False}
    train_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    torch.cuda.set_device(rank)

    config = LlamaConfig.from_pretrained("hpcaitech/openmoe-base")
    setattr(config, "router_aux_loss_factor", 0.1)
    setattr(config, "router_z_loss_factor", 0.1)
    setattr(config, "label_smoothing", 0.1)
    setattr(config, "z_loss_factor", 0.1)
    model = OpenMoeForCausalLM(config).to(rank)
    # 使用FSDP将model warp起来
    model = FSDP(
        model,
        mixed_precision=MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        ),
    )

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(
            args,
            model,
            rank,
            world_size,
            train_loader,
            optimizer,
            epoch,
            sampler=sampler1,
        )


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
