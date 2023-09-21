import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
from model.modeling_openmoe import LlamaConfig, OpenMoeForCausalLM
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import Adafactor
from transformers.models.llama import LlamaConfig

from colossalai.moe.manager import MOE_MANAGER

from .utils import PerformanceEvaluator, get_model_numel


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


def fsdp_main(rank, world_size, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "14523"
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    MOE_MANAGER.setup(seed=42, parallel=None, use_kernel_optim=False)

    dataset = RandomDataset(max_length=args.seq_length)
    sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=False)
    train_kwargs = {"batch_size": args.batch_size, "sampler": sampler}
    train_loader = torch.utils.data.DataLoader(dataset, **train_kwargs)
    torch.cuda.set_device(rank)

    config = LlamaConfig.from_pretrained("hpcaitech/openmoe-%s" % args.model_name)
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
    optimizer = Adafactor(model.parameters())
    model.train()

    model_numel = get_model_numel(model)
    performance_evaluator = PerformanceEvaluator(
        model_numel,
        enable_grad_checkpoint=True,
        ignore_steps=args.warm_up,
        dp_world_size=dist.get_world_size(),
    )

    for step, data in tqdm.tqdm(enumerate(train_loader), total=args.warm_up + args.active):
        if step == args.warm_up + args.active:
            break

        performance_evaluator.on_step_start(step)
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
        performance_evaluator.on_step_end(input_ids)

    performance_evaluator.on_fit_end()
    if dist.get_rank() == 0:
        print(f"Max CUDA memory usage: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="base",
        choices=["base", "8b"],
        help="base or 8b",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--warm_up", type=int, default=20)
    parser.add_argument("--active", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(42)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
