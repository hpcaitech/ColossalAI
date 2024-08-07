import argparse
import functools
import os

import torch
import torch.distributed as dist
import tqdm
from model.modeling_openmoe import LlamaConfig, OpenMoeDecoderLayer, OpenMoeForCausalLM, set_openmoe_args
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers.models.llama import LlamaConfig
from utils import PerformanceEvaluator, get_model_numel

from colossalai.legacy.moe.manager import MOE_MANAGER


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
    # initialize the process group

    # initialize the process group
    dist.init_process_group("nccl")

    MOE_MANAGER.setup(parallel=None)

    dp_size = dist.get_world_size()
    dataset = RandomDataset(
        max_length=args.seq_length,
        num_samples=args.batch_size * (args.warmup + args.active) * dp_size,
    )
    sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=False)
    train_kwargs = {"batch_size": args.batch_size, "sampler": sampler}
    train_loader = torch.utils.data.DataLoader(dataset, **train_kwargs)
    torch.cuda.set_device(rank)

    config = LlamaConfig.from_pretrained("hpcai-tech/openmoe-%s" % args.model_name)
    set_openmoe_args(
        config,
        num_experts=config.num_experts,
        moe_layer_interval=config.moe_layer_interval,
        enable_load_balance=False,
        enable_kernel=False,
        enable_comm_overlap=False,
    )
    torch.set_default_dtype(torch.float16)
    model = OpenMoeForCausalLM(config)
    torch.set_default_dtype(torch.float32)
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            OpenMoeDecoderLayer,
        },
    )
    model = FSDP(
        model,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.cuda.current_device(),
    )
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01, lr=1e-5)
    model.train()

    model_numel = get_model_numel(model)
    performance_evaluator = PerformanceEvaluator(
        model_numel,
        enable_grad_checkpoint=True,
        ignore_steps=args.warmup,
        dp_world_size=dist.get_world_size(),
    )

    for step, data in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
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
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--active", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(42)

    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    fsdp_main(local_rank, world_size, args)
