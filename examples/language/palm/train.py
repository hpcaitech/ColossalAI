import argparse
import gzip
from contextlib import nullcontext
from functools import partial
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from palm_pytorch import PaLM
from palm_pytorch.autoregressive_wrapper import AutoregressiveWrapper
from torch.utils.data import DataLoader, Dataset

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.lazy import LazyInitContext
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn import HybridAdam

# constants

NUM_BATCHES = int(10)
WARMUP_BATCHES = 1
GRADIENT_ACCUMULATE_EVERY = 1
LEARNING_RATE = 2e-4
VALIDATE_EVERY = 100
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 1024


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distplan",
        type=str,
        default="colossalai",
        help="The distributed plan [colossalai, pytorch].",
    )
    parser.add_argument(
        "--offload_optim_frac",
        type=float,
        default=1.0,
        help="Fraction of optimizer states to be offloaded. This is only used for gemini.",
    )
    parser.add_argument(
        "-p",
        "--plugin",
        type=str,
        default="torch_ddp",
        choices=["torch_ddp", "torch_ddp_fp16", "gemini", "low_level_zero"],
        help="plugin to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size per DP group of training.",
    )
    parser.add_argument(
        "--dummy_data",
        type=bool,
        default=False,
        help="use dummy dataset.",
    )
    args = parser.parse_args()
    return args


# helpers
def cycle(loader):
    while True:
        for data in loader:
            yield data


def decode_token(token):
    return str(chr(max(32, token)))


def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)


def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))


def get_model_size(model: nn.Module):
    total_numel = 0
    for module in model.modules():
        for p in module.parameters(recurse=False):
            total_numel += p.numel()
    return total_numel


args = parse_args()
if args.distplan not in ["colossalai", "pytorch"]:
    raise TypeError(f"{args.distplan} is error")
disable_existing_loggers()
colossalai.launch_from_torch()
logger = get_dist_logger()


def generate_dataset(dummy_data: bool = False):
    if not dummy_data:
        with gzip.open("./data/enwik8.gz") as file:
            X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
            trX, vaX = np.split(X, [int(90e6)])
            data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)
            # print(f"data_train {data_train.shape} {data_train.dtype} {max(data_train)} {min(data_train)}")
            # print(f"data_val {data_val.shape} {data_val.dtype}  {max(data_val)} {min(data_val)}")
            return data_train, data_val
    else:
        return torch.randint(0, 100, (90000000,)), torch.randint(0, 100, (5000000,))


data_train, data_val = generate_dataset(args.dummy_data)

print("generate dataset ready!")


class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len


train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=args.batch_size))
val_loader = cycle(DataLoader(val_dataset, batch_size=args.batch_size))

if args.distplan == "colossalai":
    # instantiate GPT-like decoder model

    booster_kwargs = {}
    if args.plugin == "torch_ddp_fp16":
        booster_kwargs["mixed_precision"] = "fp16"
    if args.plugin.startswith("torch_ddp"):
        plugin = TorchDDPPlugin()
    elif args.plugin == "gemini":
        plugin = GeminiPlugin(offload_optim_frac=args.offload_optim_frac, initial_scale=2**5)
    elif args.plugin == "low_level_zero":
        plugin = LowLevelZeroPlugin(initial_scale=2**5)
    logger.info(f"plugin: {plugin}")
    booster = Booster(plugin=plugin, **booster_kwargs)

    ctx = (
        LazyInitContext(default_device=get_accelerator().get_current_device())
        if args.plugin == "gemini"
        else nullcontext()
    )

    with ctx:
        model = PaLM(num_tokens=50304, dim=4096, depth=64)
        model = AutoregressiveWrapper(model, max_seq_len=SEQ_LEN)

    # optimizer

    optimizer = HybridAdam(model.parameters(), lr=LEARNING_RATE, initial_scale=2**5)
    model, optimizer, _, _, _ = booster.boost(model, optimizer)

else:
    model = PaLM(num_tokens=256, dim=512, depth=8)
    model = AutoregressiveWrapper(model, max_seq_len=2048)
    model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# model is shared after TP
numel = get_model_size(model)
get_tflops_func = partial(get_tflops, numel, args.batch_size, SEQ_LEN)

# training
model.train()
tflops_list = []
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training"):
    if args.distplan == "colossalai":
        optimizer.zero_grad()
        start = time()
        loss = model(next(train_loader))
        fwd_end = time()
        fwd_time = fwd_end - start
        # loss.backward()
        optimizer.backward(loss)
        bwd_end = time()
        bwd_time = bwd_end - fwd_end

        # print(f"training loss: {loss.item()}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        # optim.step()
        # optim.zero_grad()
        optimizer.step()
        optim_time = time() - bwd_end
        step_time = time() - start

        step_tflops = get_tflops_func(step_time)
        logger.info(
            f"[{i + 1}/{NUM_BATCHES}] Loss:{loss.item():.3f}, Step time: {step_time:.3f}s, TFLOPS: {get_tflops_func(step_time):.3f}, FWD time: {fwd_time:.3f}s, BWD time: {bwd_time:.3f}s, OPTIM time: {optim_time:.3f}s",
            ranks=[0],
        )
        if i >= WARMUP_BATCHES:
            tflops_list.append(step_tflops)

    else:
        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            loss = model(next(train_loader))
            loss.backward()

        print(f"training loss: {loss.item()}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

tflops_list.sort()
median_index = ((NUM_BATCHES - WARMUP_BATCHES) >> 1) + WARMUP_BATCHES
logger.info(f"Median TFLOPS is {tflops_list[median_index]:.3f}")

# TODO
# if i % VALIDATE_EVERY == 0:
#     model.eval()
#     with torch.no_grad():
#         loss = model(next(val_loader))
#         print(f"validation loss: {loss.item()}")

# if i % GENERATE_EVERY == 0:
#     model.eval()
#     inp = random.choice(val_dataset)[:-1]
#     prime = decode_tokens(inp)
#     print(f"%s \n\n %s", (prime, "*" * 100))

#     sample = model.generate(inp[None, ...], GENERATE_LENGTH)
#     output_str = decode_tokens(sample[0])
#     print(output_str)
