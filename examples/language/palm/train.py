import gzip
import random
from time import time
from functools import partial
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import tqdm
from packaging import version
from palm_pytorch import PaLM
from palm_pytorch.autoregressive_wrapper import AutoregressiveWrapper
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import colossalai
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer.gemini_optimizer import GeminiAdamOptimizer
from colossalai.nn.parallel import GeminiDDP, ZeroDDP
from colossalai.tensor import ColoParameter, ComputePattern, ComputeSpec, ProcessGroup, ReplicaSpec, ShardSpec
from colossalai.utils import MultiTimer, get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext

# constants

NUM_BATCHES = int(100)
WARMUP_BATCHES = 1
GRADIENT_ACCUMULATE_EVERY = 1
LEARNING_RATE = 2e-4
VALIDATE_EVERY = 100
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 1024


def parse_args():
    parser = colossalai.get_default_parser()
    parser.add_argument(
        "--distplan",
        type=str,
        default='colossalai',
        help="The distributed plan [colossalai, pytorch].",
    )
    parser.add_argument(
        "--tp_degree",
        type=int,
        default=1,
        help="Tensor Parallelism Degree. Valid when using colossalai as dist plan.",
    )
    parser.add_argument(
        "--placement",
        type=str,
        default='cpu',
        help="Placement Policy for Gemini. Valid when using colossalai as dist plan.",
    )
    parser.add_argument(
        "--shardinit",
        type=bool,
        default=False,
        help=
        "Shard the tensors when init the model to shrink peak memory size on the assigned device. Valid when using colossalai as dist plan.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size per DP group of training.",
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

# Gemini + ZeRO DDP
def gemini_zero_dpp(model: torch.nn.Module, pg: ProcessGroup, placememt_policy: str = "auto"):
    cai_version = colossalai.__version__
    if version.parse(cai_version) > version.parse("0.1.10"):
        from colossalai.nn.parallel import GeminiDDP
        model = GeminiDDP(model,
                          device=get_current_device(),
                          placement_policy=placememt_policy,
                          pin_memory=True,
                          search_range_mb=32)
    elif version.parse(cai_version) <= version.parse("0.1.10") and version.parse(cai_version) >= version.parse("0.1.9"):
        from colossalai.gemini import ChunkManager, GeminiManager
        chunk_size = ChunkManager.search_chunk_size(model, 64 * 1024**2, 32)
        gemini_manager = GeminiManager(placememt_policy, chunk_manager)
        chunk_manager = ChunkManager(chunk_size,
                                     pg,
                                     enable_distributed_storage=True,
                                     init_device=GeminiManager.get_default_device(placememt_policy))
        model = ZeroDDP(model, gemini_manager)
    else:
        raise NotImplemented(f"CAI version {cai_version} is not supported")
    return model

## Parameter Sharding Strategies for Tensor Parallelism
def split_param_single_dim_tp1d(dim: int, param: ColoParameter, pg: ProcessGroup):
    spec = (ShardSpec([dim], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    param.set_tensor_spec(*spec)


def split_param_row_tp1d(param: ColoParameter, pg: ProcessGroup):
    split_param_single_dim_tp1d(0, param, pg)


def split_param_col_tp1d(param: ColoParameter, pg: ProcessGroup):
    split_param_single_dim_tp1d(-1, param, pg)

# Tensor Parallel
def tensor_parallelize(model: torch.nn.Module, pg: ProcessGroup):
    """tensor_parallelize
    Sharding the Model Parameters.
    Args:
        model (torch.nn.Module): a torch module to be sharded
    """
    for mn, module in model.named_modules():
        for pn, param in module.named_parameters(recurse=False):
            if hasattr(param, 'visited'):
                continue
            param.set_dist_spec(ReplicaSpec())
            if 'net.0' in mn:
                split_param_col_tp1d(param, pg)    # colmn slice
            elif 'to_q' in mn:
                split_param_col_tp1d(param, pg)    # colmn slice
            elif 'to_kv' in mn:
                split_param_row_tp1d(param, pg)    # row slice
            elif 'to_out' in mn:
                split_param_row_tp1d(param, pg)    # row slice
            elif '1.1' in mn:
                split_param_col_tp1d(param, pg)    # colmn slice
            elif '1.2' in mn:
                split_param_row_tp1d(param, pg)    # row slice
            else:
                param.set_dist_spec(ReplicaSpec())
            param.visited = True


args = parse_args()
if args.distplan not in ["colossalai", "pytorch"]:
        raise TypeError(f"{args.distplan} is error")
disable_existing_loggers()
colossalai.launch_from_torch(config={})
logger = get_dist_logger()

with gzip.open("./data/enwik8.gz") as file:
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
    trX, vaX = np.split(X, [int(90e6)])
    data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)


class TextSamplerDataset(Dataset):

    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start:rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len


train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=args.batch_size))
val_loader = cycle(DataLoader(val_dataset, batch_size=args.batch_size))

if args.distplan == "colossalai":
    # instantiate GPT-like decoder model

    default_pg = ProcessGroup(tp_degree=args.tp_degree)
    default_dist_spec = ShardSpec([-1], [args.tp_degree]) if args.shardinit else None
    ctx = ColoInitContext(device='cpu', default_dist_spec=default_dist_spec, default_pg=default_pg)

    with ctx:
        model = PaLM(num_tokens=50304, dim=4096, depth=64)
        model = AutoregressiveWrapper(model, max_seq_len=SEQ_LEN)

    pg = default_pg
    tensor_parallelize(model, pg)
    model = gemini_zero_dpp(model, pg, args.placement)

    #optimizer

    #optimizer = GeminiAdamOptimizer(model, lr=1e-7, initial_scale=2**5)
    optimizer = GeminiAdamOptimizer(model, lr=LEARNING_RATE, initial_scale=2**5)
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