import gzip
import random

import numpy as np
import torch
import torch.optim as optim
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

NUM_BATCHES = int(1000)
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


def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))


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


args = parse_args()
if args.distplan not in ["colossalai", "pytorch"]:
        raise TypeError(f"{args.distplan} is error")
disable_existing_loggers()
colossalai.launch_from_torch(config={})

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
        model = PaLM(num_tokens=256, dim=512, depth=8)
        model = AutoregressiveWrapper(model, max_seq_len=SEQ_LEN)

    pg = default_pg
    #tensor_parallelize(model, pg)
    model = gemini_zero_dpp(model, pg, args.placement)

    #optimizer

    #optimizer = GeminiAdamOptimizer(model, lr=1e-7, initial_scale=2**5)
    optimizer = GeminiAdamOptimizer(model, lr=LEARNING_RATE, initial_scale=2**5)
else:
    model = PaLM(num_tokens=256, dim=512, depth=8)
    model = AutoregressiveWrapper(model, max_seq_len=2048)
    model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)



# training
model.train()

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training"):

    if args.distplan == "colossalai":
        optimizer.zero_grad()

        loss = model(next(train_loader))
        # loss.backward()
        optimizer.backward(loss)

        print(f"training loss: {loss.item()}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        # optim.step()
        # optim.zero_grad()
        optimizer.step()
    else:
        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            loss = model(next(train_loader))
            loss.backward()

        print(f"training loss: {loss.item()}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

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