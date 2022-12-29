import gzip
import random

import numpy as np
import torch
import torch.optim as optim
import tqdm
from palm_pytorch import PaLM
from palm_pytorch.autoregressive_wrapper import AutoregressiveWrapper
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from packaging import version

import colossalai
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.tensor import ColoParameter, ComputePattern, ComputeSpec, ProcessGroup, ReplicaSpec, ShardSpec
from colossalai.utils import MultiTimer, get_current_device
from colossalai.nn.parallel import ZeroDDP
from colossalai.nn.optimizer.gemini_optimizer import GeminiAdamOptimizer
from colossalai.nn.parallel import GeminiDDP
from colossalai.logging import disable_existing_loggers, get_dist_logger

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY = 100
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 1024
TPDEGREE = 2
USE_SHARD_INIT = False
placement = 'cpu'

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
    
# instantiate GPT-like decoder model

parser = colossalai.get_default_parser()
args = parser.parse_args()
disable_existing_loggers()
colossalai.launch_from_torch(config=args.config, seed=42)


# instantiate GPT-like decoder model

default_pg = ProcessGroup(tp_degree=TPDEGREE)
default_dist_spec = ShardSpec([-1], [TPDEGREE]) if USE_SHARD_INIT else None
ctx = ColoInitContext(device='cpu', default_dist_spec=default_dist_spec, default_pg=default_pg)
with ctx:
        model = PaLM(num_tokens=256,dim=512,depth=8)
        model = AutoregressiveWrapper(model, max_seq_len=SEQ_LEN)
        model.cuda()


# prepare enwik8 data

# model = PaLM(num_tokens=256, dim=512, depth=8)

# model = AutoregressiveWrapper(model, max_seq_len=SEQ_LEN)
# model.cuda()

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
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE))

#tensor_parallelize(model, pg)

pg = default_pg
# model = GeminiDDP(model,
#                             device=get_current_device(),
#                           placement_policy="auto",
#                           pin_memory=True,
#                           search_range_mb=32)
model = gemini_zero_dpp(model, pg, placement)

#optimizer

optimizer = GeminiAdamOptimizer(model, lr=1e-7, initial_scale=2**5)
#optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training"):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_loader))
        loss.backward()

    print(f"training loss: {loss.item()}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    # optim.step()
    # optim.zero_grad()
    optimizer.step()
    optimizer.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = model(next(val_loader))
            print(f"validation loss: {loss.item()}")

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:-1]
        prime = decode_tokens(inp)
        print(f"%s \n\n %s", (prime, "*" * 100))

        sample = model.generate(inp[None, ...], GENERATE_LENGTH)
        output_str = decode_tokens(sample[0])
        print(output_str)
