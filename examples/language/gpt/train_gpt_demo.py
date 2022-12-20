from functools import partial
from time import time

import psutil
import torch
import torch.nn as nn
from packaging import version
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Config, GPT2LMHeadModel

import colossalai
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.nn.optimizer.gemini_optimizer import GeminiAdamOptimizer
from colossalai.nn.optimizer.zero_optimizer import ZeroOptimizer
from colossalai.nn.parallel import ZeroDDP
from colossalai.tensor import ColoParameter, ComputePattern, ComputeSpec, ProcessGroup, ReplicaSpec, ShardSpec
from colossalai.utils import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.zero.sharded_optim import LowLevelZeroOptimizer


def parse_args():
    parser = colossalai.get_default_parser()
    parser.add_argument(
        "--distplan",
        type=str,
        default='colossalai',
        help="The distributed plan [colossalai, zero1, zero2, torch_ddp, torch_zero].",
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
    args = parser.parse_args()
    return args


## Parameter Sharding Strategies for Tensor Parallelism
def split_param_single_dim_tp1d(dim: int, param: ColoParameter, pg: ProcessGroup):
    spec = (ShardSpec([dim], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    param.set_tensor_spec(*spec)


def split_param_row_tp1d(param: ColoParameter, pg: ProcessGroup):
    split_param_single_dim_tp1d(0, param, pg)


def split_param_col_tp1d(param: ColoParameter, pg: ProcessGroup):
    split_param_single_dim_tp1d(-1, param, pg)


## Define the Model and Loss Based on Huggingface transformers GPT2LMHeadModel
class GPTLMModel(nn.Module):

    def __init__(self,
                 hidden_size=768,
                 num_layers=12,
                 num_attention_heads=12,
                 max_seq_len=1024,
                 vocab_size=50257,
                 checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = GPT2LMHeadModel(
            GPT2Config(n_embd=hidden_size,
                       n_layer=num_layers,
                       n_head=num_attention_heads,
                       n_positions=max_seq_len,
                       n_ctx=max_seq_len,
                       vocab_size=vocab_size))
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        # Only return lm_logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]


class GPTLMLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


## Randomly Generated Data
def get_data(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=torch.cuda.current_device())
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def gpt2_medium(checkpoint=False):
    return GPTLMModel(hidden_size=1024, num_layers=24, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_xl(checkpoint=True):
    return GPTLMModel(hidden_size=1600, num_layers=48, num_attention_heads=32, checkpoint=checkpoint)


def gpt2_10b(checkpoint=True):
    return GPTLMModel(hidden_size=4096, num_layers=50, num_attention_heads=16, checkpoint=checkpoint)


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2


def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024**2


def get_mem_info(prefix=''):
    return f'{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB'


def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)


# Tensor Parallel
def tensor_parallelize(model: torch.nn.Module, pg: ProcessGroup):
    """tensor_parallelize
    Sharding the Model Parameters.

    Args:
        model (torch.nn.Module): a torch module to be sharded
    """
    for mn, module in model.named_modules():
        for pn, param in module.named_parameters(recurse=False):
            # NOTE() a param maybe shared by tow modules
            if hasattr(param, 'visited'):
                continue
            param.set_dist_spec(ReplicaSpec())
            if 'mlp.c_fc' in mn:
                if 'weight' in pn or 'bias' in pn:
                    split_param_col_tp1d(param, pg)    # colmn slice
                    # keep the shape of the output from c_fc
                    param.compute_spec.set_output_replicate(False)
                else:
                    param.set_dist_spec(ReplicaSpec())
            elif 'mlp.c_proj' in mn:
                if 'weight' in pn:
                    split_param_row_tp1d(param, pg)    # row slice
                else:
                    param.set_dist_spec(ReplicaSpec())
            elif 'wte' in mn or 'wpe' in mn:
                split_param_col_tp1d(param, pg)    # colmn slice
            elif 'c_attn' in mn or 'c_proj' in mn:
                split_param_col_tp1d(param, pg)    # colmn slice
            else:
                param.set_dist_spec(ReplicaSpec())

            param.visited = True


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


def main():
    args = parse_args()

    if args.distplan not in ["colossalai", "torch_ddp", "torch_zero", "zero1", "zero2"]:
        raise TypeError(f"{args.distplan} is error")

    BATCH_SIZE = 8
    SEQ_LEN = 1024
    VOCAB_SIZE = 50257
    NUM_STEPS = 10

    disable_existing_loggers()
    colossalai.launch_from_torch(config={})

    logger = get_dist_logger()
    logger.info(f"using dist plan {args.distplan}", ranks=[0])

    # build criterion
    criterion = GPTLMLoss()

    torch.manual_seed(123)
    if args.distplan == "colossalai":
        # all param must use the same process group.
        default_pg = ProcessGroup(tp_degree=args.tp_degree)
        default_dist_spec = ShardSpec([-1], [args.tp_degree]) if args.shardinit else None

        # build GPT model
        with ColoInitContext(device='cpu', default_dist_spec=default_dist_spec, default_pg=default_pg):
            model = gpt2_medium(checkpoint=True)

        pg = default_pg
        # Tensor Parallelism (TP)
        tensor_parallelize(model, pg)
        # Gemini + ZeRO DP, Note it must be used after TP
        model = gemini_zero_dpp(model, pg, args.placement)

        # build optimizer
        optimizer = GeminiAdamOptimizer(model, lr=1e-3, initial_scale=2**5)
        # optimizer = HybridAdam(model.parameters(), lr=1e-3)
        # optimizer = ZeroOptimizer(optimizer, model, initial_scale=2**5)
        logger.info(get_mem_info(prefix='After init optim, '), ranks=[0])
    else:
        model = gpt2_medium(checkpoint=True).cuda()

    if args.distplan.startswith("torch"):
        model = DDP(model)
        if args.distplan.endswith("ddp"):
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        elif args.distplan.endswith("zero"):
            from torch.distributed.optim import ZeroRedundancyOptimizer
            optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.Adam, lr=0.01)
    elif args.distplan.startswith("zero"):
        partition_flag = args.distplan == "zero2"
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer = LowLevelZeroOptimizer(optimizer,
                                          overlap_communication=True,
                                          partition_grad=partition_flag,
                                          verbose=True)
        # notice that the model is still in fp32

    numel = sum([p.numel() for p in model.parameters()])
    logger.info(get_mem_info(prefix='After init model, '), ranks=[0])
    get_tflops_func = partial(get_tflops, numel, BATCH_SIZE, SEQ_LEN)

    torch.cuda.synchronize()
    model.train()
    for n in range(NUM_STEPS):
        # we just use randomly generated data here
        input_ids, attn_mask = get_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        optimizer.zero_grad()
        start = time()
        outputs = model(input_ids, attn_mask)
        loss = criterion(outputs, input_ids)
        logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Forward '), ranks=[0])
        if args.distplan in ["colossalai", "zero1", "zero2"]:
            optimizer.backward(loss)
        elif args.distplan in ["torch_ddp", "torch_zero"]:
            loss.backward()
        logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Backward '), ranks=[0])
        if args.distplan in ["zero1", "zero2"]:
            optimizer.sync_grad()
        optimizer.step()
        logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Optimizer step '), ranks=[0])
        step_time = time() - start
        logger.info(
            f'[{n+1}/{NUM_STEPS}] Loss:{loss.item():.3f}, Step time: {step_time:.3f}s, TFLOPS: {get_tflops_func(step_time):.3f}',
            ranks=[0])

    torch.cuda.synchronize()


if __name__ == '__main__':
    main()
