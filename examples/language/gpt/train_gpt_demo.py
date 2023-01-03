import os
from functools import partial
from time import time

import psutil
import torch
import torch.nn as nn
from model_zoo import model_builder
from packaging import version
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import get_data, get_tflops

import colossalai
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.parallel import ZeroDDP
from colossalai.tensor import ColoParameter, ComputePattern, ComputeSpec, ProcessGroup, ReplicaSpec, ShardSpec
from colossalai.utils import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext

CAI_VERSION = colossalai.__version__

if version.parse(CAI_VERSION) > version.parse("0.1.10"):
    # These are added after 0.1.10
    from colossalai.nn.optimizer.gemini_optimizer import GeminiAdamOptimizer
    from colossalai.nn.parallel import GeminiDDP
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size per DP group of training.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="gpt2_medium",
        help="model model scale",
    )
    args = parser.parse_args()
    return args


# Parameter Sharding Strategies for Tensor Parallelism
def split_param_single_dim_tp1d(dim: int, param: ColoParameter, pg: ProcessGroup):
    spec = (ShardSpec([dim], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    param.set_tensor_spec(*spec)


def split_param_row_tp1d(param: ColoParameter, pg: ProcessGroup):
    split_param_single_dim_tp1d(0, param, pg)


def split_param_col_tp1d(param: ColoParameter, pg: ProcessGroup):
    split_param_single_dim_tp1d(-1, param, pg)


class GPTLMLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2


def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024**2


def get_mem_info(prefix=''):
    return f'{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB'


def get_model_size(model: nn.Module):
    total_numel = 0
    for module in model.modules():
        for p in module.parameters(recurse=False):
            total_numel += p.numel()
    return total_numel


def model_size_formatter(numel: int) -> str:
    GB_SIZE = 10**9
    MB_SIZE = 10**6
    KB_SIZE = 10**3
    if numel >= GB_SIZE:
        return f'{numel / GB_SIZE:.1f}B'
    elif numel >= MB_SIZE:
        return f'{numel / MB_SIZE:.1f}M'
    elif numel >= KB_SIZE:
        return f'{numel / KB_SIZE:.1f}K'
    else:
        return str(numel)


def set_cpu_maximum_parallelism():
    conf_str = torch.__config__.parallel_info()
    inter_str = conf_str.split("hardware_concurrency() : ")[1]
    max_concurrency = inter_str.split('\n')[0]
    os.environ["OMP_NUM_THREADS"] = max_concurrency
    print(f"environmental variable OMP_NUM_THREADS is set to {max_concurrency}.")


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
def build_gemini(model: torch.nn.Module, pg: ProcessGroup, placement_policy: str = "auto"):
    fp16_init_scale = 2**5
    gpu_margin_mem_ratio_for_auto = 0

    if version.parse(CAI_VERSION) > version.parse("0.1.10"):
        model = GeminiDDP(model,
                          device=get_current_device(),
                          placement_policy=placement_policy,
                          pin_memory=True,
                          hidden_dim=model.config.n_embd,
                          search_range_mb=64)
        # configure the const policy
        if placement_policy == 'const':
            model.gemini_manager._placement_policy.set_const_memory_boundary(2 * 1024)
        # build a highly optimized cpu optimizer
        optimizer = GeminiAdamOptimizer(model,
                                        lr=1e-3,
                                        initial_scale=fp16_init_scale,
                                        gpu_margin_mem_ratio=gpu_margin_mem_ratio_for_auto)
    elif version.parse("0.1.9") <= version.parse(CAI_VERSION) <= version.parse("0.1.10"):
        from colossalai.gemini import ChunkManager, GeminiManager
        from colossalai.nn.optimizer import HybridAdam
        from colossalai.zero import ZeroOptimizer
        chunk_size = ChunkManager.search_chunk_size(model, 64 * 1024**2, 1024, filter_exlarge_params=True)
        chunk_manager = ChunkManager(chunk_size,
                                     pg,
                                     enable_distributed_storage=True,
                                     init_device=GeminiManager.get_default_device(placement_policy))
        gemini_manager = GeminiManager(placement_policy, chunk_manager)
        model = ZeroDDP(model, gemini_manager)
        optimizer = HybridAdam(model.parameters(), lr=1e-3)
        optimizer = ZeroOptimizer(optimizer,
                                  model,
                                  initial_scale=fp16_init_scale,
                                  gpu_margin_mem_ratio=gpu_margin_mem_ratio_for_auto)
    else:
        raise NotImplemented(f"CAI version {CAI_VERSION} is not supported")
    return model, optimizer


def main():
    # version check
    # this example is supposed to work for versions less than 0.2.0 but greater than 0.1.9
    assert version.parse(CAI_VERSION) < version.parse("0.2.0")
    assert version.parse(CAI_VERSION) >= version.parse("0.1.9")

    set_cpu_maximum_parallelism()
    args = parse_args()

    if args.distplan not in ["colossalai", "torch_ddp", "torch_zero", "zero1", "zero2"]:
        raise TypeError(f"{args.distplan} is error")

    # batch size per DP degree
    BATCH_SIZE = args.batch_size
    SEQ_LEN = 1024
    VOCAB_SIZE = 50257

    NUM_STEPS = 10
    WARMUP_STEPS = 1
    assert WARMUP_STEPS < NUM_STEPS, "warmup steps should smaller than the total steps"
    assert (NUM_STEPS - WARMUP_STEPS) % 2 == 1, "the number of valid steps should be odd to take the median "

    disable_existing_loggers()
    colossalai.launch_from_torch(config={})

    logger = get_dist_logger()
    logger.info(f"{args.model_type}, {args.distplan}, batch size {BATCH_SIZE}", ranks=[0])

    # build criterion
    criterion = GPTLMLoss()

    torch.manual_seed(123)
    if args.distplan == "colossalai":
        # all param must use the same process group.
        default_pg = ProcessGroup(tp_degree=args.tp_degree)
        default_dist_spec = ShardSpec([-1], [args.tp_degree]) if args.shardinit else None

        # build GPT model
        if version.parse(CAI_VERSION) > version.parse("0.1.10"):
            with ColoInitContext(device=get_current_device(),
                                 dtype=torch.half,
                                 default_dist_spec=default_dist_spec,
                                 default_pg=default_pg):
                model = model_builder(args.model_type)(checkpoint=True)
        else:
            with ColoInitContext(device=get_current_device()):
                model = model_builder(args.model_type)(checkpoint=True)

        pg = default_pg
        # Tensor Parallelism (TP)
        tensor_parallelize(model, pg)

        # build a Gemini model and a highly optimized cpu optimizer
        # Gemini + ZeRO DP, Note it must be used after TP
        model, optimizer = build_gemini(model, pg, args.placement)

        logger.info(get_mem_info(prefix='After init optim, '), ranks=[0])
    else:
        model = model_builder(args.model_type)(checkpoint=True).cuda()

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

    # model is shared after TP
    numel = get_model_size(model)
    logger.info(f"the size of testing model size is {model_size_formatter(numel)}.")
    logger.info(get_mem_info(prefix='After init model, '), ranks=[0])

    # Tflops_per_GPU = global_batch * global_numel * seq_len * 8 / #gpu
    # = (batch_per_DP_group * dp_degree) * (numel * tp_degree) * seq_len * 8 / (tp_degree * dp_degree)
    # = batch_per_DP_group * numel * seq_len * 8
    get_tflops_func = partial(get_tflops, numel, BATCH_SIZE, SEQ_LEN)

    torch.cuda.synchronize()
    model.train()
    tflops_list = []
    for n in range(NUM_STEPS):
        # we just use randomly generated data here
        input_ids, attn_mask = get_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        optimizer.zero_grad()

        start = time()
        outputs = model(input_ids, attn_mask)
        loss = criterion(outputs, input_ids)
        torch.cuda.synchronize()
        fwd_end = time()
        fwd_time = fwd_end - start
        logger.info(get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Forward '), ranks=[0])

        if args.distplan in ["colossalai", "zero1", "zero2"]:
            optimizer.backward(loss)
        elif args.distplan in ["torch_ddp", "torch_zero"]:
            loss.backward()
        torch.cuda.synchronize()
        bwd_end = time()
        bwd_time = bwd_end - fwd_end
        logger.info(get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Backward '), ranks=[0])

        if args.distplan in ["zero1", "zero2"]:
            optimizer.sync_grad()
        optimizer.step()
        torch.cuda.synchronize()
        optim_time = time() - bwd_end
        step_time = time() - start
        logger.info(get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Optimizer step '), ranks=[0])

        step_tflops = get_tflops_func(step_time)
        logger.info(
            f"[{n + 1}/{NUM_STEPS}] Loss:{loss.item():.3f}, Step time: {step_time:.3f}s, TFLOPS: {get_tflops_func(step_time):.3f}, FWD time: {fwd_time:.3f}s, BWD time: {bwd_time:.3f}s, OPTIM time: {optim_time:.3f}s",
            ranks=[0],
        )
        if n >= WARMUP_STEPS:
            tflops_list.append(step_tflops)

    tflops_list.sort()
    median_index = ((NUM_STEPS - WARMUP_STEPS) >> 1) + WARMUP_STEPS
    logger.info(f"Median TFLOPS is {tflops_list[median_index]:.3f}")
    torch.cuda.synchronize()


if __name__ == '__main__':
    main()
