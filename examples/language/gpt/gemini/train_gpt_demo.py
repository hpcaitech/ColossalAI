import os
from functools import partial
from time import time

import psutil
import torch
import torch.nn as nn
from commons.model_zoo import model_builder
from commons.utils import get_data, get_profile_context, get_tflops, get_time_stamp
from packaging import version
from torch.nn.parallel import DistributedDataParallel as DDP

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.tensor import ColoParameter, ComputePattern, ComputeSpec, ProcessGroup, ReplicaSpec, ShardSpec
from colossalai.utils import get_current_device
from colossalai.zero import ColoInitContext

CAI_VERSION = colossalai.__version__


def parse_args():
    parser = colossalai.get_default_parser()
    parser.add_argument(
        "--distplan",
        type=str,
        default='CAI_Gemini',
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
        action='store_true',
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
    parser.add_argument(
        "--train_step",
        type=int,
        default=10,
        help="training iterations for test",
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
            # NOTE() a param maybe shared by two modules
            if hasattr(param, 'visited'):
                continue

            # if shard init, then convert param to replica and use the dp-only ProcessGroup
            param: ColoParameter = param
            param.set_dist_spec(ReplicaSpec())
            param.set_process_group(pg)

            # shard it w.r.t tp pattern
            if 'mlp.c_fc' in mn:
                if 'weight' in pn or 'bias' in pn:
                    split_param_col_tp1d(param, pg)    # column slice
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
                split_param_col_tp1d(param, pg)    # column slice
            elif 'c_attn' in mn or 'c_proj' in mn:
                split_param_col_tp1d(param, pg)    # column slice
            else:
                param.set_dist_spec(ReplicaSpec())
            param.visited = True


def main():
    # version check
    # this example is supposed to work for versions greater than 0.2.0
    assert version.parse(CAI_VERSION) >= version.parse("0.2.0")

    set_cpu_maximum_parallelism()
    args = parse_args()

    # if args.distplan not in ["colossalai", "torch_ddp", "torch_zero", "zero1", "zero2"]:
    if args.distplan not in ["CAI_ZeRO1", "CAI_ZeRO2", "CAI_Gemini", "Pytorch_DDP", "Pytorch_ZeRO"]:
        raise TypeError(f"{args.distplan} is error")

    # batch size per DP degree
    BATCH_SIZE = args.batch_size
    SEQ_LEN = 1024
    VOCAB_SIZE = 50257

    NUM_STEPS = args.train_step

    WARMUP_STEPS = 1
    assert WARMUP_STEPS < NUM_STEPS, "warmup steps should smaller than the total steps"
    assert (NUM_STEPS - WARMUP_STEPS) % 2 == 1, "the number of valid steps should be odd to take the median"
    PROF_FLAG = False    # The flag of profiling, False by default

    disable_existing_loggers()
    colossalai.launch_from_torch(config={})

    logger = get_dist_logger()
    logger.info(f"{args.model_type}, {args.distplan}, batch size {BATCH_SIZE}", ranks=[0])

    # build criterion
    criterion = GPTLMLoss()

    torch.manual_seed(123)
    if args.distplan.startswith("CAI"):
        # all param must use the same process group.
        world_size = torch.distributed.get_world_size()
        shard_pg = ProcessGroup(tp_degree=world_size) if args.shardinit else None
        default_dist_spec = ShardSpec([-1], [world_size]) if args.shardinit else None

        if args.shardinit and args.distplan != "CAI_Gemini":
            raise RuntimeError("You can only use shardinit with CAI_Gemini")

        # build GPT model
        with ColoInitContext(device=get_current_device(),
                             dtype=torch.half,
                             default_dist_spec=default_dist_spec,
                             default_pg=shard_pg):
            model = model_builder(args.model_type)(checkpoint=True)

        tp_pg = ProcessGroup(tp_degree=args.tp_degree)
        # Tensor Parallelism (TP)
        # You should notice that v0.1.10 is not compatible with TP degree > 1
        if args.tp_degree > 1:
            tensor_parallelize(model, tp_pg)

        # assign running configurations
        if args.distplan == "CAI_ZeRO1":
            zero_stage = 1
        elif args.distplan == "CAI_ZeRO2":
            zero_stage = 2
        elif args.distplan == "CAI_Gemini":
            zero_stage = 3
        else:
            raise RuntimeError

        plugin = None
        if args.distplan.startswith("CAI_ZeRO"):
            plugin = LowLevelZeroPlugin(stage=zero_stage,
                                        reduce_bucket_size_in_m=12,
                                        overlap_communication=True,
                                        verbose=True)
        elif args.distplan == "CAI_Gemini":
            plugin = GeminiPlugin(device=get_current_device(),
                                  placement_policy=args.placement,
                                  pin_memory=True,
                                  strict_ddp_mode=args.tp_degree == 1,
                                  search_range_m=128,
                                  hidden_dim=model.config.n_embd,
                                  gpu_margin_mem_ratio=0.)
        else:
            raise RuntimeError

        # build a highly optimized gpu/cpu optimizer
        optimizer = HybridAdam(model.parameters(), lr=1e-3)

        logger.info(get_mem_info(prefix='After init optim, '), ranks=[0])
    elif args.distplan.startswith("Pytorch"):
        assert args.tp_degree == 1, "The degree of TP should be 1 for DDP examples."
        model = model_builder(args.model_type)(checkpoint=True).cuda()
        plugin = TorchDDPPlugin()
        if args.distplan.endswith("DDP"):
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        elif args.distplan.endswith("ZeRO"):
            from torch.distributed.optim import ZeroRedundancyOptimizer
            optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.Adam, lr=1e-3)

    else:
        raise RuntimeError
    # wrap your model and optimizer
    booster = Booster(plugin=plugin)
    model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)

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

    def train_step():
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
        booster.backward(loss, optimizer)

        torch.cuda.synchronize()
        bwd_end = time()
        bwd_time = bwd_end - fwd_end
        logger.info(get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Backward '), ranks=[0])

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

    demo_profiler = get_profile_context(PROF_FLAG,
                                        WARMUP_STEPS,
                                        NUM_STEPS - WARMUP_STEPS,
                                        save_dir=f"profile/{get_time_stamp()}-demo")

    with demo_profiler as prof:
        for n in range(NUM_STEPS):
            train_step()
            prof.step()

    tflops_list.sort()
    median_index = ((NUM_STEPS - WARMUP_STEPS) >> 1) + WARMUP_STEPS
    logger.info(f"Median TFLOPS is {tflops_list[median_index]:.3f}")
    torch.cuda.synchronize()


if __name__ == '__main__':
    main()
