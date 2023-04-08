import os
from functools import partial
from time import time

import psutil
import torch
from packaging import version
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AlbertConfig, AlbertForSequenceClassification, BertConfig, BertForSequenceClassification

import colossalai
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.tensor import ColoParameter, ComputePattern, ComputeSpec, ProcessGroup, ReplicaSpec, ShardSpec
from colossalai.utils import get_current_device
from colossalai.zero import ColoInitContext, zero_model_wrapper, zero_optim_wrapper

CAI_VERSION = colossalai.__version__


def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)


def get_profile_context(enable_flag, warmup_steps, active_steps, save_dir):
    from contextlib import nullcontext

    from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
    if enable_flag:
        return profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                       schedule=schedule(wait=0, warmup=warmup_steps, active=active_steps),
                       on_trace_ready=tensorboard_trace_handler(save_dir),
                       record_shapes=True,
                       profile_memory=True)
    else:

        class DummyProfiler:

            def __init__(self):
                self.step_number = 0

            def step(self):
                self.step_number += 1

        return nullcontext(DummyProfiler())


def get_time_stamp():
    import time
    cur_time = time.strftime("%d-%H:%M", time.localtime())
    return cur_time


def get_bert_data(batch_size: int, sequence_length: int, vacob_size: int, n_class: int, device: torch.device):
    input = torch.randint(
        low=0,
        high=vacob_size,
        size=(batch_size, sequence_length),
        device=device,
        dtype=torch.long,
    )
    label = torch.randint(low=0, high=n_class, size=(batch_size,), device=device, dtype=torch.long)
    return input, label


def parse_args():
    parser = colossalai.get_default_parser()
    parser.add_argument(
        "--distplan",
        type=str,
        default='CAI_Gemini',
        help="The distributed plan [colossalai, zero1, zero2, torch_ddp, torch_zero].",
    )
    parser.add_argument(
        "--placement",
        type=str,
        default='cpu',
        help="Placement Policy for Gemini. Valid when using colossalai as dist plan.",
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
        default="bert",
        help="bert or albert",
    )
    parser.add_argument(
        "--train_step",
        type=int,
        default=10,
        help="training iterations for test",
    )

    args = parser.parse_args()
    return args


SEQ_LEN = 512
VOCAB_SIZE = 1000
NUM_LABELS = 10


# Parameter Sharding Strategies for Tensor Parallelism
def split_param_single_dim_tp1d(dim: int, param: ColoParameter, pg: ProcessGroup):
    spec = (ShardSpec([dim], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    param.set_tensor_spec(*spec)


def split_param_row_tp1d(param: ColoParameter, pg: ProcessGroup):
    split_param_single_dim_tp1d(0, param, pg)


def split_param_col_tp1d(param: ColoParameter, pg: ProcessGroup):
    split_param_single_dim_tp1d(-1, param, pg)


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


def model_builder(args):
    if args.model_type == "bert":
        cfg = BertConfig(vocab_size=VOCAB_SIZE, num_labels=NUM_LABELS)
        return BertForSequenceClassification(cfg)
    elif args.model_type == "albert":
        cfg = AlbertConfig(vocab_size=VOCAB_SIZE, num_labels=NUM_LABELS)
        return AlbertForSequenceClassification(cfg)
    else:
        raise RuntimeError


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

    NUM_STEPS = args.train_step

    WARMUP_STEPS = 1
    assert WARMUP_STEPS < NUM_STEPS, "warmup steps should smaller than the total steps"
    assert (NUM_STEPS - WARMUP_STEPS) % 2 == 1, "the number of valid steps should be odd to take the median"
    PROF_FLAG = False    # The flag of profiling, False by default

    disable_existing_loggers()
    colossalai.launch_from_torch(config={})

    logger = get_dist_logger()
    logger.info(f" {args.distplan}, batch size {BATCH_SIZE}", ranks=[0])

    torch.manual_seed(123)
    if args.distplan.startswith("CAI"):
        # all param must use the same process group.
        world_size = torch.distributed.get_world_size()

        # build a base-bert model
        with ColoInitContext(device=get_current_device(), dtype=torch.half):
            model = model_builder(args)
            # model = BertForSequenceClassification(BertConfig(vocal_size =  VOCAB_SIZE))

        # asign running configurations
        gemini_config = None
        if args.distplan.startswith("CAI_ZeRO"):
            optim_config = dict(reduce_bucket_size=12 * 1024 * 1024, overlap_communication=True, verbose=True)
        elif args.distplan == "CAI_Gemini":
            gemini_config = dict(strict_ddp_mode=True,
                                 device=get_current_device(),
                                 placement_policy=args.placement,
                                 pin_memory=True,
                                 hidden_dim=model.config.hidden_size,
                                 search_range_mb=128)
            optim_config = dict(gpu_margin_mem_ratio=0.)
        else:
            raise RuntimeError

        # build a highly optimized gpu/cpu optimizer
        optimizer = HybridAdam(model.parameters(), lr=1e-3)

        if args.distplan == "CAI_ZeRO1":
            zero_stage = 1
        elif args.distplan == "CAI_ZeRO2":
            zero_stage = 2
        elif args.distplan == "CAI_Gemini":
            zero_stage = 3
        else:
            raise RuntimeError

        # wrap your model and optimizer
        model = zero_model_wrapper(model, zero_stage, gemini_config)
        optimizer = zero_optim_wrapper(model, optimizer, optim_config=optim_config)

        logger.info(get_mem_info(prefix='After init optim, '), ranks=[0])
    elif args.distplan.startswith("Pytorch"):
        model = model_builder(args).cuda()
        model = DDP(model)
        if args.distplan.endswith("DDP"):
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        elif args.distplan.endswith("ZeRO"):
            from torch.distributed.optim import ZeroRedundancyOptimizer
            optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.Adam, lr=1e-3)
    else:
        raise RuntimeError

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
        input_ids, labels = get_bert_data(BATCH_SIZE,
                                          SEQ_LEN,
                                          VOCAB_SIZE,
                                          NUM_LABELS,
                                          device=torch.cuda.current_device())
        optimizer.zero_grad()

        start = time()
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
        torch.cuda.synchronize()
        fwd_end = time()
        fwd_time = fwd_end - start
        logger.info(get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Forward '), ranks=[0])

        if args.distplan.startswith("CAI"):
            optimizer.backward(loss)
        elif args.distplan.startswith("Pytorch"):
            loss.backward()
        else:
            raise RuntimeError

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
