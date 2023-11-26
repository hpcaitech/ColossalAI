import argparse
import time
from functools import partial

import torch
from model_zoo import model_builder
from torch import nn

from colossalai.fx import ColoTracer
from colossalai.fx.passes.adding_split_node_pass import gpipe_dp_split_pass, split_with_split_nodes_pass
from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from colossalai.legacy.pipeline.middleware.adaptor import get_fx_topology
from colossalai.legacy.pipeline.rpc._pipeline_schedule import FillDrainPipelineEngine
from colossalai.legacy.pipeline.rpc.utils import rpc_run
from colossalai.logging import disable_existing_loggers, get_dist_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="gpt2_medium")
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dp_degree", type=int, default=1)
    parser.add_argument("--tp_degree", type=int, default=1)
    parser.add_argument("--num_microbatches", type=int, default=2)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="29011")
    parser.add_argument("--num_worker_threads", type=int, default=128)
    return parser.parse_args()


class GPTLMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


# Randomly Generated Data
def get_data(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=torch.cuda.current_device())
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)


# Create annotated model which is noted where to be splitted.
def get_annotated_model(model, data_kwargs, num_stages, num_microbatches):
    tracer = ColoTracer()
    meta_args = {k: v.to("meta") for k, v in data_kwargs.items()}
    graph = tracer.trace(root=model, meta_args=meta_args)
    gm = torch.fx.GraphModule(model, graph, model.__class__.__name__)

    interp_meta_args = tuple([v.to("meta") for k, v in data_kwargs.items()])
    interp = MetaInfoProp(gm)
    interp.run(*interp_meta_args)

    # annotated_model = avgnode_split_pass(gm, num_stages)
    annotated_model = gpipe_dp_split_pass(gm, num_stages, num_microbatches, mode="block", block_limit=0.01)

    return annotated_model


def create_partition_module(pp_rank: int, num_stages: int, model, data_kwargs, num_microbatches):
    annotated_model = get_annotated_model(model, data_kwargs, num_stages, num_microbatches)
    top_module, split_submodules = split_with_split_nodes_pass(annotated_model, merge_output=True)
    topo = get_fx_topology(top_module)
    for submodule in split_submodules:
        if isinstance(submodule, torch.fx.GraphModule):
            setattr(submodule, "_topo", topo)
    return split_submodules[pp_rank + 1]


def partition(model, data_kwargs, num_microbatches, pp_rank: int, chunk: int, stage_num: int):
    module = create_partition_module(pp_rank, stage_num, model, data_kwargs, num_microbatches)
    return module


def run_master(args):
    batch_size = args.batch_size
    device = args.device
    world_size = args.world_size
    stage_num = world_size
    num_microbatches = args.num_microbatches
    model_type = args.model_type
    # batch size per DP degree
    SEQ_LEN = 1024
    VOCAB_SIZE = 50257
    NUM_STEPS = 10
    WARMUP_STEPS = 1

    disable_existing_loggers()
    logger = get_dist_logger()
    logger.info(
        f"{args.model_type}, batch size {batch_size}, num stage {stage_num}, num microbatch {num_microbatches}",
        ranks=[0],
    )

    torch.manual_seed(123)

    # build criterion
    criterion = GPTLMLoss()

    # warm up pipeline fx partition
    input_ids, attn_mask = get_data(batch_size, SEQ_LEN, VOCAB_SIZE)
    warmup_data_kwargs = {"input_ids": input_ids, "attention_mask": attn_mask}

    # create model
    logger.info(f"start model_builder")
    model = model_builder(model_type)(checkpoint=False)
    logger.info(f"end model_builder")

    # set 1f1b pipeline engine
    pp_engine = FillDrainPipelineEngine(
        partition_fn=partial(partition, model, warmup_data_kwargs, num_microbatches),
        stage_num=stage_num,
        num_microbatches=num_microbatches,
        device=device,
        chunk=1,
        criterion=criterion,
        metric=None,
        checkpoint=False,
    )

    partition_numels = pp_engine.remote_numels()
    for rank, numel in partition_numels.items():
        logger.info(f"{rank=} numel in the partition:{numel}")

    # build optim
    pp_engine.initialize_optimizer(torch.optim.Adam, lr=1e-3)

    ranks_tflops = {}
    for n in range(NUM_STEPS):
        # we just use randomly generated data here
        input_ids, attn_mask = get_data(batch_size, SEQ_LEN, VOCAB_SIZE)
        batch = {"input_ids": input_ids, "attention_mask": attn_mask}

        start = time.time()
        outputs = pp_engine.forward_backward(batch=batch, labels=input_ids, forward_only=False)
        step_time = time.time() - start

        for rank, numel in partition_numels.items():
            if rank not in ranks_tflops:
                ranks_tflops[rank] = []
            step_tflops = get_tflops(numel, batch_size, SEQ_LEN, step_time)

            logger.info(
                f"Rank{rank} , [{n + 1}/{NUM_STEPS}] , Step time: {step_time:.3f}s, TFLOPS: {get_tflops(numel, batch_size, SEQ_LEN, step_time):.3f}",
                ranks=[0],
            )

            if n >= WARMUP_STEPS:
                ranks_tflops[rank].append(step_tflops)

    median_index = ((NUM_STEPS - WARMUP_STEPS) >> 1) + WARMUP_STEPS
    gpu_tflops = []
    for rank, tflops_list in ranks_tflops.items():
        tflops_list.sort()
        gpu_tflops.append(tflops_list[median_index])
        logger.info(f"GPU{rank} Median TFLOPS is {tflops_list[median_index]:.3f}")

    logger.info(f"Total TFLOPS is {sum(gpu_tflops):.3f}")
    logger.info(f"Avg TFLOPS per GPU is {sum(gpu_tflops) / world_size:.3f}")


if __name__ == "__main__":
    args = parse_args()
    rpc_run(args, run_master)
