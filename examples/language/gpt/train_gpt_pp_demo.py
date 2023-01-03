import argparse
import time
from functools import partial

import torch
from model_zoo import model_builder
from torch import nn
from tqdm import tqdm

from colossalai.fx import ColoTracer
from colossalai.fx.passes.adding_split_node_pass import avgnode_split_pass, split_with_split_nodes_pass
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.pipeline.middleware.adaptor import get_fx_topology
from colossalai.pipeline.rpc._pipeline_schedule import OneFOneBPipelineEngine
from colossalai.pipeline.rpc.utils import rpc_run


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="gpt2_medium")
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dp_degree', type=int, default=1)
    parser.add_argument('--tp_degree', type=int, default=1)
    parser.add_argument('--num_microbatches', type=int, default=2)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--master_addr', type=str, default='localhost')
    parser.add_argument('--master_port', type=str, default='29020')
    parser.add_argument('--num_worker_threads', type=int, default=128)
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


def create_partition_module(pp_rank: int, stage_num: int, model, data_kwargs):
    tracer = ColoTracer()
    meta_args = {k: v.to('meta') for k, v in data_kwargs.items()}
    graph = tracer.trace(root=model, meta_args=meta_args)
    gm = torch.fx.GraphModule(model, graph, model.__class__.__name__)
    annotated_model = avgnode_split_pass(gm, stage_num)

    top_module, split_submodules = split_with_split_nodes_pass(annotated_model, merge_output=True)
    topo = get_fx_topology(top_module)
    for submodule in split_submodules:
        if isinstance(submodule, torch.fx.GraphModule):
            setattr(submodule, '_topo', topo)
    return split_submodules[pp_rank + 1]


def partition(logger, model_type, data_kwargs, pp_rank: int, chunk: int, stage_num: int):
    # build model
    model = model_builder(model_type)(checkpoint=False)
    module = create_partition_module(pp_rank, stage_num, model, data_kwargs)
    num_params = sum(param.numel() for param in module.parameters())
    logger.info(f'{pp_rank=} number of args in this partition:{num_params}')
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

    disable_existing_loggers()
    logger = get_dist_logger()
    logger.info(f"{args.model_type}, batch size {batch_size}, num stage {stage_num}, num microbatch {num_microbatches}",
                ranks=[0])

    torch.manual_seed(123)

    # build criterion
    criterion = GPTLMLoss()

    # warm up pipeline fx partition
    input_ids, attn_mask = get_data(batch_size, SEQ_LEN, VOCAB_SIZE)
    warmup_data_kwargs = {'input_ids': input_ids, 'attention_mask': attn_mask}

    # set 1f1b pipeline engine
    pp_engine = OneFOneBPipelineEngine(partition_fn=partial(partition, logger, model_type, warmup_data_kwargs),
                                       stage_num=stage_num,
                                       num_microbatches=num_microbatches,
                                       device=device,
                                       chunk=1,
                                       criterion=criterion,
                                       metric=None,
                                       checkpoint=False)

    # build optim
    pp_engine.initialize_optimizer(HybridAdam, lr=1e-3)

    times = []
    for n in tqdm(range(NUM_STEPS)):
        # we just use randomly generated data here
        input_ids, attn_mask = get_data(batch_size, SEQ_LEN, VOCAB_SIZE)
        batch = {'input_ids': input_ids, 'attention_mask': attn_mask}

        start = time.time()
        outputs = pp_engine.forward_backward(batch=batch, labels=input_ids, forward_only=False)
        cost_time = time.time() - start
        times.append(cost_time)

    logger.info("avg cost time : {}s".format(sum(times) / len(times)))


if __name__ == '__main__':
    args = parse_args()
    rpc_run(args, run_master)
