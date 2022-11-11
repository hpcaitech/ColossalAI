from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

import colossalai
from colossalai.fx import ColoTracer
from colossalai.fx.passes.shard_1d_pass import transformer_mlp_pass
from colossalai.tensor import ProcessGroup
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port
from colossalai.utils.model.lazy_init_context import LazyInitContext
from transformers import GPT2Config, GPT2LMHeadModel


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


class MLP(torch.nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, dim)
        self.linear2 = torch.nn.Linear(dim, dim)
        self.dropout = torch.nn.Dropout(0)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def GPT_config():
    model_builder = lambda: GPTLMModel(hidden_size=1024, num_layers=24, num_attention_heads=16, checkpoint=True)
    BATCH_SIZE = 1
    SEQ_LENGTH = 16

    def data_gen():
        input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64).to('meta')
        attention_mask = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64).to('meta')
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        return kwargs

    return model_builder, data_gen()


def MLP_config():
    return lambda: MLP(16), None


MODEL_CONFIGS = [GPT_config()]


## Randomly Generated Data
def get_data(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=torch.cuda.current_device())
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def run_workflow(world_size, dev, model_config):
    # initailization
    with LazyInitContext() as ctx:
        model = model_config[0]()

    for param in model.parameters():
        assert param.is_meta

    # tracing
    tracer = ColoTracer()
    graph = tracer.trace(model, meta_args=model_config[1])
    gm = torch.fx.GraphModule(model, graph, model.__class__.__name__)

    # annotate
    annotated_gm = transformer_mlp_pass(gm, process_group=ProcessGroup(tp_degree=world_size))
    annotated_gm.recompile()

    # materialization and sharding
    ctx.lazy_init_parameters(annotated_gm, device=dev)
    for param in model.parameters():
        assert not param.is_meta

    # # check sharding
    assert list(model.linear1.weight.shape) == [16 // world_size, 16]
    assert list(model.linear1.bias.shape) == [16 // world_size]
    assert list(model.linear2.weight.shape) == [16, 16 // world_size]

    # test forward to make sure that IR transform will produce the same results
    # like how ColoTensor would do it normally
    data = torch.rand(4, 16, device=dev)
    non_fx_out = model(data)
    fx_out = annotated_gm(data)
    assert torch.equal(non_fx_out, fx_out), f'{non_fx_out} vs {fx_out}'


def run_dist(rank, world_size, dev, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    for model_builder in MODEL_CONFIGS:
        run_workflow(world_size, dev, model_builder)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 2])
@pytest.mark.parametrize('dev', ['cuda', 'cpu'])
@rerun_if_address_is_in_use()
def test_complete_workflow(world_size, dev):
    if dev == 'cpu' and world_size > 1:
        return
    run_func = partial(run_dist, world_size=world_size, dev=dev, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_complete_workflow(1, 'cuda')
