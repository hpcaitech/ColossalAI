from itertools import accumulate

import pytest
import torch
import torch.nn as nn
from transformers import BloomConfig, BloomForCausalLM, LlamaConfig, LlamaForCausalLM
from transformers.tokenization_utils_base import BatchEncoding

import colossalai
from colossalai.inference.tensor_parallel import TPInferEngine
from colossalai.inference.tensor_parallel.batch_infer_state import BatchInferState
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn

TP_SIZE = 2
MAX_BATCH_SIZE = 4
MAX_INPUT_LEN = 16
MAX_OUTPUT_LEN = 8


def test_prepare_data():
    # dummy module used for testing
    class DummyModule(nn.Module):

        def __init__(self, config):
            super(DummyModule, self).__init__()
            self.config = config

        def forward(self, x):
            return x

    # dummy config used for testing
    class DummyModelConfig:

        def __init__(self):
            self.hidden_size = 4096
            self.num_attention_heads = 32
            self.num_hidden_layers = 8

    dummy_config = DummyModelConfig()
    model = DummyModule(dummy_config)
    infer_engine = TPInferEngine(model, MAX_BATCH_SIZE, MAX_INPUT_LEN, MAX_OUTPUT_LEN)

    input_ids_list = [[80540, 15473, 3331, 11970, 90472, 361, 61335], [80540, 15473, 3331, 11970],
                      [80540, 15473, 3331, 11970], [80540, 15473]]
    batch_size = len(input_ids_list)
    max_seq_len = max(len(li) for li in input_ids_list)
    attention_mask = [[0] * max_seq_len for _ in range(batch_size)]
    for i, li in enumerate(input_ids_list):
        attention_mask[i][max_seq_len - len(li):] = [1 for _ in range(len(li))]
    data = dict(input_ids=input_ids_list, attention_mask=attention_mask)
    inputs_batch_encoding = BatchEncoding(data=data)

    seq_lengths = [len(li) for li in input_ids_list]
    start_loc = list(accumulate([0] + seq_lengths[:-1]))
    seq_lengths = torch.tensor(seq_lengths, dtype=torch.int32)
    start_loc = torch.tensor(start_loc, dtype=torch.int32)

    # input token id list as inputs
    batch_state_out1 = infer_engine.prepare_batch_state(inputs_batch_encoding)
    # BatchEncoding as inputs
    batch_state_out2 = infer_engine.prepare_batch_state(input_ids_list)

    assert batch_state_out1.batch_size == batch_state_out2.batch_size == batch_size
    assert torch.equal(batch_state_out1.seq_len.to(seq_lengths.device), seq_lengths)
    assert torch.equal(batch_state_out2.seq_len.to(seq_lengths.device), seq_lengths)
    assert torch.equal(batch_state_out1.start_loc.to(start_loc.device), start_loc)
    assert torch.equal(batch_state_out2.start_loc.to(start_loc.device), start_loc)


def test_orig_generate():
    input_ids = torch.randint(low=10, high=1000, size=(MAX_BATCH_SIZE, MAX_INPUT_LEN))

    model_config = LlamaConfig()
    model = LlamaForCausalLM(model_config)
    model = model.half()
    model.to(torch.cuda.current_device())

    shard_config = ShardConfig(enable_tensor_parallelism=False)

    # init TPInferEngine and
    infer_engine = TPInferEngine(model, MAX_BATCH_SIZE, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
    infer_engine.prepare_with_shard_config(shard_config)

    # original model generate
    generate_kwargs = dict(do_sample=False)
    infer_engine.generate(input_ids, generate_kwargs)

    torch.cuda.empty_cache()


def run():
    model_config = BloomConfig()
    model = BloomForCausalLM(model_config)
    model = model.half()
    model.to(torch.cuda.current_device())

    shard_config = ShardConfig(enable_tensor_parallelism=True, inference_only=True)
    shardformer = ShardFormer(shard_config=shard_config)

    infer_engine = TPInferEngine(model, MAX_BATCH_SIZE, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
    infer_engine.prepare_with_shard_config(shard_config=shard_config)
    infer_engine.shard_model_by(shardformer)

    assert infer_engine.cache_manager is not None
    assert infer_engine.tp_size == TP_SIZE
    assert infer_engine.head_num == model_config.num_attention_heads // TP_SIZE

    torch.cuda.empty_cache()


def check_engine(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_engine_infer():
    spawn(check_engine, TP_SIZE)


if __name__ == '__main__':
    test_prepare_data()
    test_orig_generate()
    test_engine_infer()
