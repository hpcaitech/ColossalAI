import pytest
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

import colossalai
from colossalai.inference.tensor_parallel import TPInferEngine
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn

TP_SIZE = 2
BATCH_SIZE = 4
MAX_INPUT_LEN = 16
MAX_OUTPUT_LEN = 8


def test_orig_generate():
    input_ids = torch.randint(low=10, high=1000, size=(BATCH_SIZE, MAX_INPUT_LEN))

    model_config = LlamaConfig()
    model = LlamaForCausalLM(model_config)
    shard_config = ShardConfig(enable_tensor_parallelism=False)

    # init TPInferEngine and
    infer_engine = TPInferEngine(model, BATCH_SIZE, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
    infer_engine.prepare_with_shard_config(shard_config)

    # original model generate
    generate_kwargs = dict(do_sample=False)
    infer_engine.generate(input_ids, generate_kwargs)


def run():
    model_config = LlamaConfig()
    model = LlamaForCausalLM(model_config)
    shard_config = ShardConfig(enable_tensor_parallelism=True, inference_only=True)
    shardformer = ShardFormer(shard_config=shard_config)

    infer_engine = TPInferEngine(model, BATCH_SIZE, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
    infer_engine.prepare_with_shard_config(shard_config=shard_config)
    infer_engine.shard_model_by(shardformer)

    assert infer_engine.cache_manager is not None
    assert infer_engine.tp_size == TP_SIZE
    assert infer_engine.head_num == model_config.num_attention_heads // TP_SIZE

    # TODO After adding forward replacement for CausalLM,
    #      uncomment these lines to test sharded model generate
    # generate_kwargs = dict(do_sample=False)
    # infer_engine.generate(input_ids, generate_kwargs)

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
    test_orig_generate()
    test_engine_infer()
