import os
import pytest
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, BloomForCausalLM

import colossalai
from colossalai.inference.tensor_parallel import TPInferEngine
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.testing import clear_cache_before_run, rerun_if_address_is_in_use, spawn

TP_SIZE = 2
MAX_BATCH_SIZE = 4
MAX_INPUT_LEN = 16
MAX_OUTPUT_LEN = 32


def run():

    model_path = "/data3/data/model_eval_for_commerical_use/phoenix-inst-chat-7b"
    if os.path.isdir(model_path) is False:
        return 
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    text = "Introduce some landmarks in Beijing"
    input_ids = tokenizer.batch_encode_plus([text], return_tensors='pt')

    model = BloomForCausalLM.from_pretrained(model_path, pad_token_id=tokenizer.eos_token_id)
    model = model.half()
    model.to(torch.cuda.current_device())

    shard_config = ShardConfig(enable_tensor_parallelism=True, inference_only=True)
    shardformer = ShardFormer(shard_config=shard_config)

    infer_engine = TPInferEngine(model, MAX_BATCH_SIZE, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
    infer_engine.prepare_with_shard_config(shard_config=shard_config)
    infer_engine.shard_model_by(shardformer)

    generate_kwargs = dict(do_sample=False)
    outputs = infer_engine.generate(input_ids, generate_kwargs)

    if not dist.is_initialized() or dist.get_rank() == 0:
        output_text = tokenizer.decode(outputs[0])
        print(output_text)


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
    test_engine_infer()
