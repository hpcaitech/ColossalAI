import pytest
import torch
from transformers import AutoTokenizer, BloomForCausalLM

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer.inference import InferenceEngine
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn

TP_SIZE = 2


# @parameterize
def run():
    # dummly set the model path, will revise later
    # bloom model
    model_path = "/data3/data/model_eval_for_commerical_use/phoenix-inst-chat-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = BloomForCausalLM.from_pretrained(model_path, pad_token_id=tokenizer.eos_token_id)

    text = "Introduce some landmarks in Beijing"
    input_ids = tokenizer.encode(text, return_tensors='pt')

    pg_mesh = ProcessGroupMesh(1, 1, TP_SIZE)
    shardconfig = ShardConfig(
        tensor_parallel_process_group=pg_mesh.get_group_along_axis(2),
        enable_tensor_parallelism=True,
        inference_only=True,
    )
    shardformer = ShardFormer(shard_config=shardconfig)

    infer_engine = InferenceEngine(model.half(), 4, 12, 8, tp_size=TP_SIZE)
    infer_engine.shard_model_by(shardformer)

    generate_kwargs = dict(do_sample=False)
    outputs = infer_engine.generate_by_set_infer_state(input_ids, generate_kwargs)

    output_text = tokenizer.decode(outputs)
    print(output_text)


def check_bloom(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_bloom_infer():
    spawn(check_bloom, TP_SIZE)


if __name__ == '__main__':
    test_bloom_infer()
