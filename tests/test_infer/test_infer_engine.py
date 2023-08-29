import pytest
import torch.distributed as dist
from transformers import AutoTokenizer, BloomForCausalLM

import colossalai
from colossalai.inference.tensor_parallel import TPInferEngine
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer import ShardConfig, ShardFormer

TP_SIZE = 2


def test_tp_infer():

    model_path = "/data3/data/model_eval_for_commerical_use/phoenix-inst-chat-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = BloomForCausalLM.from_pretrained(model_path, pad_token_id=tokenizer.eos_token_id)

    text = "Introduce some landmarks in Beijing"
    input_ids = tokenizer.encode(text, return_tensors='pt')

    tp_process_group = dist.new_group([0, 1])

    infer_engine = TPInferEngine(model.half(), 4, 12, 8)
    shard_config = ShardConfig(enable_tensor_parallelism=True, tensor_parallel_process_group=tp_process_group)
    shardformer = ShardFormer(shard_config=shard_config)

    infer_engine.prepare_with_shard_config(shard_config)
    infer_engine.shard_model_by(shardformer)

    generate_kwargs = dict(do_sample=False)
    outputs = infer_engine.generate_by_set_infer_state(input_ids, generate_kwargs)

    output_text = tokenizer.decode(outputs)
    print(output_text)


if __name__ == '__main__':
    test_tp_infer()
