import os

import pytest
import torch
import torch.distributed as dist
from packaging import version
from transformers import AutoTokenizer, BloomForCausalLM

import colossalai
from colossalai.inference.tensor_parallel import TPInferEngine
from colossalai.logging import disable_existing_loggers
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn

TP_SIZE = 2
MAX_BATCH_SIZE = 4
MAX_INPUT_LEN = 16
MAX_OUTPUT_LEN = 32

CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse('11.5')


@parameterize('test_config', [{
    'tp_size': TP_SIZE,
}])
def run(test_config):
    model_path = "/data3/models/bloom-7b1"
    if os.path.isdir(model_path) is False:
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    text1 = "Introduce some landmarks in Beijing"
    text2 = "how is weather today?"
    input_ids = tokenizer.batch_encode_plus([text1, text2], return_tensors='pt', padding=True)

    model = BloomForCausalLM.from_pretrained(model_path, pad_token_id=tokenizer.eos_token_id)
    model = model.half()

    infer_engine = TPInferEngine(model, MAX_BATCH_SIZE, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
    infer_engine.optimize_model(test_config)

    generate_kwargs = dict(do_sample=False)
    outputs = infer_engine.generate(input_ids, **generate_kwargs)

    assert outputs is not None

    if not dist.is_initialized() or dist.get_rank() == 0:
        # output_text = tokenizer.decode(outputs[0])
        # print(output_text)
        for o in outputs:
            output_text = tokenizer.decode(o)
            print(output_text)


def check_engine(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run()


@pytest.mark.skipif(not CUDA_SUPPORT, reason="kv-cache manager engine requires cuda version to be higher than 11.5")
@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_engine_infer():
    spawn(check_engine, TP_SIZE)


if __name__ == '__main__':
    test_engine_infer()
