import os
import warnings

import torch
import torch.distributed as dist
import argparse
from packaging import version

import colossalai
from colossalai.inference.tensor_parallel.engine import TPInferEngine
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer import ShardConfig
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn
from transformers import AutoModelForCausalLM, AutoTokenizer


os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
TPSIZE = 1
BATCH_SIZE = 4
MAX_INPUT_LEN = 32
MAX_OUTPUT_LEN = 128

CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse('11.5')


@parameterize('test_config', [{
    'tp_size': TPSIZE,
}])
def run_llama_test(test_config, args):

    model_path = args.path
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.unk_token_id
    model = AutoModelForCausalLM.from_pretrained(model_path, pad_token_id=tokenizer.eos_token_id)
    model = model.half()

    text = ["Introduce London.", "What is the genus of Poodle?"]
    input_ids = tokenizer.batch_encode_plus(text, return_tensors='pt', padding=True)

    print(input_ids)

    shard_config = ShardConfig(enable_tensor_parallelism=True if test_config['tp_size'] > 1 else False,
                               extra_kwargs={"inference_only": True})
    infer_engine = TPInferEngine(model, shard_config, BATCH_SIZE, MAX_INPUT_LEN, MAX_OUTPUT_LEN)

    generate_kwargs = dict(max_new_tokens=MAX_OUTPUT_LEN, do_sample=False)
    outputs = infer_engine.generate(input_ids, **generate_kwargs)

    assert outputs is not None

    if not dist.is_initialized() or dist.get_rank() == 0:
        for o in outputs:
            output_text = tokenizer.decode(o)
            print(output_text)


def check_llama(rank, world_size, port, args):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_llama_test(args=args)


@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_llama(args):
    spawn(check_llama, args.tp_size, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default = "hpcai-tech/Colossal-LLaMA-2-7b-base", help="Model path")
    parser.add_argument("-tp", "--tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Maximum batch size")
    parser.add_argument("--input_len", type=int, default=1024, help="Maximum input length")
    parser.add_argument("--output_len", type=int, default=128, help="Maximum output length")
    parser.add_argument(
        "--test_mode", type=str, help="Test mode", default="e2e_test", choices=["e2e_test", "decoder_test"]
    )
    args = parser.parse_args()
    test_llama(args)
