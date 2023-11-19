import os
import warnings
import time

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
from transformers import LlamaForCausalLM, LlamaTokenizer

torch.cuda.empty_cache()


os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
TPSIZE = 1
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

    shard_config = ShardConfig(enable_tensor_parallelism=True if test_config['tp_size'] > 1 else False,
                               extra_kwargs={"inference_only": True})
    infer_engine = TPInferEngine(model, shard_config, args.batch_size, args.input_len, args.output_len)

    generate_kwargs = dict(max_new_tokens=args.output_len, do_sample=False)
    
    input_tokens = {
        "input_ids": torch.randint(1, 1000, (args.batch_size, args.input_len), device="cuda"),
        "attention_mask": torch.ones((args.batch_size, args.input_len), device="cuda"),
    }
    outputs = infer_engine.generate(input_tokens, **generate_kwargs)
    assert outputs is not None
    
    times = []
    warmup = 2
    for i in range(6):
        torch.cuda.synchronize()
        start = time.time()
        outputs = infer_engine.generate(input_tokens, **generate_kwargs)
        torch.cuda.synchronize()
        end = time.time()
        out_len = outputs.shape[1]
        print("generation time {} s".format(str(end - start)))
        print(out_len - args.input_len)
        times.append((end - start) / (out_len - args.input_len))
    
    times = times[warmup:]
    latency = sum(times) / len(times)
    print("total process latency is : " + str(latency) + " s")
    print("total throughput is : " + str(1 / latency * args.batch_size))
        
        


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
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Maximum batch size")
    parser.add_argument("--input_len", type=int, default=128, help="Maximum input length")
    parser.add_argument("--output_len", type=int, default=256, help="Maximum output length")
    parser.add_argument(
        "--test_mode", type=str, help="Test mode", default="e2e_test", choices=["e2e_test", "decoder_test"]
    )
    args = parser.parse_args()
    test_llama(args)
