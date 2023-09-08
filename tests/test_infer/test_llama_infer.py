import os

import pytest
import torch
import torch.distributed as dist
from packaging import version
from transformers import LlamaForCausalLM, LlamaTokenizer

import colossalai
from colossalai.inference.tensor_parallel.engine import TPInferEngine
from colossalai.logging import disable_existing_loggers
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
TPSIZE = 2
BATCH_SIZE = 8
MAX_INPUT_LEN = 12
MAX_OUTPUT_LEN = 100

CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse('11.5')


def init_to_get_rotary(self, base=10000):
    self.config.head_dim_ = self.config.hidden_size // self.config.num_attention_heads
    if not hasattr(self.config, "rope_scaling"):
        rope_scaling_factor = 1.0
    else:
        rope_scaling_factor = self.config.rope_scaling.factor if self.config.rope_scaling is not None else 1.0
    if hasattr(self.config, "max_sequence_length"):
        max_seq_len = self.config.max_sequence_length
    elif hasattr(self.config, "max_position_embeddings"):
        max_seq_len = self.config.max_position_embeddings * rope_scaling_factor
    else:
        max_seq_len = 2048 * rope_scaling_factor
    base = float(base)
    inv_freq = 1.0 / (base**(torch.arange(0, self.config.head_dim_, 2, device="cpu", dtype=torch.float32) /
                             self.config.head_dim_))
    t = torch.arange(max_seq_len + 1024 * 64, device="cpu", dtype=torch.float32) / rope_scaling_factor
    freqs = torch.outer(t, inv_freq)

    self._cos_cached = torch.cos(freqs).to(torch.float16).cuda()
    self._sin_cached = torch.sin(freqs).to(torch.float16).cuda()
    return


@parameterize('test_config', [{
    'tp_size': TPSIZE,
}])
def run_llama_test(test_config):

    llama_model_path = "/data/scratch/llama-7b-hf"

    if os.path.isdir(llama_model_path) is False:
        sub_model_zoo = model_zoo.get_sub_registry('transformers_llama_for_casual_lm')
        for name, (model_fn, data_gen_fn, _, _, _) in sub_model_zoo.items():
            orig_model = model_fn()
            init_to_get_rotary(orig_model.model, base=10000)
            orig_model = orig_model.half()
            data = data_gen_fn()

            infer_engine = TPInferEngine(orig_model, BATCH_SIZE, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
            infer_engine.optimize_model(test_config)

            generate_kwargs = dict(do_sample=False)
            outputs = infer_engine.generate(data, **generate_kwargs)

            assert outputs is not None

            print(outputs)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(llama_model_path)
        tokenizer.pad_token_id = tokenizer.unk_token_id
        model = LlamaForCausalLM.from_pretrained(llama_model_path, pad_token_id=tokenizer.eos_token_id)
        init_to_get_rotary(model.model, base=10000)
        model = model.half()

        text = ["how is weather today?", "i am "]
        input_ids = tokenizer.batch_encode_plus(text, return_tensors='pt', padding=True, device='cuda')

        infer_engine = TPInferEngine(model, BATCH_SIZE, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
        infer_engine.optimize_model(test_config)

        generate_kwargs = dict(max_new_tokens=MAX_OUTPUT_LEN, do_sample=False)
        outputs = infer_engine.generate(input_ids, **generate_kwargs)

        assert outputs is not None

        if not dist.is_initialized() or dist.get_rank() == 0:
            for o in outputs:
                output_text = tokenizer.decode(o)
                # print(output_text)


def check_llama(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_llama_test()


@pytest.mark.skipif(not CUDA_SUPPORT, reason="kv-cache manager engine requires cuda version to be higher than 11.5")
@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_llama():
    spawn(check_llama, TPSIZE)


if __name__ == "__main__":
    test_llama()
