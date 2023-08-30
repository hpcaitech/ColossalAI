import os

import pytest
import torch
import numpy as np

import colossalai
from colossalai.logging import disable_existing_loggers
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn
from transformers import LlamaForCausalLM, LlamaTokenizer
from colossalai.cluster import ProcessGroupMesh
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.inference.tensor_parallel.engine import TPInferEngine
import torch.distributed as dist

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
TPSIZE = 1

def init_to_get_rotary(self, base=10000):
    self.config.head_dim_ = self.config.hidden_size // self.config.num_attention_heads
    if not hasattr(self.config, "rope_scaling"):
        rope_scaling_factor = 1.0
    else:
        rope_scaling_factor = self.config.rope_scaling.factor if self.config.rope_scaling is not None else 1.0
    if hasattr(self.config,"max_sequence_length"):
        max_seq_len = self.config.max_sequence_length
    elif hasattr(self.config,"max_position_embeddings"):
        max_seq_len = self.config.max_position_embeddings * rope_scaling_factor
    else:
        max_seq_len =  2048 * rope_scaling_factor
    base = float(base)
    inv_freq = 1.0 / (base ** (torch.arange(0, self.config.head_dim_, 2, device="cpu", dtype=torch.float32) / self.config.head_dim_))
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
    tokenizer = LlamaTokenizer.from_pretrained(llama_model_path)
    tokenizer.pad_token_id = tokenizer.unk_token_id
    model = LlamaForCausalLM.from_pretrained(llama_model_path, pad_token_id=tokenizer.eos_token_id)
    init_to_get_rotary(model.model, base=10000)
    model = model.half()
    model.to(torch.cuda.current_device())
    
    text = "Introduce some landmarks in Beijing"
    input_ids = tokenizer.encode(text, return_tensors='pt')
    # pg_mesh = ProcessGroupMesh(1, 1, test_config["tp_size"])
    
    infer_engine = TPInferEngine(model.half(), 4, 12, 8)
    shard_config = ShardConfig(enable_tensor_parallelism=True, inference_only=True)
    shardformer = ShardFormer(shard_config=shard_config)

    infer_engine.prepare_with_shard_config(shard_config)
    infer_engine.shard_model_by(shardformer)

    generate_kwargs = dict(do_sample=False)
    outputs = infer_engine.generate(input_ids, generate_kwargs)
    
    print("outputs: ", outputs)

    output_text = tokenizer.decode(outputs)
    print(output_text)


def check_llama(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_llama_test()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_llama():
    spawn(check_llama, TPSIZE)


if __name__ == "__main__":
    test_llama()
