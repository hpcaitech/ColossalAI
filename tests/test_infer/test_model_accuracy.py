import importlib.util

import pytest
import torch
import transformers
from packaging import version

import colossalai
from colossalai.inference import InferenceEngine
from colossalai.inference.kv_cache import BatchInferState
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils.device import get_current_device

CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.7")

HAS_LIGHTLLM_KERNEL = True
if importlib.util.find_spec("lightllm") is None:
    HAS_LIGHTLLM_KERNEL = False

MANUAL_SEED = 123
MAX_IN_LEN = 1024
MAX_OUT_LEN = 256
CONFIG_MAP = {
    "toy": transformers.LlamaConfig(num_hidden_layers=4),
}


def data_gen_fn(batch_size: int = 2, vocab_size: int = 30000, seq_len: int = 512):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=get_current_device())
    attention_mask = torch.ones_like(input_ids)
    data = dict(input_ids=input_ids, attention_mask=attention_mask)
    return data


def run_torch(bsz: int, in_len: int):
    torch.manual_seed(MANUAL_SEED)
    config = CONFIG_MAP["toy"]
    config.pad_token_id = config.eos_token_id

    model = transformers.LlamaForCausalLM(config)
    model = model.half()
    model.eval()
    model = model.to(get_current_device())
    inputs = data_gen_fn(batch_size=bsz, seq_len=in_len)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits


def run_inference(tp_size, pp_size, bsz: int, in_len: int):
    assert in_len <= MAX_IN_LEN
    torch.manual_seed(MANUAL_SEED)

    config = CONFIG_MAP["toy"]
    config.pad_token_id = config.eos_token_id
    inputs = data_gen_fn(batch_size=bsz, seq_len=in_len)

    model = transformers.LlamaForCausalLM(config)
    engine = InferenceEngine(
        tp_size=tp_size,
        pp_size=pp_size,
        model=model,
        max_input_len=MAX_IN_LEN,
        max_output_len=MAX_OUT_LEN,
        dtype="fp16",
    )
    infer_state = BatchInferState.init_from_batch(
        batch=inputs,
        max_input_len=engine.max_input_len,
        max_output_len=engine.max_output_len,
        cache_manager=engine.cache_manager_list[0],
    )
    # Bind the infer state to the model manually as not using pipeline parallel
    engine.model.model.infer_state = infer_state

    # TODO PP accuracy test to be added later
    # Currently, PP does not support model forward for a single-token output
    with torch.no_grad():
        outputs = engine.model(**inputs)

    return outputs.logits


def launch_run_inference(rank, world_size, port, tp_size, pp_size, bsz, in_len):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    assert torch.allclose(run_torch(bsz, in_len), run_inference(tp_size, pp_size, bsz, in_len), atol=1e-1, rtol=1e-2)


@pytest.mark.skipif(
    not CUDA_SUPPORT or not HAS_LIGHTLLM_KERNEL,
    reason="kv-cache manager engine requires cuda version to be higher than 11.7",
)
@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
@parameterize("tp_size", [1, 2])
@parameterize("pp_size", [1])
@parameterize("bsz", [1, 4])
@parameterize("in_len", [128])
def test_model_forward_accuracy(tp_size, pp_size, bsz, in_len):
    spawn(launch_run_inference, nprocs=tp_size * pp_size, tp_size=tp_size, pp_size=pp_size, bsz=bsz, in_len=in_len)


if __name__ == "__main__":
    test_model_forward_accuracy()
