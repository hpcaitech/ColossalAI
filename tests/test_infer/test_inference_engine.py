import pytest
import transformers
from transformers import AutoTokenizer

import colossalai
from colossalai.inference.config import InferenceConfig
from colossalai.inference.core.engine import InferenceEngine
from colossalai.testing import spawn


def check_inference_engine():
    model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=20000, hidden_size=512, intermediate_size=1536, num_attention_heads=4, num_hidden_layers=4
        )
    )
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    inference_config = InferenceConfig()
    inference_engine = InferenceEngine(model, tokenizer, inference_config, verbose=True)

    inputs = [
        "介绍一下北京",
        "介绍一下武汉",
    ]

    inference_engine.add_request(prompts=inputs)
    outputs = inference_engine.generate(None)

    for s1, s2 in zip(inputs, outputs):
        assert s1 == s2


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, port=port, host="localhost")
    check_inference_engine()


@pytest.mark.dist
def test_inference_engine():
    spawn(run_dist, 1)


if __name__ == "__main__":
    test_inference_engine()
