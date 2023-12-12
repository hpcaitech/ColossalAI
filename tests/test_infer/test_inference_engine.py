import transformers

from colossalai.inference.core.config import InferenceConfig
from colossalai.inference.core.engine import InferenceEngine


def test_inference_engine():
    model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=20000, hidden_size=512, intermediate_size=1536, num_attention_heads=4, num_hidden_layers=4
        )
    )
    inference_config = InferenceConfig(model, "hf-internal-testing/llama-tokenizer")
    inference_engine = InferenceEngine(inference_config)

    inputs = [
        "介绍一下北京",
        "介绍一下武汉",
    ]

    inference_engine.add_request(prompts=inputs)
    outputs = inference_engine.generate(None)

    for s1, s2 in zip(inputs, outputs):
        assert s1 == s2


if __name__ == "__main__":
    test_inference_engine()
