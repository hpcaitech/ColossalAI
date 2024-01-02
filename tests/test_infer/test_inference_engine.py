import pytest
from transformers import AutoTokenizer, GenerationConfig

import colossalai
from colossalai.inference.config import InferenceConfig
from colossalai.inference.core.engine import InferenceEngine
from colossalai.testing import spawn


def check_inference_engine(test_cai=False):
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=50000, hidden_size=512, intermediate_size=1536, num_attention_heads=4, num_hidden_layers=4
        )
    )

    inputs = [
        "介绍一下今天的北京",
        "介绍一下武汉",
    ]

    if test_cai:
        inference_config = InferenceConfig(max_output_len=1)
        inference_engine = InferenceEngine(model, tokenizer, inference_config, verbose=True)
        inference_engine.add_request(prompts=inputs)
        assert inference_engine.request_handler._has_waiting()
        generation_config = GenerationConfig(top_k=2, top_p=0.8, do_sample=True)
        outputs = inference_engine.generate(generation_config)
    else:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        inputs = tokenizer.batch_encode_plus(inputs, padding=True, return_tensors="pt")["input_ids"]
        generation_config = GenerationConfig(
            top_k=2, top_p=0.8, do_sample=True, pad_token_id=tokenizer.pad_token_id, max_new_tokens=1
        )
        outputs = model.generate(inputs, generation_config=generation_config)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, port=port, host="localhost")
    check_inference_engine(True)
    check_inference_engine(False)

    # TODO: There are some in sampler
    # for s1, s2 in zip(cai_outputs, transformer_outputs):
    #     assert s1 == s2


@pytest.mark.dist
def test_inference_engine():
    spawn(run_dist, 1)


if __name__ == "__main__":
    test_inference_engine()
