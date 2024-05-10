import random

import numpy as np
import torch
from transformers import AutoTokenizer, GenerationConfig

from colossalai.inference.core.rpc_engine import RpycInferenceEngine
from colossalai.inference.rpc_config import InferenceConfig


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def infer():
    setup_seed(20)

    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    # model = LlamaForCausalLM(
    #     LlamaConfig(
    #         vocab_size=50000, hidden_size=512, intermediate_size=1536, num_attention_heads=4, num_hidden_layers=16
    #     )
    # )
    model = "/home/data/models/Llama-2-7b-hf"
    # model = model.eval()

    output_len = 38
    do_sample = False
    top_p = 0.5
    top_k = 50

    inference_config = InferenceConfig(
        max_output_len=output_len,
        prompt_template=None,
        dtype="fp16",
        use_cuda_kernel=True,
        tp_size=1,
    )

    # inference_engine = RpycInferenceEngine(model, tokenizer, inference_config, verbose=True, model_policy=NoPaddingLlamaModelInferPolicy())
    inference_engine = RpycInferenceEngine(model, tokenizer, inference_config, verbose=True, model_policy=None)

    inputs = [
        "介绍一下今天的北京,比如故宫，天安门，长城或者其他的一些景点,",
        "介绍一下武汉,",
    ]
    inference_engine.add_request(prompts=inputs)
    assert inference_engine.request_handler._has_waiting()
    generation_config = GenerationConfig(do_sample=do_sample, top_p=top_p, top_k=top_k)
    outputs = inference_engine.generate(generation_config=generation_config)

    print(outputs)
    inference_engine.kill_workers()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")  # this code will not be ok for settings to fork to subprocess
    infer()
