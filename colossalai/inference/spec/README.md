# Speculative Decoding

Colossal-Inference supports speculative decoding using the inference engine, with optimized kernels and cache management for the main model.

Both a drafter model (small model) and a main model (large model) will be used during speculative decoding process. The drafter model will generate a few tokens sequentially, and then the main model will validate those candidate tokens in parallel and accept validated ones. The decoding process will be speeded up, for the latency of speculating multiple tokens by the drafter model is lower than that by the main model.

Moreover, Colossal-Inference also supports GLIDE, a modified draft model architecture that reuses key and value caches from the main model, which improves the acceptance rate and increment the speed-up ratio. Details can be found in research paper GLIDE with a CAPE - A Low-Hassle Method to Accelerate Speculative Decoding on [arXiv](https://arxiv.org/pdf/2402.02082.pdf).

Right now, Colossal-Inference offers a GLIDE model compatible with vicuna7B. You can find the fine-tuned GLIDE drafter model `cxdu/glide47m-vicuna7b` on the HuggingFace Hub: https://huggingface.co/cxdu/glide47m-vicuna7b.

## Usage

For main model, you might want to use model card  `lmsys/vicuna-7b-v1.5` at [HuggingFace Hub](https://huggingface.co/lmsys/vicuna-7b-v1.5).
For regular drafter model, you might want to use model card `JackFram/llama-68m` at [HuggingFace Hub](https://huggingface.co/JackFram/llama-68m).
For the GLIDE drafter model, you could use model card `cxdu/glide47m-vicuna7b` at [HuggingFace Hub](https://huggingface.co/cxdu/glide47m-vicuna7b).

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

import colossalai
from colossalai.inference.config import InferenceConfig
from colossalai.inference.core.engine import InferenceEngine, GenerationConfig
from colossalai.inference.modeling.models.glide_llama import GlideLlamaForCausalLM, GlideLlamaConfig

# launch colossalai, setup distributed environment
colossalai.launch_from_torch(config={})

# main model
model_path_or_name = "REPLACE_TO_VICUNA_7B_PATH_OR_MODEL_CARD"
model = AutoModelForCausalLM.from_pretrained(model_path_or_name)

# use the same tokenizer for both the main model and the drafter model
tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
tokenizer.pad_token = tokenizer.eos_token

# drafter model
drafter_model_path_or_name = "REPLACE_TO_LLAMA_68M_PATH_OR_MODEL_CARD"
drafter_model = AutoModelForCausalLM.from_pretrained(drafter_model_path_or_name)

# Initialize the inference engine
inference_config = InferenceConfig(
    dtype="fp16",
    max_batch_size=1,
    max_input_len=256,
    max_output_len=256,
    prefill_ratio=1.2,
    block_size=16,
    max_n_spec_tokens=5,
    prompt_template="vicuna",
)
engine = InferenceEngine(model, tokenizer, inference_config, verbose=True)

# turn on speculative decoding with the drafter model
engine.enable_spec_dec(drafter_model)

prompt = "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions. "
generation_config = GenerationConfig(
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_length=128,
    num_beams=1,
    do_sample=False,
)
out = engine.generate(prompts=[prompt], generation_config=generation_config)
print(out)

# use GLIDE Llama model as drafter model
drafter_model_path_or_name = "cxdu/glide47m-vicuna7b"
glide_config = GlideLlamaConfig(
    intermediate_size=8192,
    large_hidden_size=4096,
    large_num_attention_heads=32,
    num_hidden_layers=1,
)
drafter_model = GlideLlamaForCausalLM.from_pretrained(drafter_model_path_or_name, config=glide_config)

# turn on speculative decoding with the GLIDE model
engine.enable_spec_dec(drafter_model, use_glide_drafter=True)
out = engine.generate(prompts=[prompt], generation_config=generation_config)
print(out)
```

You could run the above code by
```bash
colossalai run --nproc_per_node 1 script_name.py
```

## Benchmark

With batch size 1, testing with gsm8k and MT-Bench dataset on NVIDIA H800 80G:

| Method                       | Tokens/Sec |
| :--------------------------- | :--------- |
| Non-Spec-Dec                 | ~90        |
| Spec-Dec                     | ~115       |
| Spec-Dec with GLIDE Model    | ~135       |
