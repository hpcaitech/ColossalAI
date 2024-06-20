## Run Inference

The provided example `llama_generation.py` is an example to configure, initialize the engine, and run inference on provided model. We've added `AutoModelForCausalLM` and `NoPaddingLlamaModelInferPolicy` as model class and policy class, and the script is good to run inference with Llama 3.

For a basic setting, you could run the example by:
```bash
colossalai run --nproc_per_node 1 llama_generation.py -m PATH_MODEL --max_length 128
```

Run multi-GPU inference (Tensor Parallelism), as in the following example using 2 GPUs:
```bash
colossalai run --nproc_per_node 2 llama_generation.py -m PATH_MODEL --max_length 128 --tp_size 2
```

## Run Speculative Decoding

Colossal-Inference supports speculative decoding using the inference engine, with optimized kernels and cache management for the main model.

Both a drafter model (small model) and a main model (large model) will be used during speculative decoding process. The drafter model will generate a few tokens sequentially, and then the main model will validate those candidate tokens in parallel and accept validated ones. The decoding process will be speeded up, for the latency of speculating multiple tokens by the drafter model is lower than that by the main model.

Moreover, Colossal-Inference also supports GLIDE, a modified draft model architecture that reuses key and value caches from the main model, which improves the acceptance rate and increment the speed-up ratio. Details can be found in research paper GLIDE with a CAPE - A Low-Hassle Method to Accelerate Speculative Decoding on [arXiv](https://arxiv.org/pdf/2402.02082.pdf).

Right now, Colossal-Inference offers a GLIDE model compatible with vicuna7B (https://huggingface.co/lmsys/vicuna-7b-v1.5). You can find the fine-tuned GLIDE drafter model `cxdu/glide-vicuna7b` on the HuggingFace Hub: https://huggingface.co/cxdu/glide-vicuna7b.

Benchmarking with gsm8k and MT-Bench dataset with batch size 1 on H800, the speed increase for using speculative decoding is around 1.28x, and the speed increase for using speculative decoding with Glide model (as drafter model) is around 1.5x.

## Usage

For main model, you might want to use model card  `lmsys/vicuna-7b-v1.5` at [HuggingFace Hub](https://huggingface.co/lmsys/vicuna-7b-v1.5).
For regular drafter model, you might want to use model card `JackFram/llama-68m` at [HuggingFace Hub](https://huggingface.co/JackFram/llama-68m).
For the GLIDE drafter model, you could use model card `cxdu/glide-vicuna7b` at [HuggingFace Hub](https://huggingface.co/cxdu/glide-vicuna7b).


You could run speculative decoding by
```bash
colossalai run --nproc_per_node 1 llama_generation.py -m PATH_MODEL --drafter_model PATH_DRAFTER_MODEL --max_length 128
```

Run multi-GPU inference (Tensor Parallelism), as in the following example using 2 GPUs.
```bash
colossalai run --nproc_per_node 2 llama_generation.py -m PATH_MODEL --drafter_model PATH_DRAFTER_MODEL --max_length 128 --tp_size 2
```

If you want to try the GLIDE model (glide-vicuna7b) as the drafter model with vicuna-7B, you could provide the GLIDE model path or model card as drafter model and enable the feature by
```python
from colossalai.inference.modeling.models.glide_llama import GlideLlamaForCausalLM
drafter_model = GlideLlamaForCausalLM.from_pretrained(drafter_model_path_or_name)
...
engine.enable_spec_dec(drafter_model, use_glide_drafter=True)
```
