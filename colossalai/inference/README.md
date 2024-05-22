# ⚡️ ColossalAI-Inference

## 📚 Table of Contents

- [⚡️ ColossalAI-Inference](#️-colossalai-inference)
  - [📚 Table of Contents](#-table-of-contents)
  - [📌 Introduction](#-introduction)
  - [🕹 Usage](#-usage)
  - [🗺 Roadmap](#-roadmap)
  - [🪅 Support Matrix](#-support-matrix)
  - [🛠 Design and Components](#-design-and-components)
    - [Overview](#overview)
    - [Engine](#engine)
    - [Blocked KV Cache Manager](#kv-cache)
    - [Batching](#batching)
    - [Modeling](#modeling)
  - [🌟 Acknowledgement](#-acknowledgement)


## 📌 Introduction
ColossalAI-Inference is a module which offers acceleration to the inference execution of Transformers models, especially LLMs. In ColossalAI-Inference, we leverage high-performance kernels, KV cache, paged attention, continous batching and other techniques to accelerate the inference of LLMs. We also provide simple and unified APIs for the sake of user-friendliness.


## 🕹 Usage

### :arrow_right: Quick Start

The sample usage of the inference engine is given below:

```python
import torch
import transformers
import colossalai
from colossalai.inference import InferenceEngine, InferenceConfig
from pprint import pprint

colossalai.launch_from_torch()

# Step 1: create a model in "transformers" way
model_path = "lmsys/vicuna-7b-v1.3"
model = transformers.LlamaForCausalLM.from_pretrained(model_path).cuda()
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

# Step 2: create an inference_config
inference_config = InferenceConfig(
                dtype=torch.float16,
                max_batch_size=4,
                max_input_len=1024,
                max_output_len=512,
                use_cuda_kernel=True,
            )

# Step 3: create an engine with model and config
engine = InferenceEngine(model, tokenizer, inference_config, verbose=True)

# Step 4: try inference
prompts = ['Who is the best player in the history of NBA?']
response = engine.generate(prompts)
pprint(response)
```

You could run the sample code by
```bash
colossalai run --nproc_per_node 1 your_sample_name.py
```

For detailed examples, you might want to check [inference examples](../../examples/inference/llama/README.md).

### :bookmark: Customize your inference engine
Besides the basic quick-start inference, you can also customize your inference engine via modifying inference config or uploading your own models, policies, or decoding components (logits processors or sampling strategies).

#### Inference Config
Inference Config is a unified config for initializing the inference engine, controlling multi-GPU generation (Tensor Parallelism), as well as presetting generation configs. Below are some commonly used `InferenceConfig`'s arguments:

- `max_batch_size`: The maximum batch size. Defaults to 8.
- `max_input_len`: The maximum input length (number of tokens). Defaults to 256.
- `max_output_len`: The maximum output length (number of tokens). Defaults to 256.
- `dtype`: The data type of the model for inference. This can be one of `fp16`, `bf16`, or `fp32`. Defaults to `fp16`.
- `kv_cache_dtype`: The data type used for KVCache. Defaults to the same data type as the model (`dtype`). KVCache quantization will be automatically enabled if it is different from that of model (`dtype`).
- `use_cuda_kernel`: Determine whether to use CUDA kernels or not. If disabled, Triton kernels will be used. Defaults to False.
- `tp_size`: Tensor-Parallelism size. Defaults to 1 (tensor parallelism is turned off by default).

#### Generation Config
Refer to transformers [GenerationConfig](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig) on functionalities and usage of specific configs. In ColossalAI-Inference, generation configs can be preset in `InferenceConfig`. Supported generation configs include:

- `do_sample`: Whether or not to use sampling. Defaults to False (greedy decoding).
- `top_k`: The number of highest probability vocabulary tokens to keep for top-k-filtering. Defaults to 50.
- `top_p`: If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation. Defaults to 1.0.
- `temperature`: The value used to modulate the next token probabilities. Defaults to 1.0.
- `no_repeat_ngram_size`: If set to int > 0, all ngrams of that size can only occur once. Defaults to 0.
- `repetition_penalty`: The parameter for repetition penalty. 1.0 means no penalty. Defaults to 1.0.
- `forced_eos_token_id`: The id of the token to force as the last generated token when max_length is reached. Defaults to `None`.

Users can also create a transformers [GenerationConfig](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig) as an input argument for `InferenceEngine.generate` API. For example

```python
generation_config = GenerationConfig(
    max_length=128,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=1.0,
)
response = engine.generate(prompts=prompts, generation_config=generation_config)
```

## 🗺 Roadmap

We will follow the following roadmap to develop major features of ColossalAI-Inference:

- [x] Blocked KV Cache
- [x] Paged Attention
- 🟩 Fused Kernels
- [x] Speculative Decoding
- [x] Continuous Batching
- 🟩 Tensor Parallelism
- [ ] Online Inference
- [ ] Beam Search
- [ ] SplitFuse

Notations:
- [x] Completed
- 🟩 Model specific and in still progress.

## 🪅 Support Matrix

| Model     | Model Card                                                                                     | Tensor Parallel | Lazy Initialization | Paged Attention | Fused Kernels | Speculative Decoding |
|-----------|------------------------------------------------------------------------------------------------|-----------------|---------------------|-----------------|---------------|----------------------|
| Baichuan  | `baichuan-inc/Baichuan2-7B-Base`,<br> `baichuan-inc/Baichuan2-13B-Base`, etc                   | ✅              | [ ]                   | ✅               | ✅             | [ ]                    |
| ChatGLM   |                                                                                                | [ ]             | [ ]                 | [ ]             | [ ]           | [ ]                  |
| DeepSeek  |                                                                                                | [ ]             | [ ]                 | [ ]             | [ ]           | [ ]                  |
| Llama     | `meta-llama/Llama-2-7b`,<br> `meta-llama/Llama-2-13b`,<br> `meta-llama/Meta-Llama-3-8B`,<br> `meta-llama/Meta-Llama-3-70B`, etc | ✅               | [ ]                   | ✅               | ✅             | ✅                    |
| Mixtral   |                                                                                                | [ ]             | [ ]                 | [ ]             | [ ]           | [ ]                  |
| Qwen      |                                                                                                | [ ]             | [ ]                 | [ ]             | [ ]           | [ ]                  |
| Vicuna    | `lmsys/vicuna-13b-v1.3`,<br> `lmsys/vicuna-7b-v1.5`                                            | ✅              | [ ]                   | ✅               | ✅             | ✅                    |
| Yi        | `01-ai/Yi-34B`, etc                                                                            | ✅              | [ ]                   | ✅               | ✅             | ✅                    |


## 🛠 Design and Components

### Overview

ColossalAI-Inference has **4** major components, namely `engine`, `request handler`, `kv cache manager`, and `modeling`.

<p align="center">
   <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/inference/colossalai-inference-overview-abstract.png" alt="colossalai-inference-components-overview" width="600" />
   <br/>
</p>

- **Engine**: It orchestrates the inference step. During inference, it recives a request, calls `request handler` to schedule a decoding batch, and executes the model forward pass to perform a iteration. It returns the inference results back to the user at the end.
- **Request Handler**: It manages requests and schedules a proper batch from exisiting requests.
- **KV Cache Manager** It is bound within the `request handler`, updates cache blocks and logical block tables as scheduled by the `request handler`.
- **Modelling**: We rewrite the model and layers of LLMs to simplify and optimize the forward pass for inference.


An overview of the inter-component interaction is given below (RPC version). We would also introduce more details in the next few sections.

<p align="center">
   <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/inference/colossalai-inference-framework.png" alt="colossalai-inference-framework-rpc" width="600"/>
   <br/>
</p>

### Engine

Engine is designed as the entry point where the user kickstarts an inference loop. User can easily initialize an inference engine with the inference configurations and execute with their requests. We provided several versions of inference engines, namely `InferenceEngine`, `RPCInferenceEngine`, and `AsyncInferenceEngine`, which are used for different conditions and purposes.

For examples/inference/llama and `RPCInferenceEngine`, we expose the following APIs for inference:

-  `generate`: main function which handles inputs, performs inference and returns outputs.
-  `add_request`: add a single or multiple requests to the inference engine.
-  `step`: perform one decoding iteration. The `request handler` first schedules a batch to do prefill/decoding. Then, it invokes a model to generate a batch of token and afterwards does logit processing and sampling, checks and decodes finished requests.
- `enable_spec_dec`: used for speculative decoding. Enable speculative decoding for subsequent generations.
- `disable_spec_dec`: used for speculative decoding. Disable speculative decoding for subsequent generations
- `clear_spec_dec`: clear structures and models related to speculative decoding, if exists.

For `AsyncInferenceEngine`, we expose the following APIs for inference:
- `add_request`: async method. Add a request to the inference engine, as well as to the waiting queue of the background tracker.
- `generate`: async method. Perform inference from a request.
- `step`: async method. Perform one decoding iteration, if there exists any request in waiting queue.

For now, `InferenceEngine` is used for offline generation; `AsyncInferenceEngine` is used for online serving with a single card; and `RPCInferenceEngine` is used for online serving with multiple cards. In future, we will focus on `RPCInferenceEngine` and improve user experience of LLM serving.


### KV cache

Learnt from [PagedAttention](https://arxiv.org/abs/2309.06180) by [vLLM](https://github.com/vllm-project/vllm) team, we use a unified blocked KV cache and cache manager to allocate and manage memory. The physical memory is pre-allocated during initialization and represented by a logical block table. During decoding process, cache manager administrates the physical memory through `block table` of a batch and so that other components (i.e. engine) can focus on the lightweight `block table`. More details are given below.

- `logical cache block`: We group physical memory into different memory blocks. A typical cache block is shaped `(num_kv_heads, block_size, head_size)`. We determine the block number beforehand. The memory allocation and computation are executed at the granularity of memory block.
- `block table`: Block table is the logical representation of cache blocks. Concretely, a block table of a single sequence is a 1D tensor, with each element holding a block ID. Block ID of `-1` means "Not Allocated". In each iteration, we pass through a batch block table to the corresponding model.

<p align="center">
   <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/inference/Structure/BlockTable.svg"/>
   <br/>
   <em>Example of block table for a batch</em>
</p>


### Batching

Request handler is responsible for managing requests and scheduling a proper batch from exisiting requests. Based on [Orca's](https://www.usenix.org/conference/osdi22/presentation/yu) and [vLLM's](https://github.com/vllm-project/vllm) research and work on batching requests, we applied continuous batching with unpadded sequences, which enables various number of sequences to pass projections (i.e. Q, K, and V) together in different steps by hiding the dimension of number of sequences, and decrement the latency of incoming sequences by inserting a prefill batch during a decoding step and then decoding together.

<p align="center">
   <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/inference/continuous_batching.png" width="800"/>
   <br/>
   <em>Naive Batching: decode until each sequence encounters eos in a batch</em>
</p>

<p align="center">
   <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/inference/naive_batching.png" width="800"/>
   <br/>
   <em>Continuous Batching: dynamically adjust the batch size by popping out finished sequences and inserting prefill batch</em>
</p>

### Modeling

Modeling contains models, layers, and policy, which are hand-crafted for better performance easier usage. Integrated with `shardformer`, users can define their own policy or use our preset policies for specific models. Our modeling files are aligned with [Transformers](https://github.com/huggingface/transformers). For more details about the usage of modeling and policy, please check `colossalai/shardformer`.


## 🌟 Acknowledgement

This project was written from scratch but we learned a lot from several other great open-source projects during development. Therefore, we wish to fully acknowledge their contribution to the open-source community. These projects include

- [vLLM](https://github.com/vllm-project/vllm)
- [flash-attention](https://github.com/Dao-AILab/flash-attention)

If you wish to cite relevant research papars, you can find the reference below.

```bibtex
# vllm
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}

# flash attention v1 & v2
@inproceedings{dao2022flashattention,
  title={Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
  author={Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
@article{dao2023flashattention2,
  title={Flash{A}ttention-2: Faster Attention with Better Parallelism and Work Partitioning},
  author={Dao, Tri},
  year={2023}
}
```
