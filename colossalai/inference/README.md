# ⚡️ ColossalAI-Inference

## 📚 Table of Contents

- [⚡️ ColossalAI-Inference](#️-colossalai-inference)
  - [📚 Table of Contents](#-table-of-contents)
  - [📌 Introduction](#-introduction)
  - [🛠 Design and Implementation](#-design-and-implementation)
  - [🕹 Usage](#-usage)
  - [🪅 Support Matrix](#-support-matrix)
  - [🗺 Roadmap](#-roadmap)
  - [🌟 Acknowledgement](#-acknowledgement)


## 📌 Introduction
ColossalAI-Inference is a module which offers acceleration to the inference execution of Transformers models, especially LLMs. In ColossalAI-Inference, we leverage high-performance kernels, KV cache, paged attention, continous batching and other techniques to accelerate the inference of LLMs. We also provide simple and unified APIs for the sake of user-friendliness.

## 🛠 Design and Implementation

### :book: Overview

ColossalAI-Inference has **4** major components, namely namely `engine`,`request handler`,`cache manager`, and `modeling`.

- **Engine**: It orchestrates the inference step. During inference, it recives a request, calls `request handler` to schedule a decoding batch, and executes the model forward pass to perform a iteration. It returns the inference results back to the user at the end.
- **Request Handler**: It manages requests and schedules a proper batch from exisiting requests.
- **Cache manager** It is bound within the `request handler`, updates cache blocks and logical block tables as scheduled by the `request handler`.
- **Modelling**: We rewrite the model and layers of LLMs to simplify and optimize the forward pass for inference.


A high-level view of the inter-component interaction is given below. We would also introduce more details in the next few sections.

<p align="center">
   <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/inference/Structure/Introduction.png" width="600"/>
   <br/>
</p>

### :mailbox_closed: Engine
Engine is designed as the entry point where the user kickstarts an inference loop. User can easily instantialize an inference engine with the inference configuration and execute requests. The engine object will expose the following APIs for inference:

-  `generate`: main function which handles inputs, performs inference and returns outputs
-  `add_request`: add request to the waiting list
-  `step`: perform one decoding iteration. The `request handler` first schedules a batch to do prefill/decoding. Then, it invokes a model to generate a batch of token and afterwards does logit processing and sampling, checks and decodes finished requests.

### :game_die: Request Handler

Request handler is responsible for managing requests and scheduling a proper batch from exisiting requests. According to the existing work and experiments, we do believe that it is beneficial to increase the length of decoding sequences. In our design, we partition requests into three priorities depending on their lengths, the longer sequences are first considered.

<p align="center">
   <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/inference/Structure/Request_handler.svg" width="800"/>
   <br/>
</p>

### :radio: KV cache and cache manager

We design a unified block cache and cache manager to allocate and manage memory. The physical memory is allocated before decoding and represented by a logical block table. During decoding process, cache manager administrates the physical memory through `block table` and other components(i.e. engine) can focus on the lightweight `block table`. More details are given below.

- `cache block`: We group physical memory into different memory blocks. A typical cache block is shaped `(num_kv_heads, head_size, block_size)`. We determine the block number beforehand. The memory allocation and computation are executed at the granularity of memory block.
- `block table`: Block table is the logical representation of cache blocks. Concretely, a block table of a single sequence is a 1D tensor, with each element holding a block ID. Block ID of `-1` means "Not Allocated". In each iteration, we pass through a batch block table to the corresponding model.

<figure>
<p align="center">
   <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/inference/Structure/BlockTable.svg"/>
    <br/>
   <figcation>Example of Batch Block Table</figcation>
   </p>
</figure>


### :railway_car: Modeling

Modeling contains models and layers, which are hand-crafted for better performance easier usage. Deeply integrated with `shardformer`, we also construct policy for our models. In order to minimize users' learning costs, our models are aligned with [Transformers](https://github.com/huggingface/transformers)

## 🕹 Usage

### :arrow_right: Quick Start

```python
import torch
import transformers
import colossalai
from colossalai.inference import InferenceEngine, InferenceConfig
from pprint import pprint

colossalai.launch_from_torch(config={})

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
            )

# Step 3: create an engine with model and config
engine = InferenceEngine(model, tokenizer, inference_config, verbose=True)

# Step 4: try inference
prompts = ['Who is the best player in the history of NBA?']
response = engine.generate(prompts)
pprint(response)
```

### :bookmark: Customize your inference engine
Besides the basic quick-start inference, you can also customize your inference engine via modifying config or upload your own model or decoding components (logit processors or sampling strategies).

#### Inference Config
Inference Config is a unified api for generation process. You can define the value of args to control the generation, like `max_batch_size`,`max_output_len`,`dtype` to decide the how many sequences can be handled at a time, and how many tokens to output. Refer to the source code for more detail.

#### Generation Config
In colossal-inference, Generation config api is inherited from [Transformers](https://github.com/huggingface/transformers). Usage is aligned. By default, it is automatically generated by our system and you don't bother to construct one. If you have such demand, you can also create your own and send it to your engine.

#### Logit Processors
The `Logit Processosr` receives logits and return processed results. You can take the following step to make your own.

```python
@register_logit_processor("name")
def xx_logit_processor(logits, args):
  logits = do_some_process(logits)
  return logits
```

#### Sampling Strategies
We offer 3 main sampling strategies now (i.e. `greedy sample`, `multinomial sample`, `beam_search sample`), you can refer to [sampler](/ColossalAI/colossalai/inference/sampler.py) for more details. We would strongly appreciate if you can contribute your varities.

## 🪅 Support Matrix

| Model |  KV Cache | Paged Attention | Kernels | Tensor Parallelism | Speculative Decoding |
| - | - | - | - | - | - |
| Llama |  ✅ | ✅ | ✅ | 🔜 | ✅ |


Notations:
- ✅: supported
- ❌: not supported
- 🔜: still developing, will support soon

## 🗺 Roadmap

- [x] KV Cache
- [x] Paged Attention
- [x] High-Performance Kernels
- [x] Llama Modelling
- [x] User Documentation
- [x] Speculative Decoding
- [ ] Tensor Parallelism
- [ ] Beam Search
- [ ] Early stopping
- [ ] Logger system
- [ ] SplitFuse
- [ ] Continuous Batching
- [ ] Online Inference
- [ ] Benchmarking

## 🌟 Acknowledgement

This project was written from scratch but we learned a lot from several other great open-source projects during development. Therefore, we wish to fully acknowledge their contribution to the open-source community. These projects include

- [vLLM](https://github.com/vllm-project/vllm)
- [LightLLM](https://github.com/ModelTC/lightllm)
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

# we do not find any research work related to lightllm
```
