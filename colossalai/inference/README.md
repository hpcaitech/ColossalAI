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
ColossalAI-Inference is a library which offers acceleration to Transformers models, especially LLMs. In ColossalAI-Inference, we leverage high-performance kernels, KV cache, paged attention, continous batching and other techniques to accelerate the inference of LLMs. We also provide a unified interface for users to easily use our library.

## 🛠 Design and Implementation

### :book: Overview
We build ColossalAI-Inference based on **Four** core components: `engine`,`request handler`,`cache manager(block cached)`, `hand crafted modeling`. **Engine** controls inference step, it recives `requests`, calls `request handler` to schedule a decoding batch and runs `modeling` to perform a iteration and returns finished `requests`. **Cache manager** is bound with `request handler`, updates cache blocks and logical block tables during schedule.

The interaction between different components are shown below, you can also checkout detailed introduction below.:
<p align="center">
   <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/inference/Structure/Introduction.png" width="600"/>
   <br/>
</p>

### :mailbox_closed: Design of engine
Engine is designed as starter of inference loop. User can easily instantialize an infer engine with config and execute requests. We provids apis below in engine, you can refer to source code for more information:
-  `generate`: main function, handle inputs and return outputs
-  `add_request`: add request to waitting list
-  `step`: perform one decoding iteration
    - first, `request handler` schedules a batch to do prefill/decode
    - then, invoke a model to generate a batch of token
    - after that, do logit processing and sampling, check and decode finished requests

### :game_die: Design of request_handler
Request handler is responsible manage requests and schedule a proper batch from exisiting requests. According to existing work and experiments, we do believe that it is beneficial to increase the length of decoding sequences. In our design, we partition requests into three priorities depending on their lengths, the longer sequences are first considered.
<p align="center">
   <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/inference/Structure/Request_handler.svg" width="800"/>
   <br/>
</p>

### :radio: Design of KV cache and cache manager
We design a unified blocked type cache and cache manager to distribute memory. The physical memory is allocated before decoding and represented by a logical block table. During decoding process, cache manager administrate physical memory through `block table` and other components(i.e. engine) can focus on the light-weighted `block table`. Their details are introduced below.
- `cache block` We group physical memory into different memory blocks. A typical cache block is shaped `(num_kv_heads, head_size, block_size)`. We decide block number beforehand. The memory allocation and computation are executed with the granularity of memory block.
- `block table` Block table is the logical representation of cache blocks. Concretely, a block table of a single sequence is a 1D tensor, with each element holding a block id of allocated id or `-1` for non allocated. Each iteration we pass through a batch block table to the corresponding model. For more information, you can checkout the source code.

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
You can enjoy your fast generation journey within three step
```python
# First, create a model in "transformers" way, you can provide a model config or use the default one.
model = transformers.LlamaForCausalLM(config).cuda()
# Second, create an inference_config
inference_config = InferenceConfig(
                dtype=args.dtype,
                max_batch_size=args.max_batch_size,
                max_input_len=args.seq_len,
                max_output_len=args.output_len,
            )
# Third, create an engine with model and config
engine = InferenceEngine(model, tokenizer, inference_config, verbose=True)

# Try fast infrence now!
prompts = {'Nice to meet you, Colossal-Inference!'}
engine.generate(prompts)

```

### :bookmark: Customize your inference engine
Besides the basic fast-start inference, you can also customize your inference engine via modifying config or upload your own model or decoding components (logit processors or sampling strategies).
#### Inference Config
Inference Config is a unified api for generation process. You can define the value of args to control the generation, like `max_batch_size`,`max_output_len`,`dtype` to decide the how many sequences can be handled at a time, and how many tokens to output. Refer to the source code for more detail.
#### Generation Config
In colossal-inference, Generation config api is inherited from [Transformers](https://github.com/huggingface/transformers). Usage is aligned. By default, it is automatically generated by our system and you don't bother to construct one. If you have such demand, you can also create your own and send it to your engine.

#### Logit Processors
Logit Processosr receives logits and return processed ones, take the following step to make your own.
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
| Llama |  ✅ | ✅ | ✅ | 🔜 | 🔜 |


Notations:
- ✅: supported
- ❌: not supported
- 🔜: still developing, will support soon

## 🗺 Roadmap

- [x] KV Cache
- [x] Paged Attention
- [x] High-Performance Kernels
- [x] Llama Modelling
- [ ] Tensor Parallelism
- [ ] Beam Search
- [ ] Speculative Decoding
- [ ] Continuous Batching
- [ ] Online Inference
- [ ] Benchmarking
- [ ] User Documentation

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
