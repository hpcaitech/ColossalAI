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

To be added.

## 🕹 Usage


To be added.

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
