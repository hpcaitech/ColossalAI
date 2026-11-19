---
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
tags:
- looped-language-model
- reasoning
- recurrent-depth
---

# Ouro-1.4B

📚 [Paper](https://huggingface.co/papers/2510.25741) • 🏠 [Project Page](https://ouro-llm.github.io/)

![Ouro Logo](assets/logo.png)

## Model Description


**⚠️ IMPORTANT: This model is intended for research purposes only. It is provided as-is without warranties for production use.**

**Ouro-1.4B** is a 1.4 billion parameter Looped Language Model (LoopLM) that achieves exceptional parameter efficiency through iterative shared-weight computation. 

![Model Performance](assets/benchmark.png)

## Key Features

- **Exceptional Parameter Efficiency**: Matches 3-4B standard transformer performance with only 1.4B parameters
- **Iterative Latent Reasoning**: Performs reasoning through recurrent computation in latent space
- **Adaptive Computation**: Supports early exit mechanisms for dynamic compute allocation

## Configuration

### Recurrent Steps and Adaptive Exit

The model's computational behavior can be configured through the `config.json` file:

```json
{
  "total_ut_steps": 4,
  "early_exit_threshold": 1.0
}
```

- **`total_ut_steps`**: Controls the number of recurrent steps (default: 4). You can adjust this value to trade off between performance and computation time.
- **`early_exit_threshold`**: Controls the adaptive exit mechanism (default: 1.0). Lower values encourage earlier exit, while 1.0 means always use all steps.

**Example: Modify recurrent steps**
```python
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("ByteDance/Ouro-1.4B")
config.total_ut_steps = 3  # Use 3 recurrent steps instead of 4
model = AutoModelForCausalLM.from_pretrained(
    "ByteDance/Ouro-1.4B",
    config=config,
    device_map="auto"
)
```

> **Note**: vLLM does not currently support the adaptive exit feature due to its inference optimization characteristics. When using vLLM, the model will always execute the full number of `total_ut_steps`.

## Model Architecture

Ouro-1.4B is based on the decoder-only Transformer architecture with parameter sharing across recurrent steps:

| Configuration | Value |
|:---|:---|
| **Parameters** | 1.4B |
| **Layers** | 24 |
| **Recurrent Steps** | 4 |
| **Hidden Size** | 2048 |
| **Attention Heads** | Multi-Head Attention (MHA) |
| **FFN Activation** | SwiGLU |
| **Position Embedding** | RoPE |
| **Vocabulary Size** | 49,152 |
| **Context Length** | 4K (training), extendable to 64K |
| **Normalization** | Sandwich RMSNorm |

## Training Details

- **Training Tokens**: 7.7T tokens
- **Training Pipeline**: 
  - Stage 1: Pre-training (6T tokens)
  - Stage 2: CT Annealing (1.4T tokens)
  - Stage 3: Long Context Training (20B tokens)
  - Stage 4: Mid-training (300B tokens)
- **Data Composition**: Web data, code, mathematics, long-context documents
- **Optimizer**: AdamW (β₁=0.9, β₂=0.95)
- **Learning Rate Scheduler**: Warmup-Stable-Decay (WSD)



## Quick Start

**⚠️ IMPORTANT**: Please use `transformers<4.56.0` to avoid compatibility issues. We recommend `transformers==4.54.1` or earlier versions.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "ByteDance/Ouro-1.4B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

# Generate text
inputs = tokenizer("The future of AI is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Acknowledgments

We thank [@Antizana](https://github.com/Antizana) for the KV cache fix merged from [ouro-cache-fix](https://github.com/Antizana/ouro-cache-fix), which resolved a critical compatibility issue with transformers>=4.56.0.
## Citation

```bibtex
@article{zhu2025scaling,
  title={Scaling Latent Reasoning via Looped Language Models},
  author={Zhu, Rui-Jie and Wang, Zixuan and Hua, Kai and Zhang, Tianyu and Li, Ziniu and Que, Haoran and Wei, Boyi and Wen, Zixin and Yin, Fan and Xing, He and others},
  journal={arXiv preprint arXiv:2510.25741},
  year={2025}
}

## License

This model is licensed under Apache-2.0. See the LICENSE file for details.

---