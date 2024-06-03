# Lazy initialization

Author: [Hongxin Liu](https://github.com/ver217)

**Prerequisite:**
- [Train with booster](../basics/booster_api.md)

## Introduction

Lazy initialization defers model initialization. It saves memory when initializing large models.

If your model has `N` billion parameters and your memory (or GPU memory) is `M` GB, we recommend you use lazy initialization when `4N >= M`. Otherwise, it is optional.

## Usage

Lazy initialization must be used with booster.

### API reference

{{ autodoc:colossalai.lazy.LazyInitContext }}

### Example

```python
import colossalai
from colossalai.lazy import LazyInitContext
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin

from transformers import LlamaForCausalLM, LlamaConfig, BertForPreTraining

colossalai.launch()
plugin = GeminiPlugin()
booster = Booster(plugin)

# 1. Initialize model from scratch
# Initialization on cuda will accelerate the initialization process but take more GPU memory.
with LazyInitContext(default_device="cuda"):
    model = LlamaForCausalLM(LlamaConfig(hidden_size=64, intermediate_size=172, num_hidden_layers=4, num_attention_heads=4))
model, *_ = booster.boost(model)

# 2. Initialize model from pretrained
with LazyInitContext():
    model = BertForPreTraining.from_pretrained("prajjwal1/bert-tiny")
model, *_ = booster.boost(model)
```

> ⚠️ Lazy initialization from pretrained is supported for colossalai>0.3.3 or main branch.

## Limitations

As we claimed, lazy initialization must be used with booster. And only several plugins support it.

| Plugin          | Supported | Remarks      |
|-----------------|-----------|--------------|
| Gemini          | Yes       |              |
| Hybrid Parallel | Yes       |              |
| Low Level Zero  | No        | No need      |
| Torch DDP       | No        | Incompatible |
| Torch FSDP      | No        | Incompatible |

Not all models can be lazily initialized. In some cases, a part of parameters/buffers may be early initialized. But don't worry, this part usually takes a small proportion of the whole model.

And some models are not supported at all which will raise an error. We tested models in torchvision, diffusers, timm, transformers, torchaudio and torchrec. Below models are not supported:

| Model                         | Category     |
|-------------------------------|--------------|
| wav2vec2_base                 | torchaudio   |
| hubert_base                   | torchaudio   |
| ViTModel                      | transformers |
| ViTForMaskedImageModeling     | transformers |
| ViTForImageClassification     | transformers |
| Blip2Model                    | transformers |
| Blip2ForConditionalGeneration | transformers |

<!-- doc-test-command: torchrun --standalone --nproc_per_node=2 lazy_init.py  -->
