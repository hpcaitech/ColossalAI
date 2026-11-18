"""Record the official loop/checkpoint implementation and installed HF protocol."""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
import transformers

from tests.kit.remote_models import load_remote_classes

_, _, cls = load_remote_classes("ouro14")
module = sys.modules[cls.__module__]
print("Python", sys.executable, "Torch", torch.__version__, "Transformers", transformers.__version__)
print("Official", inspect.getfile(cls))
print("Decoder MRO", module.OuroDecoderLayer.__mro__)
for target in [
    module.OuroModel.forward,
    module.OuroDecoderLayer.forward,
    transformers.PreTrainedModel.gradient_checkpointing_enable,
    transformers.PreTrainedModel._set_gradient_checkpointing,
]:
    print("\nSOURCE", target.__qualname__)
    print(inspect.getsource(target))
print(
    "Attention flags",
    {
        k: getattr(cls, k, None)
        for k in ("_supports_sdpa", "_supports_flash_attn", "_supports_flex_attn", "_supports_attention_backend")
    },
)
