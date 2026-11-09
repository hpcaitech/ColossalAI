"""Build a tiny random LLaDA on CPU and check one forward pass, without pretrained weights."""

import json
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def build_tiny_model():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    print("Torch:", torch.__version__)
    print("Transformers:", transformers.__version__)

    path = snapshot_download(
        "GSAI-ML/LLaDA-8B-Base", revision="0f2787f2d87eac5eed8a087d5ecd24277e6255b2", local_files_only=True
    )
    raw = json.loads((Path(path) / "config.json").read_text())
    print("快照版本:", Path(path).name)
    config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)

    # Change only the in-memory config, not the shared model files.
    config.update(
        {
            "d_model": 64,
            "n_heads": 4,
            "n_kv_heads": 4,
            "n_layers": 2,
            "mlp_hidden_size": 128,
            "vocab_size": 128,
            "embedding_size": 128,
            "max_sequence_length": 64,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "mask_token_id": 2,
            "init_device": "cpu",
            "flash_attention": False,
            "use_cache": False,
        }
    )

    model_class = get_class_from_dynamic_module(
        raw["auto_map"]["AutoModelForCausalLM"],
        path,
        local_files_only=True,
    )
    print("开始创建 tiny 模型")
    # This class defaults to init_params=False; explicitly initialize it.
    model = model_class(config, init_params=True).float()
    print("BUILD_OK")
    print("参数数量:", sum(p.numel() for p in model.parameters()))

    return model


def main():
    model = build_tiny_model()
    model.eval()
    input_ids = torch.randint(3, 128, (2, 16))
    print("开始前向计算")
    with torch.no_grad():
        output = model(input_ids=input_ids, return_dict=True)
    logits = output.logits
    print("输出形状:", tuple(logits.shape))
    assert tuple(logits.shape) == (2, 16, 128)
    assert torch.isfinite(logits).all().item(), "输出出现 NaN 或无穷大"
    print("FORWARD_OK")


if __name__ == "__main__":
    main()
