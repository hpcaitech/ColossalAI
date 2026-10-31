import json

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from colossalai.checkpoint_io.utils import save_config_file


def test_save_config_file_writes_model_config(tmp_path):
    # Sharded checkpoints are reloaded with `from_pretrained(checkpoint_dir)`; without config.json
    # Transformers falls back to the default (full-size) config instead of the saved model's.
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = LlamaForCausalLM(config).to(torch.bfloat16)

    save_config_file(model, str(tmp_path))

    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["hidden_size"] == 32
    assert saved["num_hidden_layers"] == 2
    assert saved["architectures"] == ["LlamaForCausalLM"]
    assert saved.get("dtype", saved.get("torch_dtype")) == "bfloat16"
    assert (tmp_path / "generation_config.json").is_file()


def test_save_config_file_skips_non_huggingface_model(tmp_path):
    save_config_file(torch.nn.Linear(2, 2), str(tmp_path))

    assert not (tmp_path / "config.json").exists()
