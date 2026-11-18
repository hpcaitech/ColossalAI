"""Offline, pinned tiny fixtures for LLaDA/LLaDA2/Ouro.

Importing this module neither downloads code nor imports any remote model.
The caller must explicitly prepare the listed snapshots before construction.
These fixtures are independent of the broad model-zoo dependency imports.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers.dynamic_module_utils import get_class_from_dynamic_module

REMOTE_MODELS = {
    "llada": ("GSAI-ML/LLaDA-8B-Base", "0f2787f2d87eac5eed8a087d5ecd24277e6255b2"),
    "llada21": ("inclusionAI/LLaDA2.1-mini", "20e64e2ad21644d0e5248586ed9c942cdd45de0f"),
    "llada22": ("inclusionAI/LLaDA2.2-mini", "e4f77d25ab5ca86f2223c6cdc6a72b887e20c8a5"),
    "ouro14": ("ByteDance/Ouro-1.4B", "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"),
    "ouro26": ("ByteDance/Ouro-2.6B", "1ed04250da1a9936042725d302e81c8fa2ab5abd"),
}


def tiny_config_overrides(key):
    if key == "llada":
        return dict(
            d_model=64,
            n_heads=4,
            n_kv_heads=4,
            n_layers=2,
            mlp_hidden_size=128,
            vocab_size=128,
            embedding_size=128,
            max_sequence_length=64,
            pad_token_id=0,
            eos_token_id=1,
            mask_token_id=2,
            init_device="cpu",
            flash_attention=False,
            use_cache=False,
        )
    values = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2 if key.startswith("llada") else 4,
        head_dim=16,
        max_position_embeddings=64,
        max_window_layers=2,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=1,
        mask_token_id=2,
        use_cache=False,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        embedding_dropout=0.0,
        output_dropout=0.0,
        use_sliding_window=False,
        dtype="float32",
    )
    if key.startswith("llada"):
        values.update(
            num_experts=4,
            num_experts_per_tok=2,
            num_shared_experts=1,
            n_group=2,
            topk_group=1,
            moe_intermediate_size=32,
            first_k_dense_replace=1,
            rotary_dim=8,
            partial_rotary_factor=0.5,
        )
        if key == "llada22":
            values.update(block_size=4, expert_capacity=4, delete_token_id=3, split_token_id=4)
    else:
        values.update(layer_types=["full_attention"] * 2, total_ut_steps=2)
    return values


def load_remote_classes(key):
    repo, revision = REMOTE_MODELS[key]
    path = snapshot_download(repo, revision=revision, local_files_only=True)
    raw = json.loads((Path(path) / "config.json").read_text())
    config_class = get_class_from_dynamic_module(raw["auto_map"]["AutoConfig"], path, local_files_only=True)
    model_class = get_class_from_dynamic_module(raw["auto_map"]["AutoModelForCausalLM"], path, local_files_only=True)
    return raw, config_class, model_class


def tiny_config(key):
    raw, config_class, _ = load_remote_classes(key)
    raw.pop("torch_dtype", None)
    raw.update(tiny_config_overrides(key))
    config = config_class(**raw)
    config._attn_implementation = "eager"
    return config


def build_remote_model(key, *, adapted=True, config=None):
    _, _, model_class = load_remote_classes(key)
    config = tiny_config(key) if config is None else config
    if adapted and key == "llada":
        from colossalai.shardformer.modeling.llada import get_llada_model_class

        model_class = get_llada_model_class(model_class)
    elif adapted and key.startswith("llada"):
        from colossalai.shardformer.modeling.llada2_moe import get_llada2_model_class

        model_class = get_llada2_model_class(model_class)
    elif adapted and key.startswith("ouro"):
        from colossalai.shardformer.modeling.ouro import get_ouro_model_class

        model_class = get_ouro_model_class(model_class)
    kwargs = {"init_params": True} if key == "llada" else {}
    return model_class(config, **kwargs).float().cpu()


def tiny_data(key):
    # Local generator avoids changing other tests' random state.
    generator = torch.Generator().manual_seed(2026)
    ids = torch.randint(5, 128, (2, 16), generator=generator)
    labels = ids.clone()
    if key.startswith("llada"):
        supervised = torch.arange(16).remainder(2).eq(0).expand_as(ids)
        ids[supervised] = 2
        labels[~supervised] = -100
    return dict(input_ids=ids, labels=labels, use_cache=False, return_dict=True)


def forward_and_loss(model, key, data):
    if key == "llada":
        inputs = {name: value for name, value in data.items() if name != "labels"}
        output = model(**inputs)
        # Same-position fixed-mask CE checks autograd, not the full diffusion objective.
        loss = F.cross_entropy(output.logits.flatten(0, 1), data["labels"].flatten())
    else:
        output = model(**data)
        loss = output.loss
    return output, loss
