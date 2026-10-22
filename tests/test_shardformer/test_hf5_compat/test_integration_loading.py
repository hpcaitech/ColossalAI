"""Offline integration of config saving and deferred lazy pretrained setup."""

import json

import pytest
import torch
from transformers import OPTConfig, OPTForCausalLM

from colossalai.checkpoint_io.utils import save_config_file
from colossalai.interface.pretrained import get_pretrained_path
from colossalai.lazy import LazyInitContext, LazyTensor


def small_model():
    return OPTForCausalLM(
        OPTConfig(
            vocab_size=64,
            hidden_size=32,
            word_embed_proj_dim=32,
            ffn_dim=48,
            num_hidden_layers=1,
            num_attention_heads=4,
            max_position_embeddings=16,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )


def test_config_roundtrip(tmp_path):
    model = small_model().bfloat16()
    save_config_file(model, str(tmp_path))
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved.get("dtype", saved.get("torch_dtype")) == "bfloat16"
    config = OPTConfig.from_pretrained(tmp_path, local_files_only=True)
    assert config.hidden_size == 32
    assert (tmp_path / "generation_config.json").exists()


def test_config_nonmaster_and_nonhf(tmp_path):
    save_config_file(small_model(), str(tmp_path), is_master=False)
    save_config_file(torch.nn.Linear(2, 2), str(tmp_path))
    assert not (tmp_path / "config.json").exists()


def test_lazy_pretrained_setup(tmp_path):
    model = small_model()
    model.save_pretrained(tmp_path)
    initial_dtype = torch.get_default_dtype()
    with LazyInitContext():
        candidate = OPTForCausalLM.from_pretrained(tmp_path, local_files_only=True, dtype=torch.bfloat16)
        assert isinstance(next(candidate.parameters()), LazyTensor)
    assert torch.get_default_dtype() == initial_dtype
    assert next(candidate.parameters()).dtype == torch.bfloat16
    assert get_pretrained_path(candidate).endswith("model.safetensors")
    # Lazy from_pretrained stores the checkpoint path; materialization alone
    # does not load saved weights. Verify explicit subsequent weight loading.
    LazyInitContext.materialize(candidate)
    # HF5 may initialize selected norms in FP32; convert the fully materialized
    # model explicitly before exercising this CPU BF16 comparison.
    candidate.bfloat16()
    candidate.load_state_dict(model.state_dict())
    inputs = torch.tensor([[1, 3, 4]])
    reference = model.bfloat16().eval()
    torch.testing.assert_close(candidate(input_ids=inputs).logits, reference(input_ids=inputs).logits)


def test_booster_loads_deferred_checkpoint(tmp_path):
    from colossalai.booster import Booster

    model = small_model().eval()
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.fill_((index + 1) / 128)
    model.save_pretrained(tmp_path, max_shard_size="4KB")
    with LazyInitContext():
        candidate = OPTForCausalLM.from_pretrained(tmp_path, local_files_only=True)
    assert get_pretrained_path(candidate).endswith("model.safetensors.index.json")
    LazyInitContext.materialize(candidate)
    candidate, *_ = Booster().boost(candidate)
    assert get_pretrained_path(candidate) is None
    expected = model.state_dict()
    for name, tensor in candidate.state_dict().items():
        torch.testing.assert_close(tensor, expected[name], atol=0, rtol=0)
    inputs = torch.tensor([[1, 3, 4]])
    torch.testing.assert_close(candidate(input_ids=inputs).logits, model(input_ids=inputs).logits)


def test_constructor_failure_restores_default_dtype(tmp_path, monkeypatch):
    model = small_model()
    model.save_pretrained(tmp_path)
    original_dtype = torch.get_default_dtype()

    def fail_constructor(*args, **kwargs):
        raise RuntimeError("deliberate constructor failure")

    monkeypatch.setattr(OPTForCausalLM, "__init__", fail_constructor)
    with LazyInitContext(), pytest.raises(RuntimeError, match="deliberate constructor"):
        OPTForCausalLM.from_pretrained(tmp_path, dtype=torch.float64, local_files_only=True)
    assert torch.get_default_dtype() == original_dtype


def test_strict_booster_still_rejects_missing_untied_weight(tmp_path):
    from colossalai.booster import Booster

    model = small_model()
    model.config.tie_word_embeddings = False
    model.lm_head = torch.nn.Linear(32, 64, bias=False)
    state = model.state_dict()
    state.pop("lm_head.weight")
    torch.save(state, tmp_path / "incomplete.bin")
    with pytest.raises(RuntimeError, match="lm_head.weight"):
        Booster().load_model(model, str(tmp_path / "incomplete.bin"), strict=True)


def test_auth_conflict_does_not_expose_values(tmp_path):
    with LazyInitContext(), pytest.raises(ValueError, match="Conflicting token") as exc:
        OPTForCausalLM.from_pretrained(tmp_path, token="first-private-value", use_auth_token="second-private-value")
    assert "private-value" not in str(exc.value)
