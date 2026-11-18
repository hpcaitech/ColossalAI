"""CPU contracts for Ouro checkpointing, lazy construction and HF reload."""

import pytest
import torch
from transformers import __version__ as transformers_version

from colossalai.lazy import LazyInitContext, LazyTensor
from tests.kit.ouro import make_ouro, ouro_data


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("key", ["ouro14", "ouro26"])
@pytest.mark.parametrize("loops", [2, 4])
def test_checkpoint_matches_official(key, loops):
    torch.manual_seed(42)
    reference = make_ouro(key, loops, reference=True)
    candidate = make_ouro(key, loops, checkpoint=True)
    candidate.load_state_dict(reference.state_dict())
    data = ouro_data()
    expected, actual = reference(**data), candidate(**data)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=2e-5, atol=2e-6)
    expected.loss.backward()
    actual.loss.backward()
    assert all(layer._ouro_calls > loops for layer in candidate.model.layers), "No checkpoint recomputation"
    ref = dict(reference.named_parameters())
    for name, p in candidate.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        torch.testing.assert_close(p.grad, ref[name].grad, rtol=2e-5, atol=2e-6, msg=name)
    for model in (reference, candidate):
        torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False).step()
    for name, p in candidate.named_parameters():
        torch.testing.assert_close(p, ref[name], rtol=2e-5, atol=2e-6, msg=name)


@pytest.mark.parametrize("key", ["ouro14", "ouro26"])
def test_lazy_and_default_hf_reload(key, tmp_path):
    torch.manual_seed(42)
    original = make_ouro(key)
    with LazyInitContext(default_device="cpu"):
        lazy = make_ouro(key, apply_policy=False)
    assert all(isinstance(p, LazyTensor) for p in lazy.parameters())
    LazyInitContext.materialize(lazy)
    lazy.load_state_dict(original.state_dict(), strict=True)
    batch = ouro_data()
    torch.testing.assert_close(lazy(**batch).logits, original(**batch).logits, rtol=0, atol=0)
    original.save_pretrained(tmp_path, max_shard_size="100KB")
    # Exercise the generic resume tool's exact default loading call.
    restored = type(original).from_pretrained(tmp_path, local_files_only=True)
    torch.testing.assert_close(restored(**batch).logits, original(**batch).logits, rtol=0, atol=0)
    assert restored.config._attn_implementation == "eager"


def test_registered_entries():
    from tests.kit.model_zoo import model_zoo

    for key in ("ouro14", "ouro26"):
        entry = model_zoo[f"transformers_{key}_for_causal_lm"]
        assert entry[4].numerics.reference_fn is not None
        assert entry[4].numerics.token_loss is not None


@pytest.mark.parametrize("key", ["ouro14", "ouro26"])
def test_registered_lazy_contract(key, monkeypatch):
    from colossalai.lazy.lazy_init import _MyTensor
    from tests.kit.model_zoo import model_zoo
    from tests.test_lazy.lazy_init_utils import check_lazy_init

    entry = model_zoo[f"transformers_{key}_for_causal_lm"]
    with LazyInitContext(default_device="cpu"):
        model = entry[0]()
        assert all(isinstance(p, LazyTensor) for p in model.parameters())
    # The existing model-zoo contract sets global test hooks; restore them even
    # on failure so this test cannot alter later constructors.
    monkeypatch.setattr(_MyTensor, "_pre_op_fn", None)
    monkeypatch.setattr(LazyTensor, "_pre_op_fn", None)
    check_lazy_init(entry, default_device="cpu", check_forward=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gradient_checkpointing_kwargs": {"use_reentrant": True}},
        {"every_n_layers": 2},
        {"offload": True},
    ],
)
def test_checkpoint_boundary(kwargs):
    model = make_ouro()
    with pytest.raises(NotImplementedError, match="non-reentrant"):
        model.gradient_checkpointing_enable(**kwargs)
    assert not model.is_gradient_checkpointing


@pytest.mark.parametrize("key", ["ouro14", "ouro26"])
def test_official_config_without_padding_token(key, tmp_path):
    from colossalai.shardformer.modeling.ouro import get_ouro_model_class
    from tests.kit.remote_models import load_remote_classes, tiny_config_overrides

    raw, Config, Original = load_remote_classes(key)
    raw.pop("torch_dtype", None)
    assert "pad_token_id" not in raw
    overrides = tiny_config_overrides(key)
    overrides.pop("pad_token_id", None)
    raw.update(overrides)
    config = Config.from_dict(raw)
    config._attn_implementation = "eager"
    assert not hasattr(config, "pad_token_id")
    model = get_ouro_model_class(Original)(config)
    assert model.model.embed_tokens.padding_idx is None
    ids = torch.tensor([[0, 5, 7, 0, 11, 8]])
    output = model(input_ids=ids, labels=ids)
    output.loss.backward()
    assert model.model.embed_tokens.weight.grad[0].abs().sum() > 0
    model.save_pretrained(tmp_path)
    restored, info = type(model).from_pretrained(tmp_path, local_files_only=True, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"] and not info["mismatched_keys"]
    torch.testing.assert_close(restored(input_ids=ids, labels=ids).logits, output.logits, rtol=0, atol=0)


pytestmark = pytest.mark.skipif(transformers_version != "5.17.0", reason="Ouro adapter requires Transformers 5.17.0")
