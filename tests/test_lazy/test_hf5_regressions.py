"""Lazy construction regressions requiring no GPU or model downloads."""

import subprocess
import sys
from typing import Optional, get_args

import pytest
import torch

from colossalai.lazy import LazyInitContext, LazyTensor


@pytest.mark.parametrize("name,dtype", [("LongTensor", torch.int64), ("FloatTensor", torch.float32)])
def test_legacy_constructor_annotations(name, dtype):
    original = getattr(torch, name)
    real_tensor = original([1, 2, 3])
    with LazyInitContext():
        constructor = getattr(torch, name)
        assert isinstance(constructor, type)
        assert isinstance(real_tensor, constructor)
        assert get_args(constructor | None) == (original, type(None))
        assert get_args(None | constructor) == (type(None), original)
        assert get_args(Optional[constructor])
        x = constructor([1, 2, 3])
        sized = constructor(3)
        empty = constructor()
        assert isinstance(x, LazyTensor)
    assert getattr(torch, name) is original
    torch.testing.assert_close(x.materialize(), torch.tensor([1, 2, 3], dtype=dtype))
    assert sized.materialize().shape == (3,)
    assert empty.materialize().shape == (0,)


def test_copy_does_not_modify_source_history():
    with LazyInitContext():
        source = torch.arange(4, dtype=torch.float32)
        a, b = torch.zeros(4), torch.zeros(4)
        a.copy_(source)
        b.copy_(source)
    # Materializing either destination must not recurse through the source.
    torch.testing.assert_close(b.materialize(), torch.arange(4, dtype=torch.float32))
    torch.testing.assert_close(a.materialize(), torch.arange(4, dtype=torch.float32))
    torch.testing.assert_close(source.materialize(), torch.arange(4, dtype=torch.float32))


def test_inplace_repeated_operand_is_recorded_once():
    with LazyInitContext():
        x = torch.ones(4)
        x.add_(x)
    torch.testing.assert_close(x.materialize(), torch.full((4,), 2.0))


def test_keyword_initializer():
    with LazyInitContext():
        x = torch.empty(4)
        torch.nn.init.constant_(tensor=x, val=3)
    torch.testing.assert_close(x.materialize(), torch.full((4,), 3.0))


@pytest.mark.parametrize("family", ["qwen2", "qwen3"])
def test_hf_cold_import_and_materialize(family):
    # Each family starts with no HF modeling module in sys.modules. Loading it
    # outside the lazy context would hide the annotation regression.
    program = f"""
import copy
import torch
import transformers
from colossalai.lazy import LazyInitContext, LazyTensor
family = {family!r}
prefix = "Qwen2" if family == "qwen2" else "Qwen3"
config = getattr(transformers, prefix + "Config")(vocab_size=32, hidden_size=16,
    intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
    num_key_value_heads=2, head_dim=8)
with LazyInitContext():
    model = getattr(transformers, prefix + "ForCausalLM")(config)
    duplicate = copy.deepcopy(model)
assert isinstance(next(model.parameters()), LazyTensor)
LazyInitContext.materialize(model)
LazyInitContext.materialize(duplicate)
for a, b in zip(model.parameters(), duplicate.parameters()):
    torch.testing.assert_close(a, b, rtol=0, atol=0)
ref = getattr(transformers, prefix + "ForCausalLM")(config)
ref.load_state_dict(model.state_dict())
for name, buffer in model.named_buffers():
    torch.testing.assert_close(buffer, dict(ref.named_buffers())[name], rtol=0, atol=0)
inputs = torch.tensor([[1, 2, 3, 4]])
actual = model(input_ids=inputs, labels=inputs)
expected = ref(input_ids=inputs, labels=inputs)
torch.testing.assert_close(actual.loss, expected.loss)
actual.loss.backward()
expected.loss.backward()
for a, b in zip(model.parameters(), ref.parameters()):
    if a.grad is not None:
        torch.testing.assert_close(a.grad, b.grad)
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
