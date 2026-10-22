import pytest
import torch

from colossalai.lazy import LazyInitContext, LazyTensor
from colossalai.shardformer.modeling.chatglm2_6b.configuration_chatglm import ChatGLMConfig
from colossalai.shardformer.modeling.chatglm2_6b.modeling_chatglm import ChatGLMForConditionalGeneration


def tiny_chatglm(legacy_max_length=None):
    """Build a small CPU model without relying on Transformers' generation defaults."""
    config = ChatGLMConfig(
        num_layers=1,
        padded_vocab_size=64,
        hidden_size=32,
        ffn_hidden_size=64,
        kv_channels=8,
        num_attention_heads=4,
        seq_length=32,
        use_cache=False,
    )
    # Transformers 4 supplies this generation field; Transformers 5 does not.
    # Explicitly remove it so the fallback is covered on either version.
    if hasattr(config, "max_length"):
        del config.max_length
    if legacy_max_length is not None:
        config.max_length = legacy_max_length
    model = ChatGLMForConditionalGeneration(config, empty_init=False, device="cpu")
    for module in model.modules():
        if module.__class__.__name__ == "RMSNorm":
            torch.nn.init.ones_(module.weight)
    return model


def chatglm_inputs():
    """Return deterministic CPU input tokens and next-token labels."""
    tokens = torch.tensor([[1, 2, 3, 4]])
    return {"input_ids": tokens, "labels": tokens.clone()}


@pytest.mark.parametrize("use_cache", [False, True])
def test_chatglm_config_retains_use_cache(use_cache):
    config = ChatGLMConfig(use_cache=use_cache)
    assert config.use_cache is use_cache
    assert ChatGLMConfig.from_dict(config.to_dict()).use_cache is use_cache


@pytest.mark.parametrize("legacy_max_length", [None, 17])
def test_chatglm_sequence_length_and_training_cpu(legacy_max_length):
    model = tiny_chatglm(legacy_max_length)
    assert model.max_sequence_length == (32 if legacy_max_length is None else legacy_max_length)
    output = model(**chatglm_inputs())
    assert output.logits.shape == (1, 4, 64)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() for gradient in gradients)


def test_chatglm_lazy_materialization_cpu():
    context = LazyInitContext(default_device="cpu")
    with context:
        model = tiny_chatglm()
        assert all(isinstance(parameter, LazyTensor) for parameter in model.parameters())
    context.materialize(model)
    assert model.max_sequence_length == 32
    assert all(
        not isinstance(parameter, LazyTensor) and parameter.device.type == "cpu" for parameter in model.parameters()
    )
    output = model(**chatglm_inputs())
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
