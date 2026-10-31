import torch
import torch.distributed as dist
from verify_image import verify_transformers_peft


def test_cpu_lora_checkpoint_roundtrip():
    """The image smoke check must exercise local and distributed adapter reloads."""
    threads = torch.get_num_threads()
    rng_state = torch.random.get_rng_state().clone()
    result = verify_transformers_peft()
    assert result["device"] == "cpu"
    assert result["steps"] == 2
    assert result["checkpoint_checks"] == ["local-bin", "local-safetensors", "gloo-bin", "gloo-safetensors"]
    assert not dist.is_initialized()
    assert torch.get_num_threads() == threads
    assert torch.equal(torch.random.get_rng_state(), rng_state)
