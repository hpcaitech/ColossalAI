"""Validate the dependency image without importing a checkout or accessing GPUs."""

import importlib
import importlib.metadata as metadata
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from packaging.requirements import Requirement
from packaging.version import Version


def verify_lora_checkpoints(model, base_model, tokens):
    """Check adapter reload and resumed updates, including the distributed PEFT path."""
    import torch
    import torch.distributed as dist
    from peft import PeftModel, get_peft_model_state_dict

    assert dist.is_available() and dist.is_gloo_available(), "CPU LoRA checkpoint checks require Gloo"
    assert not dist.is_initialized(), "Run image validation outside an existing process group"
    model.eval()
    with torch.no_grad():
        reference_logits = model(input_ids=tokens).logits
    expected_adapter = get_peft_model_state_dict(model)
    checked = []
    with TemporaryDirectory(prefix="ci-lora-checkpoint-") as temporary:
        for distributed in (False, True):
            if distributed:
                # PEFT 0.19.1 only takes the broken TP import path when a process
                # group exists, even for a non-TP model. One CPU rank reproduces it.
                dist.init_process_group(
                    "gloo",
                    init_method=Path(temporary, "rendezvous").as_uri(),
                    rank=0,
                    world_size=1,
                    timeout=timedelta(seconds=30),
                )
            try:
                for safe_serialization in (False, True):
                    mode = f"{'gloo' if distributed else 'local'}-{'safetensors' if safe_serialization else 'bin'}"
                    checkpoint = Path(temporary, mode)
                    model.save_pretrained(checkpoint, safe_serialization=safe_serialization)
                    expected_file = "adapter_model.safetensors" if safe_serialization else "adapter_model.bin"
                    assert (checkpoint / expected_file).is_file(), f"Missing {mode} checkpoint"
                    restored = PeftModel.from_pretrained(
                        deepcopy(base_model), checkpoint, is_trainable=True, local_files_only=True
                    )
                    actual_adapter = get_peft_model_state_dict(restored)
                    assert actual_adapter.keys() == expected_adapter.keys(), f"Adapter keys changed for {mode}"
                    for name, expected in expected_adapter.items():
                        torch.testing.assert_close(actual_adapter[name], expected, rtol=0, atol=0)
                    restored.eval()
                    with torch.no_grad():
                        torch.testing.assert_close(restored(input_ids=tokens).logits, reference_logits)
                    restored.train()
                    trainable = [parameter for parameter in restored.parameters() if parameter.requires_grad]
                    assert trainable, f"Reloaded {mode} adapter is not trainable"
                    before = [parameter.detach().clone() for parameter in trainable]
                    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
                    optimizer.zero_grad(set_to_none=True)
                    loss = restored(input_ids=tokens, labels=tokens).loss
                    assert torch.isfinite(loss), f"Non-finite resumed loss for {mode}"
                    loss.backward()
                    gradients = [parameter.grad for parameter in trainable]
                    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
                    assert any(torch.count_nonzero(gradient).item() for gradient in gradients)
                    optimizer.step()
                    assert any(not torch.equal(old, new) for old, new in zip(before, trainable))
                    checked.append(mode)
            finally:
                if distributed:
                    dist.destroy_process_group()
    return checked


def verify_transformers_peft():
    """Exercise offline CPU LoRA training, checkpoint reload and resumed training."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            config = LlamaConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
                attn_implementation="eager",
                use_cache=False,
            )
            base_model = LlamaForCausalLM(config).to(device="cpu", dtype=torch.float32)
            model = get_peft_model(
                deepcopy(base_model),
                LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]),
            )
            model.train()
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            assert trainable, "LoRA has no trainable parameters"
            assert all(parameter.device.type == "cpu" for parameter in model.parameters())
            before = [parameter.detach().clone() for parameter in trainable]
            optimizer = torch.optim.AdamW(trainable, lr=1e-3)
            tokens = torch.randint(0, config.vocab_size, (2, 16), device="cpu")
            losses = []
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                loss = model(input_ids=tokens, labels=tokens).loss
                assert torch.isfinite(loss), "LoRA loss is not finite"
                loss.backward()
                gradients = [parameter.grad for parameter in trainable]
                assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
                assert any(
                    torch.count_nonzero(gradient).item() for gradient in gradients
                ), "LoRA gradients are all zero"
                optimizer.step()
                losses.append(loss.item())
            assert any(not torch.equal(old, new) for old, new in zip(before, trainable)), "LoRA weights did not update"
            checkpoints = verify_lora_checkpoints(model, base_model, tokens)
            return {"device": "cpu", "steps": len(losses), "losses": losses, "checkpoint_checks": checkpoints}
    finally:
        torch.set_num_threads(previous_threads)


def verify_diffusers_pipelines():
    """Resolve lazy pipeline imports used by ColossalAI, without downloading models."""
    import diffusers

    pipelines = ("DiffusionPipeline", "PixArtAlphaPipeline", "StableDiffusion3Pipeline")
    for name in pipelines:
        assert issubclass(getattr(diffusers, name), diffusers.DiffusionPipeline)
    return list(pipelines)


def main():
    assert sys.version_info[:2] == (3, 10), sys.version
    root = Path(__file__).resolve().parent
    versions = {}
    for lock_name in ("requirements.lock", "requirements-torch.lock"):
        for line in (root / lock_name).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            requirement = Requirement(line.split(" --hash=", 1)[0])
            actual = metadata.version(requirement.name)
            assert Version(actual) in requirement.specifier, (requirement.name, actual, str(requirement.specifier))
            versions[requirement.name] = actual

    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    for executable in ("nvcc", "g++", "cmake", "ninja", "git"):
        assert shutil.which(executable), f"Missing build tool: {executable}"
    nvcc = subprocess.check_output(["nvcc", "--version"], text=True)
    assert "release 13.0" in nvcc, nvcc

    # A base image must not silently supply a stale ColossalAI package or AOT .so.
    try:
        metadata.distribution("colossalai")
    except metadata.PackageNotFoundError:
        pass
    else:
        raise AssertionError("Install ColossalAI from the CI source SHA at runtime, not in this image")

    for module in ("transformers", "peft", "diffusers", "triton", "testmon.pytest_testmon"):
        importlib.import_module(module)

    import torch
    import torchaudio
    import torchvision

    assert torch.version.cuda == "13.0", torch.version.cuda
    # CPU extension smoke checks catch some ABI mismatches that pip check cannot.
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    assert torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8]), 0.5).tolist() == [0]
    waveform = torch.zeros(1, 160)
    assert torchaudio.functional.resample(waveform, 16000, 8000).shape == (1, 80)
    hf_smoke = verify_transformers_peft()
    diffusion_imports = verify_diffusers_pipelines()
    # pytest --help loads installed entry points, including pytest-testmon.
    subprocess.run([sys.executable, "-m", "pytest", "--help"], check=True, stdout=subprocess.DEVNULL)
    print(
        json.dumps(
            {
                "python": sys.version,
                "torch_cuda": torch.version.cuda,
                "packages": versions,
                "hf_smoke": hf_smoke,
                "diffusion_pipeline_imports": diffusion_imports,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
