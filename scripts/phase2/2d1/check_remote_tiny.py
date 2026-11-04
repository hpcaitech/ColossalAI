"""Probe pinned remote-code models on CPU; optional temporary Ouro diagnostic adapters."""

import argparse
import gc
import json
import os
import sys
import traceback
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers.dynamic_module_utils import get_class_from_dynamic_module

MODELS = {
    "llada21": ("inclusionAI/LLaDA2.1-mini", "20e64e2ad21644d0e5248586ed9c942cdd45de0f"),
    "llada22": ("inclusionAI/LLaDA2.2-mini", "e4f77d25ab5ca86f2223c6cdc6a72b887e20c8a5"),
    "ouro14": ("ByteDance/Ouro-1.4B", "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"),
    "ouro26": ("ByteDance/Ouro-2.6B", "1ed04250da1a9936042725d302e81c8fa2ab5abd"),
}


def tiny_overrides(key):
    values = {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2 if key.startswith("llada") else 4,
        "head_dim": 16,
        "max_position_embeddings": 64,
        "max_window_layers": 2,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 1,
        "mask_token_id": 2,
        "use_cache": False,
        "tie_word_embeddings": False,
        "attention_dropout": 0.0,
        "embedding_dropout": 0.0,
        "output_dropout": 0.0,
        "use_sliding_window": False,
        "dtype": "float32",
    }
    if key.startswith("llada"):
        values.update(
            {
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "num_shared_experts": 1,
                "n_group": 2,
                "topk_group": 1,
                "moe_intermediate_size": 32,
                "first_k_dense_replace": 1,
                "rotary_dim": 8,
                "partial_rotary_factor": 0.5,
            }
        )
        if key == "llada22":
            values.update({"block_size": 4, "expert_capacity": 4, "delete_token_id": 3, "split_token_id": 4})
    else:
        values.update({"layer_types": ["full_attention"] * 2, "total_ut_steps": 2})
    return values


def probe(key, compat="none"):
    repo_id, revision = MODELS[key]
    result = {
        "key": key,
        "repo_id": repo_id,
        "revision": revision,
        "mode": compat if key.startswith("ouro") else "original",
        "passed": False,
        "stage": "snapshot",
    }
    print(f"\n========== {repo_id} ==========", flush=True)
    stack = ExitStack()
    try:
        torch.manual_seed(42)
        path = snapshot_download(repo_id, revision=revision, local_files_only=True)
        raw = json.loads((Path(path) / "config.json").read_text())
        config_reference = raw["auto_map"]["AutoConfig"]
        model_reference = raw["auto_map"]["AutoModelForCausalLM"]
        print("快照版本:", revision)
        result["stage"] = "config"
        config_class = get_class_from_dynamic_module(config_reference, path, local_files_only=True)
        overrides = tiny_overrides(key)
        raw.pop("torch_dtype", None)
        raw.update(overrides)
        config = config_class(**raw)
        config._attn_implementation = "eager"
        result["tiny_overrides"] = overrides
        print("TINY_CONFIG:", json.dumps(overrides, ensure_ascii=False))
        print("CONFIG_OK")

        result["stage"] = "import"
        model_class = get_class_from_dynamic_module(model_reference, path, local_files_only=True)
        print("IMPORT_OK:", model_class.__name__)
        if compat != "none" and key.startswith("ouro"):
            from ouro_compat_probe import ouro_compat

            stack.enter_context(ouro_compat(sys.modules[model_class.__module__], config, compat))
            print("DIAGNOSTIC_ADAPTER:", compat)
        result["stage"] = "build"
        model = model_class(config).float().cpu().eval()
        parameter_count = sum(p.numel() for p in model.parameters())
        assert parameter_count < 1_000_000, f"Tiny config did not take effect: {parameter_count}"
        assert all(p.device.type == "cpu" for p in model.parameters())
        result["parameter_count"] = parameter_count
        print("BUILD_OK; 参数数量:", parameter_count)

        expected_class = "LLaDA2MoeSparseMoeBlock" if key.startswith("llada") else "OuroDecoderLayer"
        calls = {}
        handles = []
        for name, module in model.named_modules():
            if type(module).__name__ == expected_class:
                calls[name] = 0

                def count_call(module, inputs, output, name=name):
                    calls[name] += 1

                handles.append(module.register_forward_hook(count_call))
        assert calls, f"Expected architecture component missing: {expected_class}"

        result["stage"] = "forward"
        input_ids = torch.randint(5, 128, (2, 16))
        try:
            with torch.no_grad():
                output = model(input_ids=input_ids, use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        logits = output.logits
        assert tuple(logits.shape) == (2, 16, 128), tuple(logits.shape)
        assert torch.isfinite(logits).all().item(), "Non-finite logits"
        if key.startswith("llada"):
            assert all(count >= 1 for count in calls.values()), calls
        else:
            assert len(calls) == 2 and all(count == 2 for count in calls.values()), calls
        if key.startswith("ouro"):
            # Changing future tokens must leave earlier causal outputs unchanged.
            changed_ids = input_ids.clone()
            changed_ids[:, 8:] = (changed_ids[:, 8:] - 5 + 1) % 123 + 5
            with torch.no_grad():
                changed_logits = model(input_ids=changed_ids, use_cache=False, return_dict=True).logits
            delta = (logits[:, :8] - changed_logits[:, :8]).abs().max().item()
            torch.testing.assert_close(logits[:, :8], changed_logits[:, :8], atol=1e-6, rtol=1e-5)
            result["causal_prefix_max_abs_diff"] = delta
            print("CAUSAL_PREFIX_OK; max_abs_diff:", delta)
        result.update(
            {"passed": True, "stage": "complete", "logits_shape": list(logits.shape), "component_calls": calls}
        )
        print("关键模块调用次数:", calls)
        print("FORWARD_OK:", tuple(logits.shape))
    except Exception as error:
        result.update({"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        print("FAILED; stage:", result["stage"])
        traceback.print_exc()
    finally:
        stack.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", *MODELS], default="all")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--compat", choices=["none", "rope-registry", "rope", "rope-mask"], default="none")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    print("Torch:", torch.__version__)
    print("Transformers:", transformers.__version__)
    print("Scope: CPU / FP32 / eager / no KV cache / tiny eval forward")
    print("Ouro diagnostic adapter:", args.compat)
    results = []
    for key in MODELS if args.model == "all" else [args.model]:
        results.append(probe(key, args.compat))
        gc.collect()
    report = args.report
    if report is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        report = (
            Path(os.environ.get("D2D1_LOG_DIR", "/results/2d1"))
            / f"remote-tiny-{args.compat}-{stamp}-{os.getpid()}.json"
        )
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x") as stream:
        json.dump(
            {"torch": torch.__version__, "transformers": transformers.__version__, "results": results},
            stream,
            indent=2,
            ensure_ascii=False,
        )
    print("\nJSON report:", report)
    for result in results:
        print(result["key"], "PASS" if result["passed"] else "FAIL", result["stage"], result.get("error", ""))
    return int(any(not result["passed"] for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
