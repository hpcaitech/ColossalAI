"""Pinned LLaDA2 tiny training probe; same-position masked CE, not the full diffusion objective."""
import argparse
import hashlib
import json
import os
import traceback
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from huggingface_hub import snapshot_download
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from check_remote_tiny import MODELS, tiny_overrides

def require(value, message):
    if not value:
        raise AssertionError(message)

def probe(key, seed):
    repo, revision = MODELS[key]
    result = {"key": key, "repo_id": repo, "revision": revision, "seed": seed, "passed": False,
              "stage": "snapshot", "scope": "CPU/FP32/eager/train/no cache/tiny/fixed masked same-position CE"}
    try:
        torch.manual_seed(seed)
        path = Path(snapshot_download(repo, revision=revision, local_files_only=True))
        raw = json.loads((path / "config.json").read_text())
        config_class = get_class_from_dynamic_module(raw["auto_map"]["AutoConfig"], str(path), local_files_only=True)
        raw.pop("torch_dtype", None)
        raw.update(tiny_overrides(key))
        config = config_class(**raw)
        config._attn_implementation = "eager"
        cls = get_class_from_dynamic_module(raw["auto_map"]["AutoModelForCausalLM"], str(path), local_files_only=True)
        result["model_code_sha256"] = hashlib.sha256((path / "modeling_llada2_moe.py").read_bytes()).hexdigest()
        result["stage"] = "build"
        model = cls(config).float().cpu().train()
        params = {name: p for name, p in model.named_parameters() if p.requires_grad}
        count = sum(p.numel() for p in params.values())
        require(count < 1_000_000, f"Tiny size guard failed: {count}")
        require(all(p.device.type == "cpu" and p.dtype == torch.float32 for p in params.values()), "Wrong device/dtype")
        result["parameter_count"] = count
        print("BUILD_OK", key, count, flush=True)
        routes, supervised_routes, expert_tokens, activations = {}, {}, {}, {}
        with ExitStack() as stack:
            for name, module in model.named_modules():
                if type(module).__name__ == "LLaDA2MoeGate":
                    def gate_hook(module, inputs, output, name=name):
                        ids = output[0].detach()
                        routes[name] = torch.bincount(ids.flatten(), minlength=config.num_experts).tolist()
                        supervised_ids = ids.reshape(*masked.shape, -1)[masked]
                        supervised_routes[name] = torch.bincount(supervised_ids.flatten(), minlength=config.num_experts).tolist()
                    stack.callback(module.register_forward_hook(gate_hook).remove)
                if ".experts." in name and type(module).__name__ == "LLaDA2MoeMLP":
                    def expert_hook(module, inputs, output, name=name):
                        expert_tokens[name] = int(inputs[0].shape[0])
                    stack.callback(module.register_forward_hook(expert_hook).remove)
                if type(module).__name__ == "LLaDA2MoeSparseMoeBlock":
                    def moe_hook(module, inputs, output, name=name):
                        require(module.training, "MoE used inference mode")
                        output[0].retain_grad()
                        activations[name] = output[0]
                    stack.callback(module.register_forward_hook(moe_hook).remove)
            original_ids = torch.randint(5, config.vocab_size, (2, 16))
            masked = torch.zeros_like(original_ids, dtype=torch.bool)
            masked[:, ::2] = True
            input_ids = original_ids.clone()
            input_ids[masked] = config.mask_token_id
            labels = original_ids.clone()
            labels[~masked] = -100
            result.update(input_ids=input_ids.tolist(), labels=labels.tolist(), masked_tokens=int(masked.sum()))
            optimizer = torch.optim.AdamW(params.values(), lr=1e-3, weight_decay=0.0)
            optimizer.zero_grad(set_to_none=True)
            result["stage"] = "forward"
            output = model(input_ids=input_ids, labels=labels, use_cache=False, return_dict=True)
            require(tuple(output.logits.shape) == (2, 16, 128), "Wrong logits shape")
            require(torch.isfinite(output.logits).all().item(), "Nonfinite logits")
            require(output.loss is not None and output.loss.requires_grad and torch.isfinite(output.loss).item(),
                    "Invalid differentiable loss")
            manual = F.cross_entropy(output.logits[masked], original_ids[masked])
            torch.testing.assert_close(output.loss, manual, atol=1e-6, rtol=1e-5)
            require(len(routes) == 1 and len(activations) == 1, "Expected one MoE layer")
            active_experts, supervised_experts = [], []
            for gate_name, histogram in routes.items():
                prefix = gate_name.rsplit(".", 1)[0]
                require(sum(histogram) == input_ids.numel() * config.num_experts_per_tok, "Routing count mismatch")
                for expert_id, tokens in enumerate(histogram):
                    name = f"{prefix}.experts.{expert_id}"
                    require(expert_tokens.get(name) == tokens, f"Expert dispatch mismatch: {name}")
                    if tokens:
                        active_experts.append(name)
                    if supervised_routes[gate_name][expert_id]:
                        supervised_experts.append(name)
            result.update(loss=output.loss.item(), manual_loss=manual.item(),
                          loss_abs_diff=(output.loss.detach()-manual.detach()).abs().item(),
                          router_token_counts=routes, supervised_router_token_counts=supervised_routes,
                          expert_input_tokens=expert_tokens)
            print("FORWARD_OK; SAME_POSITION_LOSS_OK", output.loss.item(), flush=True)
            print("ROUTING_OK", routes, "supervised:", supervised_routes, flush=True)
            result["stage"] = "backward"
            output.loss.backward()
            missing = [n for n, p in params.items() if p.grad is None]
            invalid = [n for n, p in params.items() if p.grad is not None and not torch.isfinite(p.grad).all().item()]
            unexpected_missing = [n for n in missing if ".experts." not in n or
                                  any(n.startswith(prefix + ".") for prefix in active_experts)]
            require(not unexpected_missing and not invalid, f"Gradient failure: {unexpected_missing}, {invalid}")
            result["moe_unmasked_output_grad_max"] = {}
            for name, output_tensor in activations.items():
                require(output_tensor.grad is not None and torch.isfinite(output_tensor.grad).all().item()
                        and torch.count_nonzero(output_tensor.grad).item() > 0, f"MoE disconnected: {name}")
                # This tiny has its sole MoE in the last layer; only pointwise norm/head follow.
                # Tokens excluded from the CE therefore receive zero gradient here.
                unmasked_max = output_tensor.grad[~masked].abs().max().item()
                result["moe_unmasked_output_grad_max"][name] = unmasked_max
                require(unmasked_max == 0.0, "Unexpected gradient at an unsupervised last-layer token")
            nonzero = [n for n, p in params.items() if p.grad is not None and torch.count_nonzero(p.grad).item() > 0]
            groups = ["model.word_embeddings.", "model.norm.", "lm_head.",
                      "model.layers.0.", "model.layers.1.",
                      *[n + "." for n in routes],
                      *[n + ".shared_experts." for n in activations],
                      *[n + "." for n in supervised_experts]]
            result.update(missing_gradients=missing, invalid_gradients=invalid,
                          zero_gradients=[n for n, p in params.items() if p.grad is not None and n not in nonzero],
                          parameter_grad_norms={n: None if p.grad is None else p.grad.norm().item() for n, p in params.items()},
                          nonzero_gradient_parameters=nonzero)
            missing_groups = [prefix for prefix in groups if not any(n.startswith(prefix) for n in nonzero)]
            result["missing_nonzero_groups"] = missing_groups
            require(not missing_groups, f"Required groups without nonzero gradients: {missing_groups}")
            print("BACKWARD_OK; nonzero parameter tensors:", len(nonzero), flush=True)
            result["stage"] = "step"
            before = {n: p.detach().clone() for n, p in params.items()}
            optimizer.step()
            require(all(torch.isfinite(p).all().item() for p in params.values()), "Nonfinite updated weights")
            changed = [n for n, p in params.items() if not torch.equal(before[n], p.detach())]
            require(all(any(n.startswith(prefix) for n in changed) for prefix in groups), "Required group did not update")
            result["updated_parameters"] = changed
            result["optimizer"] = {"name": "AdamW", "lr": 0.001, "weight_decay": 0.0}
            print("STEP_OK; updated parameter tensors:", len(changed), flush=True)
        result.update(passed=True, stage="complete")
    except Exception as e:
        result.update(error_type=type(e).__name__, error=str(e), traceback=traceback.format_exc())
        traceback.print_exc()
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["llada21", "llada22"], default="llada21")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    print("Torch:", torch.__version__, "Transformers:", transformers.__version__, flush=True)
    result = probe(args.model, args.seed)
    report = args.report or Path(os.environ.get("D2D1_LOG_DIR", "/results/2d1")) / (
        f"llada2-backward-{args.model}-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x") as f:
        json.dump({"torch": torch.__version__, "transformers": transformers.__version__, "result": result},
                  f, indent=2, ensure_ascii=False)
    print("JSON report:", report)
    print(args.model, "PASS" if result["passed"] else "FAIL", result["stage"])
    return int(not result["passed"])

if __name__ == "__main__":
    raise SystemExit(main())
