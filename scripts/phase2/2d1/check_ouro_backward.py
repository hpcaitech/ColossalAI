"""CPU tiny Ouro training smoke check, using the pinned model's labels branch.

The existing diagnostic RoPE/mask adapter stays active through backward/step.
This checks gradient connectivity, not reference gradient accuracy or convergence.
"""

import argparse
import hashlib
import json
import os
import sys
import traceback
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from check_remote_tiny import MODELS, tiny_overrides
from huggingface_hub import snapshot_download
from ouro_compat_probe import ouro_compat
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def require(condition, message):
    # Do not let python -O silently remove diagnostic checks.
    if not condition:
        raise AssertionError(message)


def probe(args):
    repo_id, revision = MODELS[args.model]
    result = {
        "repo_id": repo_id,
        "revision": revision,
        "model": args.model,
        "compat": args.compat,
        "seed": args.seed,
        "passed": False,
        "stage": "snapshot",
        "scope": "CPU / FP32 / eager / train / no cache / two recurrent steps / one AdamW update",
        "loss_path": "official labels branch: exit-probability-weighted logits, next-token cross entropy",
    }
    remote_module = None
    try:
        torch.manual_seed(args.seed)
        path = Path(snapshot_download(repo_id, revision=revision, local_files_only=True))
        raw = json.loads((path / "config.json").read_text())
        print("模型:", repo_id)
        print("快照版本:", revision)
        result["stage"] = "config"
        config_class = get_class_from_dynamic_module(raw["auto_map"]["AutoConfig"], str(path), local_files_only=True)
        overrides = tiny_overrides(args.model)
        raw.pop("torch_dtype", None)
        raw.update(overrides)
        config = config_class(**raw)
        config._attn_implementation = "eager"
        result["tiny_overrides"] = overrides
        print("TINY_CONFIG:", json.dumps(overrides, ensure_ascii=False))
        print("CONFIG_OK")

        result["stage"] = "import"
        model_class = get_class_from_dynamic_module(
            raw["auto_map"]["AutoModelForCausalLM"], str(path), local_files_only=True
        )
        remote_module = sys.modules[model_class.__module__]
        original_registry = remote_module.ROPE_INIT_FUNCTIONS
        original_mask = remote_module.create_causal_mask
        missing = object()
        original_default = vars(remote_module.OuroRotaryEmbedding).get("compute_default_rope_parameters", missing)
        result["model_code_sha256"] = hashlib.sha256(Path(remote_module.__file__).read_bytes()).hexdigest()
        print("IMPORT_OK:", model_class.__name__)
        print("DIAGNOSTIC_ADAPTER:", args.compat)

        with ExitStack() as stack:
            if args.compat != "none":
                stack.enter_context(ouro_compat(remote_module, config, args.compat))
            result["stage"] = "build"
            model = model_class(config).float().cpu().train()
            parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
            require(bool(parameters), "No trainable parameters")
            require(
                all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters()),
                "Expected CPU / FP32 parameters",
            )
            count = sum(p.numel() for p in model.parameters())
            require(count < 1_000_000, f"Tiny configuration did not take effect: {count}")
            require(model.model.total_ut_steps == 2 and len(model.model.layers) == 2, "Expected two layers/two rounds")
            result["parameter_count"] = count
            print("BUILD_OK; 参数数量:", count)

            # Observe the same decoder objects in both rounds, keeping each output's gradient.
            activations = {}
            for name, module in model.named_modules():
                if type(module).__name__ != "OuroDecoderLayer":
                    continue
                activations[name] = []

                def capture(module, inputs, kwargs, output, name=name):
                    require(
                        isinstance(output, torch.Tensor) and output.requires_grad,
                        f"{name}: decoder output is disconnected from autograd",
                    )
                    output.retain_grad()
                    activations[name].append((kwargs.get("current_ut"), output))

                stack.callback(module.register_forward_hook(capture, with_kwargs=True).remove)
            require(len(activations) == 2, "Expected two OuroDecoderLayer objects")

            input_ids = torch.randint(5, config.vocab_size, (2, 16))
            labels = input_ids.clone()
            result["input_ids"] = input_ids.tolist()
            optimizer = torch.optim.AdamW(parameters.values(), lr=1e-3, weight_decay=0.0)
            optimizer.zero_grad(set_to_none=True)

            result["stage"] = "forward"
            # Passing labels selects the model's weighted training branch, including the exit gate.
            output = model(input_ids=input_ids, labels=labels, use_cache=False, return_dict=True)
            logits, loss = output.logits, output.loss
            require(tuple(logits.shape) == (2, 16, 128), f"Unexpected logits shape: {tuple(logits.shape)}")
            require(torch.isfinite(logits).all().item(), "Non-finite logits")
            rounds = {name: [step for step, _ in calls] for name, calls in activations.items()}
            require(all(steps == [0, 1] for steps in rounds.values()), f"Unexpected recurrent rounds: {rounds}")
            result.update(logits_shape=list(logits.shape), decoder_rounds=rounds)
            print("FORWARD_OK:", tuple(logits.shape))
            print("循环层执行轮次:", rounds)

            result["stage"] = "loss"
            require(loss is not None and loss.numel() == 1 and loss.requires_grad, "Missing differentiable scalar loss")
            require(torch.isfinite(loss).item(), "Non-finite loss")
            # Independent loss calculation checks the next-token shift; no padding in this input.
            with torch.no_grad():
                manual_loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, config.vocab_size),
                    labels[:, 1:].reshape(-1),
                )
            torch.testing.assert_close(loss.detach(), manual_loss, atol=1e-6, rtol=1e-5)
            result.update(
                loss=loss.item(),
                manual_next_token_loss=manual_loss.item(),
                loss_abs_diff=(loss.detach() - manual_loss).abs().item(),
            )
            print("LOSS_OK:", loss.item())
            print("LOSS_ALIGNMENT_OK; 手工移位交叉熵:", manual_loss.item())

            result["stage"] = "backward"
            loss.backward()
            absent = [name for name, p in parameters.items() if p.grad is None]
            invalid = [
                name for name, p in parameters.items() if p.grad is not None and not torch.isfinite(p.grad).all().item()
            ]
            result.update(missing_gradients=absent, invalid_gradients=invalid)
            # This pinned labels branch uses every trainable parameter, including early_exit_gate.
            # Other exit/loss branches may legitimately leave the gate disconnected.
            require(not absent, f"Unexpected missing gradients in weighted labels branch: {absent}")
            require(not invalid, f"Non-finite parameter gradients: {invalid}")
            nonzero = [name for name, p in parameters.items() if torch.count_nonzero(p.grad).item() > 0]
            result["zero_gradients"] = [name for name in parameters if name not in nonzero]
            result["parameter_grad_norms"] = {name: p.grad.norm().item() for name, p in parameters.items()}

            round_norms = {}
            for name, calls in activations.items():
                round_norms[name] = []
                for step, activation in calls:
                    require(activation.grad is not None, f"{name} round {step}: missing output gradient")
                    require(
                        torch.isfinite(activation.grad).all().item(), f"{name} round {step}: invalid output gradient"
                    )
                    require(
                        torch.count_nonzero(activation.grad).item() > 0, f"{name} round {step}: zero output gradient"
                    )
                    round_norms[name].append(activation.grad.norm().item())
            result["decoder_output_grad_norms"] = round_norms

            groups = {
                "embedding": "model.embed_tokens.",
                "lm_head": "lm_head.",
                "final_norm": "model.norm.",
                "exit_gate": "model.early_exit_gate.",
                **{name: name + "." for name in activations},
            }
            group_nonzero = {
                group: [name for name in nonzero if name.startswith(prefix)] for group, prefix in groups.items()
            }
            require(all(group_nonzero.values()), f"Missing nonzero gradients in required groups: {group_nonzero}")
            result["nonzero_gradient_parameters"] = nonzero
            print("BACKWARD_OK; 有限梯度参数张量:", len(parameters), "; 非零:", len(nonzero))
            print("各轮输出梯度范数:", round_norms)
            print(
                "EARLY_EXIT_GATE_GRAD_OK:",
                {name: result["parameter_grad_norms"][name] for name in group_nonzero["exit_gate"]},
            )

            result["stage"] = "step"
            before = {name: p.detach().clone() for name, p in parameters.items()}
            optimizer.step()
            require(all(torch.isfinite(p).all().item() for p in parameters.values()), "Non-finite updated parameters")
            changed = [name for name, p in parameters.items() if not torch.equal(before[name], p.detach())]
            group_updates = {
                group: [name for name in changed if name in names] for group, names in group_nonzero.items()
            }
            require(all(group_updates.values()), f"Required groups did not update: {group_updates}")
            result.update(
                updated_parameters=changed,
                updated_groups=group_updates,
                optimizer={"name": "AdamW", "lr": 1e-3, "weight_decay": 0.0},
            )
            print("STEP_OK; 已更新参数张量:", len(changed))
        result.update(passed=True, stage="complete")
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        print("FAILED; stage:", result["stage"])
        traceback.print_exc()
    finally:
        if remote_module is not None:
            restored = (
                remote_module.ROPE_INIT_FUNCTIONS is original_registry
                and remote_module.create_causal_mask is original_mask
                and vars(remote_module.OuroRotaryEmbedding).get("compute_default_rope_parameters", missing)
                is original_default
            )
            result["adapter_restored"] = restored
            if not restored:
                result.update(passed=False, stage="restore", error="Diagnostic adapter was not restored")
            print("ADAPTER_RESTORED:", restored)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["ouro14", "ouro26"], default="ouro14")
    parser.add_argument("--compat", choices=["none", "rope-mask"], default="rope-mask")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    print("Torch:", torch.__version__)
    print("Transformers:", transformers.__version__)
    print("Scope: CPU / FP32 / eager / train / no cache / tiny / one optimizer step")
    result = probe(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    report = args.report or Path(os.environ.get("D2D1_LOG_DIR", "/results/2d1")) / (
        f"ouro-backward-{args.model}-{args.compat}-{stamp}-{os.getpid()}.json"
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x") as stream:
        json.dump(
            {"torch": torch.__version__, "transformers": transformers.__version__, "result": result},
            stream,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    print("JSON report:", report)
    print(args.model, "PASS" if result["passed"] else "FAIL", result["stage"])
    print("范围: 未验证参考梯度数值对齐、真实权重、GPU、KV cache、分布式或收敛。")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
