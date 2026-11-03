"""Calibration of the baseline-error multiplier: ratios of correct and of buggy implementations.

The multiplier (3 for float16 / bfloat16, 2 for float32) separates the two only if correct implementations that compute
in another order stay below it and typical bugs land far above it. Each case below is an operator checked with
``check_fwd_bwd`` against a plain reference written from the formula; the largest ratio over its outputs and
gradients is reported for a few seeds.

    python -m tests.kit.numerics.calibrate              # every case, bfloat16 and float16
    python -m tests.kit.numerics.calibrate --dtype bf16 --seeds 5

Correct cases use kernels with another reduction order than the reference (fused attention backends, fused
normalization, the grouped-GEMM experts of transformers); their ratio should be near 1. Buggy cases are the kind a
fused kernel gets wrong (a scale off by 1%, an epsilon, an off-by-one mask); their ratio should be well above the
multiplier. Needs a CUDA device for the fused kernels; cases whose kernel is unavailable are skipped.
"""

import argparse
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .judge import error_multiplier
from .scenarios.independent_impl import check_fwd_bwd


@dataclass
class Case:
    name: str
    correct: bool
    # -> (test_fn, ref_fn, inputs[, expect_calls]), or None if the kernel is unavailable
    build: Callable[[torch.dtype, int], Optional[tuple]]


# ----- references, written from the formulas -----


def naive_attention(q: Tensor, k: Tensor, v: Tensor, scale: float = 1.0, shift: int = 0) -> Tensor:
    d = q.shape[-1]
    scores = q @ k.transpose(-1, -2) / (math.sqrt(d) * scale)
    n = q.shape[-2]
    mask = torch.ones(n, n, dtype=torch.bool, device=q.device).tril(shift)
    scores = scores.masked_fill(~mask, float("-inf"))
    return scores.softmax(-1) @ v


def naive_rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    return x / (x.pow(2).mean(-1, keepdim=True) + eps).sqrt() * weight


def naive_layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    mean = x.mean(-1, keepdim=True)
    var = (x - mean).pow(2).mean(-1, keepdim=True)
    return (x - mean) / (var + eps).sqrt() * weight + bias


def naive_cross_entropy(logits: Tensor, labels: Tensor, smoothing: float = 0.0) -> Tensor:
    logp = logits - logits.logsumexp(-1, keepdim=True)
    nll = -logp.gather(-1, labels[:, None]).squeeze(-1)
    return ((1 - smoothing) * nll - smoothing * logp.mean(-1)).mean()


class _Fn(nn.Module):
    """A function with its own weights, so that parameter gradients are compared too."""

    def __init__(self, fn: Callable, **weights: Tensor):
        super().__init__()
        self.fn = fn
        for name, w in weights.items():
            self.register_parameter(name, nn.Parameter(w.clone()))

    def forward(self, **inputs):
        return self.fn(**inputs, **dict(self.named_parameters()))


class _FirstOutput(nn.Module):
    # a module, not a closure, so that check_fwd_bwd copies and casts it for each run
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, **inputs):
        out = self.module(**inputs)
        return out[0] if isinstance(out, tuple) else out


# ----- cases -----


def _device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _qkv(seed: int) -> Dict[str, Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {name: torch.randn(2, 4, 128, 64, generator=g).to(_device()) for name in "qkv"}


def _sdpa_case(backend_name: str):
    def build(dtype, seed):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        backend = getattr(SDPBackend, backend_name)

        def test_fn(q, k, v):
            with sdpa_kernel(backend):
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        if _device().type != "cuda":
            return None
        try:
            probe = {k: v.to(dtype) for k, v in _qkv(0).items()}
            test_fn(**probe)
        except RuntimeError:
            return None  # this backend does not support the dtype or the device
        return test_fn, naive_attention, _qkv(seed)

    return build


def _attention_bug(**bug):
    def build(dtype, seed):
        return (lambda q, k, v: naive_attention(q, k, v, **bug)), naive_attention, _qkv(seed)

    return build


def _norm_inputs(seed: int, bias: bool = False) -> tuple:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(256, 512, generator=g).to(_device())
    weights = {"weight": 1 + 0.1 * torch.randn(512, generator=g).to(_device())}
    if bias:
        weights["bias"] = 0.1 * torch.randn(512, generator=g).to(_device())
    return {"x": x}, weights


def _rms_norm_case(eps: float):
    def build(dtype, seed):
        if not hasattr(F, "rms_norm"):
            return None
        inputs, weights = _norm_inputs(seed)
        test = _Fn(lambda x, weight: F.rms_norm(x, (x.shape[-1],), weight, eps=eps), **weights)
        ref = _Fn(lambda x, weight: naive_rms_norm(x, weight, eps=1e-6), **weights)
        return test, ref, inputs

    return build


def _layer_norm_case(eps: float):
    def build(dtype, seed):
        inputs, weights = _norm_inputs(seed, bias=True)
        test = _Fn(lambda x, weight, bias: F.layer_norm(x, (x.shape[-1],), weight, bias, eps=eps), **weights)
        ref = _Fn(lambda x, weight, bias: naive_layer_norm(x, weight, bias, eps=1e-5), **weights)
        return test, ref, inputs

    return build


def _cross_entropy_case(smoothing: float):
    def build(dtype, seed):
        g = torch.Generator().manual_seed(seed)
        logits = (3 * torch.randn(512, 1000, generator=g)).to(_device())
        labels = torch.randint(0, 1000, (512,), generator=g).to(_device())

        def test_fn(logits, labels):
            return F.cross_entropy(logits.float(), labels, label_smoothing=smoothing).to(logits.dtype)

        def ref_fn(logits, labels):
            return naive_cross_entropy(logits.float(), labels).to(logits.dtype)

        return test_fn, ref_fn, {"logits": logits, "labels": labels}

    return build


def _experts_case(dtype, seed):
    # transformers' grouped-GEMM experts kernel against its own eager loop. The kernel is chosen at forward time from
    # the module's config, so each side gets its own config; expect_calls makes sure the test side really ran it.
    try:
        import transformers
        from transformers.integrations import moe
    except ImportError:
        return None
    if _device().type != "cuda" or not hasattr(moe, "_grouped_mm"):
        return None

    def block(implementation):
        config = transformers.MixtralConfig(
            hidden_size=64,
            intermediate_size=128,
            num_local_experts=8,
            num_experts_per_tok=2,
            num_hidden_layers=1,
            num_attention_heads=4,
            vocab_size=128,
            experts_implementation=implementation,
        )
        return transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock(config).to(_device())

    torch.manual_seed(seed)
    ref = block("eager")
    # a bare block is not initialized (from_pretrained / PreTrainedModel init does that): the experts are empty memory
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0, 0.1)
    test = block("grouped_mm")
    test.load_state_dict(ref.state_dict())
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(2, 64, 64, generator=g).to(_device())

    return _FirstOutput(test), _FirstOutput(ref), {"hidden_states": hidden}, [(moe, "_grouped_mm")]


CASES: List[Case] = [
    Case("sdpa flash vs naive attention", True, _sdpa_case("FLASH_ATTENTION")),
    Case("sdpa efficient vs naive attention", True, _sdpa_case("EFFICIENT_ATTENTION")),
    Case("sdpa math vs naive attention", True, _sdpa_case("MATH")),
    Case("F.rms_norm vs naive", True, _rms_norm_case(1e-6)),
    Case("F.layer_norm vs naive", True, _layer_norm_case(1e-5)),
    Case("F.cross_entropy vs naive", True, _cross_entropy_case(0.0)),
    Case("grouped_mm experts vs eager", True, _experts_case),
    Case("attention scale off by 1%", False, _attention_bug(scale=1.01)),
    Case("causal mask off by one", False, _attention_bug(shift=1)),
    Case("rms_norm eps 1e-2 instead of 1e-6", False, _rms_norm_case(1e-2)),
    Case("layer_norm eps 1e-2 instead of 1e-5", False, _layer_norm_case(1e-2)),
    Case("cross_entropy label smoothing 0.01", False, _cross_entropy_case(0.01)),
]


def calibrate(dtypes: List[torch.dtype], seeds: int = 3) -> List[dict]:
    """Run every case at every dtype for ``seeds`` seeds; one row per case and dtype."""
    rows = []
    for case in CASES:
        for dtype in dtypes:
            ratios, passed = [], []
            for seed in range(seeds):
                built = case.build(dtype, seed)
                if built is None:
                    break
                test_fn, ref_fn, inputs, *expect = built
                try:
                    report = check_fwd_bwd(test_fn, ref_fn, inputs, dtype, expect_calls=expect[0] if expect else ())
                except (RuntimeError, NotImplementedError) as exc:  # e.g. a kernel without this dtype
                    rows.append(dict(case=case.name, correct=case.correct, dtype=str(dtype), unavailable=str(exc)))
                    ratios = None
                    break
                finite = [e.ratio for e in report.entries if e.ratio is not None and math.isfinite(e.ratio)]
                ratios.append(max(finite) if finite else float("nan"))
                passed.append(report.passed)
            if ratios is None:
                continue
            rows.append(
                dict(
                    case=case.name,
                    correct=case.correct,
                    dtype=str(dtype).replace("torch.", ""),
                    multiplier=error_multiplier(dtype, 1 << 20),
                    ratios=ratios,
                    verdicts=passed,
                )
            )
    return rows


def format_rows(rows: List[dict]) -> str:
    lines = [f"{'case':<40} {'kind':<8} {'dtype':<9} {'max ratio per seed':<28} verdict"]
    for r in rows:
        kind = "correct" if r["correct"] else "bug"
        if not r.get("ratios"):
            lines.append(f"{r['case']:<40} {kind:<8} {r['dtype']:<9} {'(kernel unavailable)':<28} -")
            continue
        ratios = " ".join(f"{x:.2f}" for x in r["ratios"])
        right = all(v == r["correct"] for v in r["verdicts"])
        verdict = "as expected" if right else ("FALSE ALARM" if r["correct"] else "MISSED")
        lines.append(f"{r['case']:<40} {kind:<8} {r['dtype']:<9} {ratios:<28} {verdict}")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dtype", choices=["bf16", "fp16", "both"], default="both")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args(argv)
    dtypes = {"bf16": [torch.bfloat16], "fp16": [torch.float16], "both": [torch.bfloat16, torch.float16]}[args.dtype]
    print(format_rows(calibrate(dtypes, args.seeds)))


if __name__ == "__main__":
    main()
