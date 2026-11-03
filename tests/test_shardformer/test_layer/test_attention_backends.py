"""Explicit attention contracts and strict GPU acceptance.

CPU: pytest <this file> -k 'not gpu'
FA3 acceptance (missing CUDA/package is a failure):
    COLOSSALAI_TEST_ATTENTION_BACKEND=fa3 pytest <this file> -k gpu
Set COLOSSALAI_ATTENTION_TRACE_DIR to retain CUDA profiler traces as CI artifacts.
GPU skips in an ordinary CPU run do not establish GPU compatibility.
"""

import hashlib
import json
import os
from contextlib import nullcontext
from importlib.metadata import distribution
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from colossalai.kernel import kernel_loader
from colossalai.kernel.kernel_loader import FlashAttentionBackendLoader
from colossalai.shardformer.layer.attn import AttnMaskType, ColoAttention
from tests.kit.numerics import assert_aligned, run_fwd_bwd
from tests.kit.numerics.scenarios import independent_impl


def _reference(q, k, v, *, causal=False, alignment=None, mask=None, scale=None):
    """Independent formula, including GQA and all-masked rows; no production helpers."""
    groups = q.shape[1] // k.shape[1]
    k, v = (x.repeat_interleave(groups, dim=1) for x in (k, v))
    # CUDA attention accumulates QK and computes softmax at higher precision.
    # Do not inflate the allowed error with an already rounded low-precision QK.
    compute_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    scores = q.to(compute_dtype) @ k.to(compute_dtype).transpose(-2, -1)
    scores = scores * (q.shape[-1] ** -0.5 if scale is None else scale)
    keep = torch.ones_like(scores, dtype=torch.bool)
    if causal:
        sq, sk = q.shape[-2], k.shape[-2]
        offset = sk - sq if alignment == "lower_right" else 0
        # Deliberately construct the reference by rows, not the production helper.
        causal_mask = torch.tensor([[j <= i + offset for j in range(sk)] for i in range(sq)], device=q.device)
        keep = keep & causal_mask
    if mask is not None:
        if mask.dtype == torch.bool:
            keep = keep & mask
        else:
            scores = scores + mask
            keep = keep & ~torch.isneginf(mask)
    scores = scores.masked_fill(~keep, float("-inf"))
    # Do not backpropagate through softmax([-inf, ...]); the all-masked result is zero.
    scores = torch.where(keep.any(dim=-1, keepdim=True), scores, torch.zeros_like(scores))
    softmax_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    weights = scores.softmax(dim=-1, dtype=softmax_dtype).to(q.dtype).masked_fill(~keep, 0)
    return weights @ v


def _packed_reference(q, k, v, q_lengths, kv_lengths, **kwargs):
    outputs = []
    qs = ks = 0
    for q_len, kv_len in zip(q_lengths, kv_lengths):
        out = _reference(
            q[qs : qs + q_len].transpose(0, 1).unsqueeze(0),
            k[ks : ks + kv_len].transpose(0, 1).unsqueeze(0),
            v[ks : ks + kv_len].transpose(0, 1).unsqueeze(0),
            **kwargs,
        )
        outputs.append(out.squeeze(0).transpose(0, 1))
        qs += q_len
        ks += kv_len
    return torch.cat(outputs)


def _inputs(sq=5, sk=7, heads=2, kv_heads=2, dim=8, device="cpu", seed=41):
    generator = torch.Generator(device=device).manual_seed(seed)
    return {
        name: torch.randn(2, nh, seq, dim, generator=generator, device=device, dtype=torch.float64)
        for name, nh, seq in (
            ("q", heads, sq),
            ("k", kv_heads, sk),
            ("v", kv_heads, sk),
        )
    }


def _assert_traces(test, ref, truth):
    required = {"out", "grad/q", "grad/k", "grad/v"}
    for trace in (truth, ref, test):
        assert set(trace) == required
        for tensor in trace.values():
            assert tensor.numel() > 0
            assert torch.isfinite(tensor).all()
    assert_aligned(test, ref, truth)


@pytest.mark.parametrize("sq,sk", [(5, 5), (2, 5), (5, 2)])
@pytest.mark.parametrize("alignment", [None, "upper_left", "lower_right"])
def test_attention_dense_cpu(sq, sk, alignment):
    inputs = _inputs(sq, sk)
    options = {
        "is_causal": alignment is not None,
        "causal_alignment": alignment,
        "scale": 0.2,
    }
    reference = lambda **kw: _reference(**kw, causal=alignment is not None, alignment=alignment, scale=0.2)
    actual = lambda **kw: ColoAttention.attention(**kw, backend="sdpa", **options)
    truth = run_fwd_bwd(reference, inputs, torch.float64)
    _assert_traces(run_fwd_bwd(actual, inputs, torch.float64), truth, truth)


@pytest.mark.parametrize("alignment,expected", [("upper_left", [1.0, 1.5]), ("lower_right", [2.5, 3.0])])
def test_attention_causal_hand_example(alignment, expected):
    q, k = torch.zeros(1, 1, 2, 4), torch.zeros(1, 1, 5, 4)
    v = torch.arange(1, 6, dtype=torch.float32).reshape(1, 1, 5, 1)
    out = ColoAttention.attention(q, k, v, backend="sdpa", is_causal=True, causal_alignment=alignment)
    torch.testing.assert_close(out.flatten(), torch.tensor(expected))


@pytest.mark.parametrize("floating", [False, True])
def test_attention_custom_mask_gqa_and_gradients(floating):
    inputs = _inputs(heads=4, kv_heads=2)
    mask = torch.ones(5, 7, dtype=torch.bool)
    mask[0] = False
    mask[1, 1:] = False
    mask[:, -1] = False
    if floating:
        mask = torch.zeros(5, 7, dtype=torch.float64).masked_fill(~mask, float("-inf"))
    options = {"causal": True, "alignment": "lower_right", "mask": mask}
    truth = run_fwd_bwd(lambda **kw: _reference(**kw, **options), inputs, torch.float64)
    test = run_fwd_bwd(
        lambda **kw: ColoAttention.attention(
            **kw,
            backend="sdpa",
            attention_mask=mask,
            is_causal=True,
            causal_alignment="lower_right",
        ),
        inputs,
        torch.float64,
    )
    _assert_traces(test, truth, truth)
    assert torch.count_nonzero(test["out"][:, :, 0]) == 0


@pytest.mark.parametrize("alignment", [None, "upper_left", "lower_right"])
def test_attention_varlen_cpu_and_isolation(alignment):
    torch.manual_seed(7)
    inputs = {"q": torch.randn(8, 4, 8, dtype=torch.float64)}
    inputs.update({name: torch.randn(6, 2, 8, dtype=torch.float64) for name in ("k", "v")})
    cuq, cuk = torch.tensor([0, 3, 8], dtype=torch.int32), torch.tensor([0, 2, 6], dtype=torch.int32)

    def actual(**kw):
        return ColoAttention.attention_varlen(
            **kw,
            cu_seqlens_q=cuq,
            cu_seqlens_kv=cuk,
            max_seqlen_q=5,
            max_seqlen_kv=4,
            backend="sdpa",
            is_causal=alignment is not None,
            causal_alignment=alignment,
        )

    truth = run_fwd_bwd(
        lambda **kw: _packed_reference(
            **kw,
            q_lengths=[3, 5],
            kv_lengths=[2, 4],
            causal=alignment is not None,
            alignment=alignment,
        ),
        inputs,
        torch.float64,
    )
    _assert_traces(run_fwd_bwd(actual, inputs, torch.float64), truth, truth)
    changed = {name: x.clone() for name, x in inputs.items()}
    for name in changed:
        changed[name][: 3 if name == "q" else 2] += 10
    torch.testing.assert_close(actual(**changed)[3:], actual(**inputs)[3:], rtol=0, atol=0)


@pytest.mark.parametrize(
    "prefix,maximum,message",
    [
        ([1, 3, 8], 5, "partition"),
        ([0, 3, 7], 5, "partition"),
        ([0, 0, 8], 8, "nonempty"),
        ([0, 9, 8], 9, "nonempty"),
        ([0, 3, 8], 4, "smaller"),
    ],
)
def test_attention_invalid_prefix(prefix, maximum, message):
    q = torch.randn(8, 2, 8)
    with pytest.raises(ValueError, match=message):
        ColoAttention.attention_varlen(
            q,
            q,
            q,
            torch.tensor(prefix, dtype=torch.int32),
            torch.tensor([0, 3, 8], dtype=torch.int32),
            maximum,
            5,
            backend="sdpa",
        )


@pytest.mark.parametrize(
    "options,error",
    [
        ({"backend": "unknown"}, ValueError),
        ({"backend": "fa3"}, NotImplementedError),
        ({"backend": "fa2"}, NotImplementedError),
        ({"backend": "sdpa", "is_causal": True}, ValueError),
        ({"backend": "sdpa", "causal_alignment": "lower_right"}, ValueError),
        ({"backend": "sdpa", "dropout_p": -0.1}, ValueError),
        ({"backend": "sdpa", "scale": float("nan")}, ValueError),
        ({"backend": "sdpa", "attention_mask_type": AttnMaskType.CAUSAL}, ValueError),
        ({"backend": "sdpa", "q_indices": torch.tensor([0])}, ValueError),
        ({"backend": "sdpa", "typo": True}, TypeError),
    ],
)
def test_attention_invalid_requests(options, error):
    with pytest.raises(error):
        ColoAttention.attention(**_inputs(), **options)


def test_attention_auto_cpu_and_legacy_dispatch(monkeypatch):
    inputs = _inputs()
    torch.testing.assert_close(
        ColoAttention.attention(**inputs, backend="auto"),
        ColoAttention.attention(**inputs, backend="sdpa"),
    )
    calls = []

    def legacy(q, k, v, **kwargs):
        calls.append(kwargs)
        return q

    monkeypatch.setattr(ColoAttention, "_dispatch_kernel", lambda *args: legacy)
    out = ColoAttention.attention(
        **inputs,
        attention_mask=torch.zeros(5, 7),
        attention_mask_type=AttnMaskType.CAUSAL,
    )
    assert out is inputs["q"]
    assert calls[0]["is_causal"] is True


def test_attention_strict_capability_and_missing_package(monkeypatch):
    # CPU-only capability tests do not pretend to execute a CUDA kernel.
    q = SimpleNamespace(device=torch.device("cuda"), dtype=torch.bfloat16, shape=(1, 2, 5, 128))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    for kwargs, message in [
        ({"dropout_p": 0.1}, "dropout"),
        ({"attention_mask": torch.ones(5, 5)}, "mask"),
        (
            {"is_causal": True, "causal_alignment": "upper_left", "non_square": True},
            "upper-left",
        ),
    ]:
        options = dict(
            attention_mask=None,
            dropout_p=0.0,
            is_causal=False,
            causal_alignment=None,
            non_square=False,
        )
        options.update(kwargs)
        with pytest.raises(NotImplementedError, match=message):
            ColoAttention._select_backend(q, q, q, "fa3", **options)
    monkeypatch.setattr(FlashAttentionBackendLoader, "is_available", lambda backend: backend == "sdpa")
    assert ColoAttention._select_backend(q, q, q, "auto", None, 0.0, False, None, False) == "sdpa"
    FlashAttentionBackendLoader.load.cache_clear()
    with pytest.raises(ModuleNotFoundError, match="fa3"):
        FlashAttentionBackendLoader.load("fa3")
    wide = SimpleNamespace(device=torch.device("cuda"), dtype=torch.bfloat16, shape=(1, 2, 5, 256))
    with pytest.raises(NotImplementedError, match="at most 192"):
        ColoAttention._select_backend(wide, wide, wide, "fa2", None, 0, False, None, False)


def test_attention_old_fa2_cannot_change_causal_semantics(monkeypatch):
    monkeypatch.setattr(
        kernel_loader,
        "find_spec",
        lambda name: object() if name == "flash_attn" else None,
    )
    monkeypatch.setattr(kernel_loader, "version", lambda name: "2.0.9")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    q = SimpleNamespace(device=torch.device("cuda"), dtype=torch.bfloat16, shape=(1, 2, 5, 128))
    assert not FlashAttentionBackendLoader.is_available("fa2")
    assert ColoAttention._select_backend(q, q, q, "auto", None, 0, True, "lower_right", True) == "sdpa"
    FlashAttentionBackendLoader.load.cache_clear()
    with pytest.raises(NotImplementedError, match=r"flash-attn >= 2.1"):
        FlashAttentionBackendLoader.load("fa2")
    monkeypatch.setattr(kernel_loader, "version", lambda name: "2.1.0")
    assert FlashAttentionBackendLoader.is_available("fa2")


def test_attention_broken_fa3_import_is_not_unavailable(monkeypatch):
    monkeypatch.setattr(kernel_loader, "find_spec", lambda name: object())

    def broken(self):
        raise ImportError("undefined symbol ABI sentinel")

    monkeypatch.setattr(kernel_loader.FlashAttentionDaoV3CudaExtension, "load", broken)
    FlashAttentionBackendLoader.load.cache_clear()
    assert FlashAttentionBackendLoader.is_available("fa3")
    with pytest.raises(ImportError, match="ABI sentinel"):
        FlashAttentionBackendLoader.load("fa3")


@pytest.mark.parametrize("case", ["dtype", "heads", "kv_length", "qk_dim", "empty"])
def test_attention_invalid_tensor_shapes(case):
    inputs = _inputs()
    if case == "dtype":
        inputs["v"] = inputs["v"].float()
    elif case == "heads":
        inputs["q"] = torch.randn(2, 3, 5, 8, dtype=torch.float64)
    elif case == "kv_length":
        inputs["v"] = inputs["v"][:, :, :-1]
    elif case == "qk_dim":
        inputs["q"] = inputs["q"][..., :-1]
    else:
        inputs["q"] = inputs["q"][:, :, :0]
    with pytest.raises(ValueError):
        ColoAttention.attention(**inputs, backend="sdpa")


def test_attention_noncontiguous_last_dimension():
    inputs = {name: x[..., ::2].requires_grad_() for name, x in _inputs(dim=16).items()}
    assert all(x.stride(-1) == 2 for x in inputs.values())
    out = ColoAttention.attention(**inputs, backend="sdpa")
    ref = _reference(**inputs)
    torch.testing.assert_close(out, ref)
    grads = torch.autograd.grad(out.sum(), tuple(inputs.values()), retain_graph=True)
    ref_grads = torch.autograd.grad(ref.sum(), tuple(inputs.values()))
    for actual, expected in zip(grads, ref_grads):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("case", ["dtype", "noncontiguous", "sample_count"])
def test_attention_invalid_prefix_metadata(case):
    q = torch.randn(8, 2, 8)
    cq = torch.tensor([0, 3, 8], dtype=torch.int32)
    ck = cq
    if case == "dtype":
        cq = cq.long()
    elif case == "noncontiguous":
        cq = torch.tensor([0, 0, 3, 0, 8, 0], dtype=torch.int32)[::2]
    else:
        ck = torch.tensor([0, 8], dtype=torch.int32)
    with pytest.raises(ValueError):
        ColoAttention.attention_varlen(q, q, q, cq, ck, 8, 8, backend="sdpa")


@pytest.mark.parametrize("backend", ["fa3", "auto"])
def test_attention_execution_error_is_not_fallback(monkeypatch, backend):
    calls = []

    def broken(*args, **kwargs):
        raise RuntimeError("native backend failure sentinel")

    def load(backend, packed=False):
        calls.append(backend)
        return broken

    monkeypatch.setattr(ColoAttention, "_select_backend", lambda *args: "fa3")
    monkeypatch.setattr(FlashAttentionBackendLoader, "load", load)
    with pytest.raises(RuntimeError, match="sentinel"):
        ColoAttention.attention(**_inputs(), backend=backend)
    assert calls == ["fa3"]


def test_attention_compile_dense_forward_backward():
    fn = lambda q, k, v: ColoAttention.attention(
        q, k, v, backend="sdpa", is_causal=True, causal_alignment="lower_right"
    )
    inputs = _inputs()
    fn(**inputs)  # Resolve the optional backend before entering the compiled graph.
    compiled = torch.compile(fn, backend="aot_eager", fullgraph=True)
    reference = run_fwd_bwd(fn, inputs, torch.float64)
    _assert_traces(run_fwd_bwd(compiled, inputs, torch.float64), reference, reference)


def test_attention_numerics_detect_wrong_scale():
    inputs = _inputs()
    truth = run_fwd_bwd(_reference, inputs, torch.float64)
    wrong = run_fwd_bwd(
        lambda **kw: ColoAttention.attention(**kw, backend="sdpa", scale=0),
        inputs,
        torch.float64,
    )
    with pytest.raises(AssertionError):
        _assert_traces(wrong, truth, truth)


def _forbid_sdpa_fallback(*args, **kwargs):
    # Dynamo inspects torch function metadata; keep this a real function, not a Mock.
    raise AssertionError("SDPA fallback")


def _require_gpu(backend):
    required = os.environ.get("COLOSSALAI_TEST_ATTENTION_BACKEND")
    if required and required not in ("fa2", "fa3", "sdpa", "all"):
        pytest.fail("COLOSSALAI_TEST_ATTENTION_BACKEND must be fa2, fa3, sdpa or all")
    if required not in (None, "all", backend):
        pytest.skip(f"This acceptance run selects {required}")
    if not torch.cuda.is_available():
        if required:
            pytest.fail("Strict GPU attention acceptance requires a CUDA GPU")
        pytest.skip("CUDA unavailable; this is not GPU validation")
    # Ordinary regression runs may lack optional packages. Explicit acceptance
    # must fail instead; installed-but-broken packages always surface their error.
    if backend == "sdpa":
        return
    package = "flash_attn_3" if backend == "fa3" else "flash_attn"
    if required is None and find_spec(package) is None:
        pytest.skip(f"Optional {package} absent; this is not {backend} validation")


def _watch_real_backend(backend, packed, monkeypatch):
    if backend == "sdpa":
        module = torch.nn.functional
        others = (kernel_loader.FlashAttentionDaoCudaExtension, kernel_loader.FlashAttentionDaoV3CudaExtension)
        function_name = "scaled_dot_product_attention"
    elif backend == "fa3":
        from flash_attn_3 import flash_attn_interface as module

        others = (kernel_loader.FlashAttentionDaoCudaExtension,)
        function_name = "flash_attn_varlen_func" if packed else "flash_attn_func"
    else:
        import flash_attn as module

        others = (kernel_loader.FlashAttentionDaoV3CudaExtension,)
        function_name = "flash_attn_varlen_func" if packed else "flash_attn_func"
    original = getattr(module, function_name)
    calls = []

    def entered(*args, **kwargs):
        calls.append(function_name)
        return original(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Fallback to the other FlashAttention backend")

    for other in others:
        monkeypatch.setattr(other, "load", forbidden)
        monkeypatch.setattr(other, "load_varlen", forbidden)
    monkeypatch.setattr(module, function_name, entered)
    FlashAttentionBackendLoader.load.cache_clear()
    return module, function_name, calls


def test_attention_required_device_does_not_skip(monkeypatch):
    monkeypatch.setenv("COLOSSALAI_TEST_ATTENTION_BACKEND", "fa3")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(pytest.fail.Exception, match="requires a CUDA GPU"):
        _require_gpu("fa3")


@pytest.mark.parametrize("backend", ["fa2", "fa3", "sdpa"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [32, 64, 128])
@pytest.mark.parametrize("heads", [2, 4])
@pytest.mark.parametrize("seed", [0, 41])
@pytest.mark.parametrize(
    "packed,causal,sq,sk",
    [
        (False, False, 127, 129),
        (False, True, 129, 129),
        (False, True, 2, 5),
        (False, True, 5, 2),
        (True, False, 127, 129),
        (True, True, 5, 2),
        (False, False, 1, 1),
        (True, False, 1, 1),
    ],
)
def test_attention_gpu_numerics_no_fallback(backend, dtype, dim, heads, seed, packed, causal, sq, sk, monkeypatch):
    _require_gpu(backend)
    alignment = "lower_right" if causal else None
    if packed:
        torch.manual_seed(seed)
        q_lengths, kv_lengths = [3, sq], [4, sk]
        inputs = {
            "q": torch.randn(sum(q_lengths), heads, dim, device="cuda", dtype=torch.float64),
            "k": torch.randn(sum(kv_lengths), 2, dim, device="cuda", dtype=torch.float64),
            "v": torch.randn(sum(kv_lengths), 2, dim, device="cuda", dtype=torch.float64),
        }
        cuq = torch.tensor([0, 3, sum(q_lengths)], dtype=torch.int32, device="cuda")
        cuk = torch.tensor([0, 4, sum(kv_lengths)], dtype=torch.int32, device="cuda")
        reference = lambda **kw: _packed_reference(
            **kw,
            q_lengths=q_lengths,
            kv_lengths=kv_lengths,
            causal=causal,
            alignment=alignment,
        )
        actual = lambda **kw: ColoAttention.attention_varlen(
            **kw,
            cu_seqlens_q=cuq,
            cu_seqlens_kv=cuk,
            max_seqlen_q=max(q_lengths),
            max_seqlen_kv=max(kv_lengths),
            backend=backend,
            is_causal=causal,
            causal_alignment=alignment,
        )
    else:
        inputs = _inputs(sq, sk, heads=heads, kv_heads=2, dim=dim, device="cuda", seed=seed)
        reference = lambda **kw: _reference(**kw, causal=causal, alignment=alignment)
        actual = lambda **kw: ColoAttention.attention(
            **kw, backend=backend, is_causal=causal, causal_alignment=alignment
        )
    _check_backend_numerics(actual, reference, inputs, dtype, backend, packed, monkeypatch)


def _check_backend_numerics(actual, reference, inputs, dtype, backend, packed, monkeypatch):
    module, function_name, calls = _watch_real_backend(backend, packed, monkeypatch)
    original_load = FlashAttentionBackendLoader.load
    capture = independent_impl.run_fwd_bwd
    traces = []

    def record_trace(*args, **kwargs):
        trace = capture(*args, **kwargs)
        traces.append((kwargs["role"], trace))
        return trace

    def strict_load(selected, packed=False):
        assert selected == backend, f"Unexpected fallback to {selected}"
        return original_load(selected, packed)

    monkeypatch.setattr(FlashAttentionBackendLoader, "load", strict_load)
    with (
        patch.object(independent_impl, "run_fwd_bwd", new=record_trace),
        (
            nullcontext()
            if backend == "sdpa"
            else patch("torch.nn.functional.scaled_dot_product_attention", new=_forbid_sdpa_fallback)
        ),
    ):
        report = independent_impl.check_fwd_bwd(
            actual, reference, inputs, dtype, expect_calls=[(module, function_name)]
        )
    assert calls and set(calls) == {function_name}
    assert [role for role, _ in traces] == ["truth", "ref", "test"]
    call_checks = [
        {"name": entry.name, "passed": entry.passed, "reason": entry.reason}
        for entry in report.entries
        if entry.name.startswith("<called>/")
    ]
    for _, trace in traces:
        trace.meta["test_call_checks"] = call_checks
    _assert_traces(traces[2][1], traces[1][1], traces[0][1])
    report.assert_pass()
    original_load.cache_clear()


@pytest.mark.parametrize("backend", ["fa2", "fa3", "sdpa"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length", [512, 2048])
def test_attention_gpu_long_sequence(backend, dtype, length, monkeypatch):
    _require_gpu(backend)
    inputs = _inputs(length, length, heads=4, kv_heads=2, dim=128, device="cuda")
    actual = lambda **kw: ColoAttention.attention(**kw, backend=backend)
    _check_backend_numerics(actual, _reference, inputs, dtype, backend, False, monkeypatch)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("floating", [False, True])
@pytest.mark.parametrize("alignment", ["upper_left", "lower_right"])
@pytest.mark.parametrize("seed", [0, 41])
def test_attention_gpu_sdpa_mask(dtype, floating, alignment, seed, monkeypatch):
    _require_gpu("sdpa")
    inputs = _inputs(5, 7, heads=4, kv_heads=2, dim=64, device="cuda", seed=seed)
    mask = torch.ones(5, 7, device="cuda", dtype=torch.bool)
    mask[0] = False
    mask[1, 2:] = False
    mask[:, -1] = False
    if floating:
        mask = torch.full((5, 7), 0.25, device="cuda", dtype=torch.float32).masked_fill(~mask, float("-inf"))
    reference = lambda **kw: _reference(**kw, mask=mask, causal=True, alignment=alignment)
    actual = lambda **kw: ColoAttention.attention(
        **kw, attention_mask=mask, backend="sdpa", is_causal=True, causal_alignment=alignment
    )
    _check_backend_numerics(actual, reference, inputs, dtype, "sdpa", False, monkeypatch)


@pytest.mark.parametrize("backend", ["fa2", "fa3", "sdpa"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seed", [0, 41])
def test_attention_gpu_varlen_isolation(backend, dtype, seed, monkeypatch):
    _require_gpu(backend)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    inputs = {
        name: torch.randn(tokens, heads, 64, device="cuda", dtype=torch.float64, generator=generator)
        for name, tokens, heads in (("q", 8, 4), ("k", 6, 2), ("v", 6, 2))
    }
    cuq = torch.tensor([0, 3, 8], device="cuda", dtype=torch.int32)
    cuk = torch.tensor([0, 2, 6], device="cuda", dtype=torch.int32)

    def actual(q, k, v):
        return ColoAttention.attention_varlen(q, k, v, cuq, cuk, 5, 4, backend=backend)[3:]

    def reference(q, k, v):
        # Only the second sample contributes to the loss. The first sample's
        # input gradients must be exactly zero, not just numerically close.
        return _packed_reference(q, k, v, [3, 5], [2, 4])[3:]

    changed = {name: value.clone() for name, value in inputs.items()}
    for name in changed:
        changed[name][: 3 if name == "q" else 2] += 3
    original = run_fwd_bwd(actual, inputs, dtype)
    perturbed = run_fwd_bwd(actual, changed, dtype)
    torch.testing.assert_close(original["out"], perturbed["out"], rtol=0, atol=0)
    for trace in (original, perturbed):
        for name in inputs:
            assert torch.count_nonzero(trace[f"grad/{name}"][: 3 if name == "q" else 2]) == 0
    # Compare both executions to the independent reference rather than allowing
    # an arbitrary epsilon between two potentially nondeterministic backwards.
    for data in (inputs, changed):
        with monkeypatch.context() as scoped:
            _check_backend_numerics(actual, reference, data, dtype, backend, True, scoped)


def test_attention_numerics_detect_wrong_causal_alignment():
    inputs = _inputs(2, 5)
    reference = lambda **kw: _reference(**kw, causal=True, alignment="lower_right")
    wrong = lambda **kw: _reference(**kw, causal=True, alignment="upper_left")
    with pytest.raises(AssertionError, match="numerical alignment"):
        independent_impl.check_fwd_bwd(wrong, reference, inputs, torch.float64).assert_pass()


def test_attention_numerics_require_backend_call():
    backend = SimpleNamespace(attention=_reference)
    report = independent_impl.check_fwd_bwd(
        _reference, _reference, _inputs(), torch.float64, expect_calls=[(backend, "attention")]
    )
    assert len(report.failures) == 1
    assert report.failures[0].reason == "never called"
    with pytest.raises(AssertionError, match="never called"):
        report.assert_pass()


@pytest.mark.parametrize("backend", ["fa2", "fa3"])
@pytest.mark.parametrize("packed", [False, True])
def test_attention_gpu_actual_kernel_and_error(backend, packed, monkeypatch, tmp_path, record_property):
    _require_gpu(backend)
    module, function_name, calls = _watch_real_backend(backend, packed, monkeypatch)

    torch.manual_seed(41)
    inputs = {name: torch.randn(8, 2, 128, device="cuda", dtype=torch.float64) for name in ("q", "k", "v")}

    def actual(q, k, v):
        # Exercise last-dimension copies while retaining gradients to Q/K/V.
        inputs = [torch.stack((x, x), dim=-1).flatten(-2)[..., ::2] for x in (q, k, v)]
        assert all(x.stride(-1) == 2 for x in inputs)
        if packed:
            cu = torch.tensor([0, 3, 8], device="cuda", dtype=torch.int32)
            return ColoAttention.attention_varlen(*inputs, cu, cu, 5, 5, backend=backend)
        return ColoAttention.attention(*(x.transpose(0, 1).unsqueeze(0) for x in inputs), backend=backend)

    def reference(q, k, v):
        if packed:
            return _packed_reference(q, k, v, q_lengths=[3, 5], kv_lengths=[3, 5])
        return _reference(*(x.transpose(0, 1).unsqueeze(0) for x in (q, k, v)))

    truth = run_fwd_bwd(reference, inputs, torch.float64)
    ref = run_fwd_bwd(reference, inputs, torch.bfloat16)
    with patch(
        "torch.nn.functional.scaled_dot_product_attention",
        new=_forbid_sdpa_fallback,
    ):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            test = run_fwd_bwd(actual, inputs, torch.bfloat16)
            torch.cuda.synchronize()
    assert calls == [function_name]
    _assert_traces(test, ref, truth)
    names = [event.name for event in prof.events()]
    if backend == "fa3":
        assert any("flash_attn_3::_flash_attn_forward" in name for name in names), names
        assert any("flash_attn_3::_flash_attn_backward" in name for name in names), names
    else:
        assert any("FlashAttn" in name and "Backward" in name for name in names), names
    assert any(event.device_type == torch.autograd.DeviceType.CUDA for event in prof.events())
    trace_dir = Path(os.environ.get("COLOSSALAI_ATTENTION_TRACE_DIR", tmp_path))
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"{backend}-{'packed' if packed else 'dense'}.json"
    prof.export_chrome_trace(str(trace_path))
    for role, trace in (("truth", truth), ("ref", ref), ("test", test)):
        trace.save(trace_dir / f"{trace_path.stem}-{role}.safetensors")
    record_property("backend_module", module.__file__)
    installed = distribution("flash_attn_3" if backend == "fa3" else "flash-attn")
    record_property("backend_version", installed.version)
    direct_url = installed.read_text("direct_url.json")
    record_property("backend_direct_url", direct_url or "unavailable")
    provenance = json.loads(direct_url) if direct_url else {}
    record_property("installed_vcs_commit", provenance.get("vcs_info", {}).get("commit_id", "unavailable"))
    # The expected source revision in the adapter docstring is not proof of the
    # installed build's origin. Retain a wheel/hash when the build job provides it.
    if backend == "fa3" and os.environ.get("COLOSSALAI_FA3_WHEEL_PATH"):
        wheel_path = Path(os.environ["COLOSSALAI_FA3_WHEEL_PATH"])
        wheel_hash = hashlib.sha256()
        with wheel_path.open("rb") as wheel_file:
            for chunk in iter(lambda: wheel_file.read(1024 * 1024), b""):
                wheel_hash.update(chunk)
        record_property("wheel_path", str(wheel_path))
        record_property("wheel_sha256", wheel_hash.hexdigest())
    record_property("torch_cuda", f"{torch.__version__}/{torch.version.cuda}")
    record_property("gpu", torch.cuda.get_device_name())
    record_property("profiler", str(trace_path))

    def broken(*args, **kwargs):
        raise RuntimeError("GPU backend failure sentinel")

    monkeypatch.setattr(module, function_name, broken)
    FlashAttentionBackendLoader.load.cache_clear()
    with pytest.raises(RuntimeError, match="GPU backend failure sentinel"):
        actual(**{name: x.bfloat16() for name, x in inputs.items()})
    FlashAttentionBackendLoader.load.cache_clear()


@pytest.mark.parametrize("backend", ["fa2", "fa3"])
def test_attention_gpu_compile_forward_backward(backend):
    _require_gpu(backend)
    inputs = _inputs(17, 17, dim=128, device="cuda")
    fn = lambda q, k, v: ColoAttention.attention(q, k, v, backend=backend)
    # Resolve imports before compilation. Keep real CUDA execution and backward
    # in both runs; a fullgraph compiler failure must remain an acceptance failure.
    reference = run_fwd_bwd(fn, inputs, torch.bfloat16)
    compiled = torch.compile(fn, fullgraph=True)
    with patch("torch.nn.functional.scaled_dot_product_attention", new=_forbid_sdpa_fallback):
        test = run_fwd_bwd(compiled, inputs, torch.bfloat16)
    truth = run_fwd_bwd(_reference, inputs, torch.float64)
    ref = run_fwd_bwd(_reference, inputs, torch.bfloat16)
    _assert_traces(reference, ref, truth)
    _assert_traces(test, ref, truth)
