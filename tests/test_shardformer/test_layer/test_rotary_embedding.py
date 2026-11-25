import itertools
import math
import os
from unittest.mock import patch

import pytest
import torch
from torch.testing import assert_close

from colossalai.shardformer.layer import apply_rotary, make_rotary_3d, rotary_embedding
from tests.kit.numerics import assert_aligned, judge
from tests.kit.numerics.scenarios import independent_impl

MODEL_CONVENTIONS = [((44, 42, 42), "adjacent"), ((16, 56, 56), "half"), ((16, 56, 56), "adjacent")]


def _grid(t=2, h=2, w=3):
    # Explicit Cartesian product keeps width fastest, independent of production code.
    return torch.tensor(list(itertools.product(range(t), range(h), range(w))))


def _complex_reference(x, positions, axes_dims, pairing):
    """Independent complex rotation, with FP64 truth or FP32 low-precision arithmetic."""
    compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    flat = x.permute(0, 2, 1, 3).reshape(-1, x.shape[1], x.shape[-1]) if x.ndim == 4 else x
    flat = flat.to(compute_dtype)
    positions = positions.reshape(-1, 3).to(compute_dtype)
    output = [None] * x.shape[-1]
    pair_index = 0
    for axis, dim in enumerate(axes_dims):
        for frequency in range(dim // 2):
            angle = positions[:, axis, None] * math.pow(10000, -2 * frequency / dim)
            phase = torch.polar(torch.ones_like(angle), angle)
            if pairing == "adjacent":
                first, second = 2 * pair_index, 2 * pair_index + 1
            else:
                first, second = pair_index, pair_index + x.shape[-1] // 2
            rotated = torch.complex(flat[..., first], flat[..., second]) * phase
            output[first], output[second] = rotated.real, rotated.imag
            pair_index += 1
    result = torch.stack(output, dim=-1).to(x.dtype)
    if x.ndim == 4:
        result = result.reshape(x.shape[0], x.shape[2], x.shape[1], x.shape[3]).permute(0, 2, 1, 3)
    return result


def _assert_traces(test, ref, truth):
    expected = {"out/q", "out/k", "grad/q", "grad/k"}
    for trace in (truth, ref, test):
        assert set(trace) == expected
        for value in trace.values():
            assert value.numel() and torch.isfinite(value).all()
    assert_aligned(test, ref, truth)


def _check_alignment(device, dtype, packed, axes_dims, pairing):
    torch.manual_seed(7)
    if packed:
        # A has 12 tokens; B has 3 and restarts its own coordinates.
        positions = torch.cat((_grid(), _grid(t=1, h=1, w=3)), dim=0).to(device)
        shape = (15, 2, 128)
    else:
        positions = torch.stack((_grid(), _grid() + torch.tensor([1, 2, 3]))).to(device)
        shape = (2, 2, 12, 128)
    inputs = {name: torch.randn(shape, dtype=torch.float64, device=device) for name in ("q", "k")}

    def reference(q, k):
        return {name: _complex_reference(x, positions, axes_dims, pairing) for name, x in (("q", q), ("k", k))}

    def candidate(q, k):
        cos, sin = rotary_embedding.make_rotary_3d(positions, axes_dims)
        return {
            "q": rotary_embedding.apply_rotary(q, cos, sin, pairing),
            "k": rotary_embedding.apply_rotary(k, cos, sin, pairing),
        }

    capture = independent_impl.run_fwd_bwd
    traces = []

    def record_trace(*args, **kwargs):
        trace = capture(*args, **kwargs)
        traces.append((kwargs["role"], trace))
        return trace

    with patch.object(independent_impl, "run_fwd_bwd", new=record_trace):
        report = independent_impl.check_fwd_bwd(
            candidate,
            reference,
            inputs,
            dtype,
            expect_calls=[(rotary_embedding, "make_rotary_3d"), (rotary_embedding, "apply_rotary")],
        )
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


@pytest.mark.parametrize("axes_dims,pairing", MODEL_CONVENTIONS)
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rotary_output_and_gradients(axes_dims, pairing, packed, dtype):
    _check_alignment("cpu", dtype, packed, axes_dims, pairing)


@pytest.mark.parametrize("axes_dims,pairing", MODEL_CONVENTIONS)
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rotary_cuda_output_and_gradients(axes_dims, pairing, packed, dtype):
    if not torch.cuda.is_available():
        required = os.environ.get("COLOSSALAI_TEST_ROTARY_CUDA") == "1"
        required |= os.environ.get("COLOSSALAI_TEST_ATTENTION_BACKEND") == "fa3"
        if required:
            pytest.fail("CUDA is required for the requested RoPE GPU acceptance tests")
        pytest.skip("RoPE CUDA tests require a GPU; set COLOSSALAI_TEST_ROTARY_CUDA=1 to require execution")
    _check_alignment("cuda", dtype, packed, axes_dims, pairing)


@pytest.mark.parametrize("pairing", ["adjacent", "half"])
def test_rotary_zero_position_and_no_mutation(pairing):
    torch.manual_seed(8)
    q, k, v = (torch.randn(2, 3, 4, 8, requires_grad=True) for _ in range(3))
    originals = [x.detach().clone() for x in (q, k, v)]
    cos, sin = make_rotary_3d(torch.zeros(2, 4, 3, dtype=torch.long), (4, 2, 2))
    q_out, k_out = apply_rotary(q, cos, sin, pairing), apply_rotary(k, cos, sin, pairing)
    assert_close(q_out, q, atol=0, rtol=0)
    assert_close(k_out, k, atol=0, rtol=0)
    assert q_out.data_ptr() != q.data_ptr() and k_out.data_ptr() != k.data_ptr()
    (q_out.sum() + k_out.sum()).backward()
    assert_close(q.grad, torch.ones_like(q))
    assert_close(k.grad, torch.ones_like(k))
    assert v.grad is None
    for tensor, original in zip((q, k, v), originals):
        assert_close(tensor, original, atol=0, rtol=0)


@pytest.mark.parametrize("pairing", ["adjacent", "half"])
@pytest.mark.parametrize("axis", range(3))
def test_rotary_changes_only_selected_axis_pairs(pairing, axis):
    axes_dims = (4, 2, 6)
    positions = torch.eye(3, dtype=torch.long)[axis].reshape(1, 3)
    cos, sin = make_rotary_3d(positions, axes_dims, compute_dtype=torch.float64)
    x = torch.arange(1, 13, dtype=torch.float64).reshape(1, 1, 12)
    result = apply_rotary(x, cos, sin, pairing)
    pair_start, pair_stop = sum(axes_dims[:axis]) // 2, sum(axes_dims[: axis + 1]) // 2
    changed = []
    for pair in range(pair_start, pair_stop):
        changed.extend((2 * pair, 2 * pair + 1) if pairing == "adjacent" else (pair, pair + 6))
    unchanged = [i for i in range(12) if i not in changed]
    assert_close(result[..., unchanged], x[..., unchanged], atol=0, rtol=0)
    assert not torch.allclose(result[..., changed], x[..., changed])
    assert_close(result, _complex_reference(x, positions, axes_dims, pairing), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("pairing", ["adjacent", "half"])
def test_rotary_preserves_norm_and_same_position_dot_product(pairing):
    torch.manual_seed(9)
    positions = _grid().unsqueeze(0)
    q, k = (torch.randn(2, 3, 12, 12, dtype=torch.float64) for _ in range(2))
    cos, sin = make_rotary_3d(positions, (4, 2, 6), compute_dtype=torch.float64)
    q_rotated, k_rotated = (apply_rotary(x, cos, sin, pairing) for x in (q, k))
    assert_close(q_rotated.square().sum(-1), q.square().sum(-1), atol=1e-12, rtol=1e-12)
    assert_close((q_rotated * k_rotated).sum(-1), (q * k).sum(-1), atol=1e-12, rtol=1e-12)
    assert not torch.allclose(q_rotated @ k_rotated.transpose(-2, -1), q @ k.transpose(-2, -1))


@pytest.mark.parametrize("pairing", ["adjacent", "half"])
def test_rotary_packed_positions_reset_and_strided_inputs(pairing):
    torch.manual_seed(10)
    positions = torch.cat((_grid(), _grid(t=1, h=1, w=3)))
    # Last-dimension stride 2 exercises inputs that do not have contiguous feature pairs.
    x = torch.randn(15, 2, 24, dtype=torch.float64)[..., ::2]
    assert not x.is_contiguous()
    cos, sin = make_rotary_3d(positions, (4, 2, 6), compute_dtype=torch.float64)
    packed = apply_rotary(x, cos, sin, pairing)
    expected = _complex_reference(x, positions, (4, 2, 6), pairing)
    assert_close(packed, expected, atol=1e-12, rtol=1e-12)
    assert_close(packed[12], x[12], atol=0, rtol=0)  # first token of B has position (0, 0, 0)
    pieces = []
    for start, stop in ((0, 12), (12, 15)):
        part_cos, part_sin = make_rotary_3d(positions[start:stop].unsqueeze(0), (4, 2, 6), compute_dtype=torch.float64)
        dense = x[start:stop].transpose(0, 1).unsqueeze(0)
        pieces.append(apply_rotary(dense, part_cos, part_sin, pairing).squeeze(0).transpose(0, 1))
    assert_close(packed, torch.cat(pieces), atol=0, rtol=0)


def test_rotary_pairing_and_axis_negative_controls():
    positions = torch.tensor([[1, 2, 3], [2, 1, 4]])
    x = torch.arange(1, 25, dtype=torch.float64).reshape(2, 1, 12)
    cos, sin = make_rotary_3d(positions, (4, 2, 6), compute_dtype=torch.float64)
    adjacent = apply_rotary(x, cos, sin, "adjacent")
    wrong_pairing = apply_rotary(x, cos, sin, "half")
    wrong_cos, wrong_sin = make_rotary_3d(positions[:, [2, 1, 0]], (4, 2, 6), compute_dtype=torch.float64)
    wrong_axes = apply_rotary(x, wrong_cos, wrong_sin)
    truth = _complex_reference(x, positions, (4, 2, 6), "adjacent")
    assert_aligned(adjacent, truth, truth)
    assert not judge(wrong_pairing, truth, truth).passed
    assert not judge(wrong_axes, truth, truth).passed
    # A whole-head feature permutation converts conventions, but omitting it changes the function.
    permutation = torch.tensor(list(range(0, 12, 2)) + list(range(1, 12, 2)))
    converted = apply_rotary(x[..., permutation], cos, sin, "half")[..., permutation.argsort()]
    assert_close(converted, adjacent, atol=0, rtol=0)


@pytest.mark.parametrize("pairing", ["adjacent", "half"])
@pytest.mark.parametrize("packed", [False, True])
def test_rotary_gradcheck(pairing, packed):
    torch.manual_seed(11)
    positions = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    x = torch.randn(2, 1, 6, dtype=torch.float64, requires_grad=True)
    if not packed:
        positions = positions.unsqueeze(0)
        x = x.transpose(0, 1).unsqueeze(0)

    def rotate(value, coordinates):
        cos, sin = make_rotary_3d(coordinates, (2, 2, 2), compute_dtype=torch.float64)
        return apply_rotary(value, cos, sin, pairing)

    assert torch.autograd.gradcheck(rotate, (x, positions))


@pytest.mark.parametrize("pairing", ["adjacent", "half"])
def test_rotary_compile_forward_and_backward(pairing):
    torch.manual_seed(13)
    positions = _grid().unsqueeze(0)

    def rotate(x):
        cos, sin = make_rotary_3d(positions, (4, 2, 6))
        return apply_rotary(x, cos, sin, pairing)

    compiled = torch.compile(rotate, backend="aot_eager", fullgraph=True)
    x = torch.randn(1, 2, 12, 12, requires_grad=True)
    compiled_x = x.detach().clone().requires_grad_(True)
    expected, actual = rotate(x), compiled(compiled_x)
    assert_close(actual, expected)
    upstream = torch.randn_like(x)
    actual.backward(upstream)
    expected.backward(upstream)
    assert_close(compiled_x.grad, x.grad)


def test_rotary_keeps_integer_position_precision():
    positions = torch.tensor([[256, 0, 0], [257, 0, 0]])
    cos, sin = make_rotary_3d(positions, (2, 2, 2))
    assert cos.dtype == sin.dtype == torch.float32
    assert not torch.allclose(cos[0], cos[1])
    assert_close(cos[:, 0], positions[:, 0].float().cos(), atol=0, rtol=0)
    assert_close(sin[:, 0], positions[:, 0].float().sin(), atol=0, rtol=0)


def test_rotary_matches_diffusers_wan_reference():
    from diffusers.models.embeddings import apply_rotary_emb
    from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed

    # Diffusers 0.35.2: real model frequency generation on a 2 x 2 x 3 patch grid.
    native = WanRotaryPosEmbed(128, (1, 2, 2), 32)
    native_cos, native_sin = native(torch.zeros(1, 4, 2, 4, 6))
    positions = _grid().unsqueeze(0)
    cos, sin = make_rotary_3d(positions, (44, 42, 42))
    assert_close(cos, native_cos.squeeze(2)[..., ::2], atol=1e-6, rtol=1e-6)
    assert_close(sin, native_sin.squeeze(2)[..., ::2], atol=1e-6, rtol=1e-6)
    torch.manual_seed(12)
    q = torch.randn(1, 2, 12, 128, requires_grad=True)
    reference_q = q.detach().clone().requires_grad_(True)
    expected = apply_rotary_emb(reference_q, (native_cos[0, :, 0], native_sin[0, :, 0]))
    actual = apply_rotary(q, cos, sin, "adjacent")
    assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    upstream = torch.randn_like(q)
    actual.backward(upstream)
    expected.backward(upstream)
    assert_close(q.grad, reference_q.grad, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("axes_dims", [(1, 2, 2), (0, 2, 2), (2, 2), (2, 2, 2, 2), (2.0, 2, 2), (True, 2, 2)])
def test_rotary_rejects_invalid_axes(axes_dims):
    with pytest.raises(ValueError, match="three positive even integers"):
        make_rotary_3d(torch.zeros(1, 3), axes_dims)


@pytest.mark.parametrize("theta", [0, -1, math.inf, math.nan, True, "10000"])
def test_rotary_rejects_invalid_theta(theta):
    with pytest.raises(ValueError, match="positive finite"):
        make_rotary_3d(torch.zeros(1, 3), (2, 2, 2), theta=theta)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.bool, torch.complex64])
def test_rotary_rejects_imprecise_position_dtype(dtype):
    with pytest.raises(TypeError, match="position_ids"):
        make_rotary_3d(torch.zeros(1, 3, dtype=dtype), (2, 2, 2))


@pytest.mark.parametrize("shape", [(3,), (2, 4), (1, 2, 3, 3)])
def test_rotary_rejects_invalid_position_shape(shape):
    with pytest.raises(ValueError, match="position_ids"):
        make_rotary_3d(torch.zeros(shape), (2, 2, 2))


def test_rotary_rejects_invalid_application_contract():
    x = torch.zeros(2, 3, 4, 8)
    cos, sin = make_rotary_3d(torch.zeros(2, 4, 3), (4, 2, 2))
    with pytest.raises(TypeError, match="compute_dtype"):
        make_rotary_3d(torch.zeros(2, 4, 3), (4, 2, 2), compute_dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="pairing"):
        apply_rotary(x, cos, sin, pairing="unknown")
    with pytest.raises(ValueError, match="positive even D"):
        apply_rotary(x[..., :7], cos, sin)
    with pytest.raises(ValueError, match="dense"):
        apply_rotary(x, cos[:, :1], sin[:, :1])
    with pytest.raises(ValueError, match="dense"):
        apply_rotary(x, cos[..., :3], sin[..., :3])
    with pytest.raises(ValueError, match="packed"):
        apply_rotary(x[0], cos, sin)
    with pytest.raises(ValueError, match="same shape"):
        apply_rotary(x, cos, sin[:1])
    with pytest.raises(TypeError, match="same float32 or float64"):
        apply_rotary(x, cos.half(), sin.half())
    with pytest.raises(TypeError, match="x must have dtype"):
        apply_rotary(x.long(), cos, sin)
    with pytest.raises(ValueError, match="same device"):
        apply_rotary(x, cos.to("meta"), sin.to("meta"))
