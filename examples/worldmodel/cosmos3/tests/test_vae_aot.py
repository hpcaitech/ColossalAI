"""CPU tests for the AOT runner: gate/dispatch logic with fake compiled functions, plus one real ``torch.export``
graph. Inductor compilation and real packages still need CUDA; no latent claim is made here."""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vae_aot import AOTChunkRunner, _cache_signature, _eager_chunk, _variant_key


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.register_buffer("offset", torch.zeros(1))

    def forward(self, x, feat_cache, feat_idx):
        out = x * self.weight + self.offset
        feat_cache[0] = out[:, :, -1:].clone()
        return out


def example():
    x = torch.arange(16.0).reshape(1, 2, 2, 2, 2)
    cache = [torch.ones(1, 2, 1, 2, 2), None]
    return x, cache


def test_variant_covers_full_metadata():
    x, cache = example()
    key = _variant_key(x, cache)
    assert key != _variant_key(x.half(), cache)
    assert key != _variant_key(x.transpose(-1, -2), cache)
    assert key != _variant_key(x[:, :1], cache)
    assert key != _variant_key(x, [cache[0].bfloat16(), None])
    assert key != _variant_key(x, [cache[0].transpose(-1, -2), None])
    assert key != _variant_key(x, [torch.ones(1, 3, 1, 2, 2), None])
    assert key != _variant_key(x, [None, cache[0]])
    assert key != _variant_key(x.to("meta"), cache)
    assert key != _variant_key(x, [cache[0].to("meta"), None])
    assert _cache_signature(cache) != _cache_signature(cache + [None])


def test_identity_tracks_weights_buffers_patches_and_versions(tmp_path):
    encoder = Encoder().eval()
    runner = AOTChunkRunner(encoder, tmp_path)
    key = runner._encoder_identity()
    with torch.no_grad():
        encoder.weight.add_(1)
    assert key != runner._encoder_identity()
    key = runner._encoder_identity()
    encoder.offset.add_(1)
    assert key != runner._encoder_identity()
    key = runner._encoder_identity()
    encoder.forward = lambda *args, **kwargs: None
    assert key != runner._encoder_identity()
    key = runner._encoder_identity()
    runner.version_identity = "checkpoint-v2"
    assert key != runner._encoder_identity()
    # A version check does not copy/hash weights or state_dict on each call.
    encoder.state_dict = lambda: pytest.fail("hot-path state_dict copy")
    runner._encoder_identity()


@pytest.mark.parametrize("failure", ["none", "exception"])
def test_failed_variant_compiled_once_then_invalidated(tmp_path, failure):
    runner = AOTChunkRunner(Encoder().eval(), tmp_path)
    calls = []

    def fail(*args):
        calls.append(1)
        if failure == "exception":
            raise RuntimeError("compiler failed")
        return None

    runner._compile = fail
    x, _ = example()
    with torch.no_grad():
        for _ in range(3):
            torch.testing.assert_close(runner._dispatch(x, [None, None]), x)
        assert len(calls) == 1
        runner.invalidate()
        runner._dispatch(x, [None, None])
        assert len(calls) == 2
        runner.encoder.weight.add_(1)
        torch.testing.assert_close(runner._dispatch(x, [None, None]), 2 * x)
        assert len(calls) == 3


def test_non_cuda_rejected_before_compile_or_files(tmp_path):
    path = tmp_path / "not-created"
    runner = AOTChunkRunner(Encoder().eval(), path)
    runner._compile = lambda *args: pytest.fail("must reject before compiling")
    with pytest.raises(ValueError, match="requires CUDA"):
        runner(torch.ones(1), [None])
    assert not path.exists()


@pytest.mark.parametrize("bad", ["nan", "inf", "extra", "missing", "shape", "dtype", "latent", "cache"])
def test_gate_rejects_nonfinite_structure_and_wrong_cache(tmp_path, bad):
    runner = AOTChunkRunner(Encoder().eval(), tmp_path)
    x, cache = example()
    expected, next_cache = _eager_chunk(runner.encoder, x, cache)

    def function(*args):
        out, c = expected.clone(), next_cache[0].clone()
        if bad == "nan":
            c.fill_(float("nan"))
        if bad == "inf":
            out.fill_(float("inf"))
        if bad == "extra":
            return out, c, c
        if bad == "missing":
            return (out,)
        if bad == "shape":
            return out.flatten(), c
        if bad == "dtype":
            return out.double(), c
        if bad == "latent":
            out.add_(0.2)
        if bad == "cache":
            c.add_(0.2)
        return out, c

    assert not runner._matches_eager(function, x, cache, expected, next_cache, (False, True))


def test_gate_rejects_nonfinite_reference_and_none_mask(tmp_path):
    runner = AOTChunkRunner(Encoder().eval(), tmp_path)
    x, cache = example()
    ref, next_cache = _eager_chunk(runner.encoder, x, cache)
    assert not runner._matches_eager(lambda *args: (ref, next_cache[0]), x, cache, ref, next_cache, (True, False))
    ref.fill_(float("nan"))
    assert not runner._matches_eager(lambda *args: (ref, next_cache[0]), x, cache, ref, next_cache, (False, True))


def test_gate_private_validation_inputs_and_relative_stats(tmp_path):
    runner = AOTChunkRunner(Encoder().eval(), tmp_path)
    x, cache = example()
    original_x, original_cache = x.clone(), cache[0].clone()
    ref, next_cache = _eager_chunk(runner.encoder, x, cache)

    def mutating(a, c):
        a.fill_(999)
        c.fill_(888)
        return ref.clone(), next_cache[0].clone()

    assert runner._matches_eager(mutating, x, cache, ref, next_cache, (False, True))
    assert torch.equal(x, original_x)
    assert torch.equal(cache[0], original_cache)
    assert len(runner.gate_stats) == 2
    assert all(s["relative_l2"] == 0 and s["max_abs"] == 0 for s in runner.gate_stats)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("scale", [1e-5, 1.0, 1e4])
def test_gate_budget_scales_with_reference_not_fixed_point_two(tmp_path, dtype, scale):
    runner = AOTChunkRunner(Encoder().eval(), tmp_path)
    reference = torch.full((1, 2, 1, 1, 1), scale, dtype=dtype)
    assert runner._matches_eager(lambda *args: (reference.clone(),), reference, [], reference, [], ())
    bad = reference * 1.25
    assert not runner._matches_eager(lambda *args: (bad,), reference, [], reference, [], ())
    assert runner.gate_stats[0]["relative_l2"] > 0.2


def test_configurable_gate_and_relative_l2_enforced(tmp_path):
    ref = torch.ones(1, 2, 1, 1, 1)
    function = lambda *args: (ref + 0.01,)
    permissive = AOTChunkRunner(Encoder().eval(), tmp_path, gate_atol=0.02, gate_relative_l2=0.02)
    assert permissive._matches_eager(function, ref, [], ref, [], ())
    strict = AOTChunkRunner(Encoder().eval(), tmp_path, gate_atol=0.02, gate_relative_l2=0.001)
    assert not strict._matches_eager(function, ref, [], ref, [], ())
    zero = torch.zeros_like(ref)
    assert not permissive._matches_eager(lambda *args: (ref,), zero, [], zero, [], ())
    for value in (-1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            AOTChunkRunner(Encoder().eval(), tmp_path, gate_atol=value)


def test_runtime_clones_reused_latent_and_cache_buffers(tmp_path):
    runner = AOTChunkRunner(Encoder().eval(), tmp_path)
    workspace = torch.empty(1, 2, 2, 2, 2)

    def function(x, *present):
        workspace.copy_(x)
        x.fill_(-100)  # cannot escape into caller input
        for entry in present:
            entry.fill_(-200)
        return workspace, workspace[:, :, -1:]

    def compile(key, *args):
        runner.output_none_masks[key] = (False, True)
        return function

    runner._compile = compile
    x, cache = example()
    original = x.clone()
    original_cache = cache[0]
    with torch.no_grad():
        out = runner._dispatch(x, cache)
        saved_cache = cache[0]
        runner._dispatch(x + 10, cache)
    torch.testing.assert_close(out, original)
    torch.testing.assert_close(saved_cache, original[:, :, -1:])
    assert torch.equal(x, original)
    assert (original_cache == 1).all()
    assert out.data_ptr() != workspace.data_ptr()


def test_runtime_failure_preserves_inputs_and_disables_variant(tmp_path):
    runner = AOTChunkRunner(Encoder().eval(), tmp_path)
    calls = []

    def broken(x, *cache):
        x.fill_(-100)
        for entry in cache:
            entry.fill_(-200)
        raise RuntimeError("workspace failure")

    def compile(key, *args):
        calls.append(1)
        runner.output_none_masks[key] = (False, True)
        return broken

    runner._compile = compile
    x, cache = example()
    saved_x, saved_cache = x.clone(), cache[0].clone()
    with torch.no_grad():
        out = runner._dispatch(x, cache)
        torch.testing.assert_close(out, saved_x)
        assert torch.equal(x, saved_x)
        # Same input metadata: failure must stay negatively cached.
        again = runner._dispatch(x, [saved_cache, None])
        torch.testing.assert_close(again, saved_x)
    assert len(calls) == 1
    torch.testing.assert_close(saved_cache, torch.ones_like(saved_cache))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_default_gate_budgets_are_one_eps_rms_and_two_eps_relative_l2(tmp_path, dtype):
    """The defaults are acceptance budgets, not magic constants: atol is one dtype epsilon of each reference
    tensor's RMS and the independent relative-L2 budget is two epsilon. With a reference of exactly 2.0 one ULP is
    two epsilon, so both boundaries are exercised exactly rather than approximately."""
    reference = torch.full((1, 2, 2, 2, 4), 2.0, dtype=dtype)
    eps = torch.finfo(dtype).eps
    infinity = torch.full_like(reference, float("inf"))
    one_ulp = torch.nextafter(reference, infinity)
    two_ulps = torch.nextafter(one_ulp, infinity)

    runner = AOTChunkRunner(Encoder().eval(), tmp_path, gate_rtol=0.0)
    # One ULP is exactly the atol budget and half the relative-L2 budget.
    assert runner._matches_eager(lambda *args: (one_ulp,), reference, [], reference, [], ())
    assert runner.gate_stats[0]["atol"] == pytest.approx(eps * 2.0)
    assert runner.gate_stats[0]["relative_l2"] == pytest.approx(eps)
    # Two ULP is twice the atol budget; relative L2 (two epsilon) is still at its own limit, so atol must reject.
    assert not runner._matches_eager(lambda *args: (two_ulps,), reference, [], reference, [], ())

    # A loose atol leaves only the relative-L2 budget to reject three epsilon of error.
    l2_only = AOTChunkRunner(Encoder().eval(), tmp_path, gate_rtol=0.0, gate_atol=1.0)
    three_ulps = torch.nextafter(two_ulps, infinity)
    assert not l2_only._matches_eager(lambda *args: (three_ulps,), reference, [], reference, [], ())


def test_real_export_program_passes_the_gate_on_cpu(tmp_path, monkeypatch):
    """Run ``torch.export`` for real and execute the exported graph in the gate.

    Only the Inductor package compile/load is replaced. This checks that the wrapper exports at all (the cache
    list mutation and the ``None`` entries included) and that the traced graph reproduces the eager latent and
    cache; Inductor compilation and the real package remain device work.
    """
    programs = []

    def compile(exported, package_path, **kwargs):
        programs.append(exported)
        Path(package_path).write_bytes(b"exported")

    class Function:
        def __init__(self, program):
            self.program = program

        def get_constant_fqns(self):
            return []

        def load_constants(self, *args, **kwargs):
            pass

        def __call__(self, *args):
            return self.program.module()(*args)

    def load(path, **kwargs):
        assert Path(path).read_bytes() == b"exported"
        return Function(programs[-1])

    monkeypatch.setattr(torch._inductor, "aoti_compile_and_package", compile)
    monkeypatch.setattr(torch._inductor, "aoti_load_package", load)
    runner = AOTChunkRunner(Encoder().eval(), tmp_path)
    x, cache = example()
    # _compile is called directly: the CUDA guard lives in __call__, and the export itself runs on CPU.
    with torch.no_grad():
        function = runner._compile(("real-export",), x, cache)
    assert function is not None, "the exported graph must pass the latent/cache gate"
    assert runner.output_none_masks[("real-export",)] == (False, True)
    assert runner.gate_stats and all(stats["relative_l2"] == 0 and stats["max_abs"] == 0 for stats in runner.gate_stats)


def test_package_paths_isolated_and_atomic_with_fake_compiler(tmp_path, monkeypatch):
    # Exercise file orchestration only; never invoke torch export/Inductor.
    compiled_paths = []
    monkeypatch.setattr(torch.export, "export", lambda *args, **kwargs: object())

    def compile(exported, package_path, **kwargs):
        path = Path(package_path)
        path.write_bytes(b"fake package")
        compiled_paths.append(path)

    class Function:
        def get_constant_fqns(self):
            return ["encoder.weight", "encoder.offset"]

        def load_constants(self, *args, **kwargs):
            pass

        def __call__(self, x, *present):
            return x.clone(), x[:, :, -1:].clone()

    def load(path, **kwargs):
        assert Path(path).read_bytes() == b"fake package"
        assert ".building." not in path
        return Function()

    monkeypatch.setattr(torch._inductor, "aoti_compile_and_package", compile)
    monkeypatch.setattr(torch._inductor, "aoti_load_package", load)
    x, cache = example()
    runners = [AOTChunkRunner(Encoder().eval(), tmp_path) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(runner._compile, ("same-key",), x, cache) for runner in runners]
        assert all(future.result() is not None for future in futures)
    assert compiled_paths[0].parent != compiled_paths[1].parent
    assert all(not path.exists() for path in compiled_paths)
    assert len(list(tmp_path.glob("*/*.pt2"))) == 2
