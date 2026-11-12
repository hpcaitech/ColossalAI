"""Opt-in CUDA AOTInductor chunks, gated against eager latent and every cache.

Packages are private to a runner/process (no cross-rank package reuse). Weights
are identified by tensor identity and mutation version without CPU copies or
hashing their contents. Recreate/invalidate the runner after changes through
``.data`` or external storage, which PyTorch's version counter cannot detect.
No gate on a single chunk establishes end-to-end accuracy for a whole video.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
import threading
from pathlib import Path

import torch

__all__ = ["AOTChunkRunner"]


def _tensor_signature(tensor):
    if tensor is None:
        return None
    return (
        tuple(tensor.shape),
        tensor.dtype,
        tensor.device,
        tensor.layout,
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.requires_grad,
    )


def _cache_signature(cache):
    return tuple(_tensor_signature(entry) for entry in cache)


def _variant_key(x_chunk, cache):
    return (_tensor_signature(x_chunk), _cache_signature(cache))


def _clone_tensor(tensor):
    # Preserve strides for export/runtime; use independent storage so validation
    # cannot mutate the caller. Storage offsets are deliberately rebased.
    if any(stride == 0 and size > 1 for size, stride in zip(tensor.shape, tensor.stride())):
        raise ValueError("AOT chunks do not support expanded/overlapping inputs")
    copy = torch.empty_strided(tensor.shape, tensor.stride(), dtype=tensor.dtype, device=tensor.device)
    copy.copy_(tensor.detach())
    return copy


def _clone_cache(cache):
    return [None if entry is None else _clone_tensor(entry) for entry in cache]


def _eager_chunk(encoder, x_chunk, cache):
    local_cache = _clone_cache(cache)
    with torch.no_grad():
        out = encoder(_clone_tensor(x_chunk), feat_cache=local_cache, feat_idx=[0])
    # Snapshot before a compiled callable can reuse a shared output buffer.
    return out.detach().clone(), _clone_cache(local_cache)


class _ChunkForAOT(torch.nn.Module):
    def __init__(self, encoder, none_mask):
        super().__init__()
        self.encoder = encoder
        self.none_mask = tuple(none_mask)

    def forward(self, x_chunk, *present):
        iterator = iter(present)
        cache = [None if is_none else next(iterator) for is_none in self.none_mask]
        out = self.encoder(x_chunk, feat_cache=cache, feat_idx=[0])
        return (out, *[entry for entry in cache if entry is not None])


class AOTChunkRunner:
    """CUDA inference-only dispatcher; construction on CPU before ``.to`` is OK.

    ``gate_rtol`` is an elementwise relative tolerance. ``gate_atol=None`` uses
    one output-dtype epsilon times the eager tensor RMS, separately for each
    latent/cache tensor (no fixed latent-scale assumption). An explicit atol is
    in that tensor's units. ``gate_relative_l2=None`` additionally bounds relative
    L2 error by two output-dtype epsilons. These are experimental acceptance
    budgets, not empirical guarantees; pass checkpoint-specific budgets if known.
    Stats for every attempted gate are available in ``gate_stats`` and verbose logs.
    """

    def __init__(
        self,
        encoder,
        save_dir=None,
        verbose=False,
        *,
        gate_rtol=1e-3,
        gate_atol=None,
        gate_relative_l2=None,
        version_identity=None,
    ):
        for name, value in (("gate_rtol", gate_rtol), ("gate_atol", gate_atol), ("gate_relative_l2", gate_relative_l2)):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        self.encoder = encoder
        self.save_dir = Path(save_dir) if save_dir is not None else Path("aot_vae")
        self.verbose = verbose
        self.gate_rtol = gate_rtol
        self.gate_atol = gate_atol
        self.gate_relative_l2 = gate_relative_l2
        self.version_identity = version_identity
        self.functions = {}
        self.failed_variants = set()
        self.output_none_masks = {}
        # user_managed=True requires the CUDA storage to outlive each package.
        self.constant_tensors = {}
        self.gate_stats = []
        self._package_dir = None
        self._lock = threading.RLock()

    def invalidate(self):
        """Forget successful/failed variants after untracked external changes."""
        with self._lock:
            self.functions.clear()
            self.failed_variants.clear()
            self.output_none_masks.clear()
            self.constant_tensors.clear()

    def _encoder_identity(self):
        weights = []
        for name, tensor in (*self.encoder.named_parameters(), *self.encoder.named_buffers()):
            try:
                version = tensor._version
            except RuntimeError as error:
                raise ValueError("AOT requires version-tracked weights, not inference tensors") from error
            weights.append((name, id(tensor), tensor.data_ptr(), version, _tensor_signature(tensor)))
        modules = []
        for name, module in self.encoder.named_modules():
            forward = getattr(module.forward, "__func__", module.forward)

            def setting(value):
                if isinstance(value, (str, int, float, bool, type(None), torch.dtype, torch.device)):
                    return value
                if isinstance(value, (tuple, list)):
                    return tuple(setting(item) for item in value)
                # Parameters/buffers/submodules are fingerprinted separately.
                return type(value).__qualname__

            settings = tuple(
                (key, setting(value))
                for key, value in vars(module).items()
                if not key.startswith("_") or key == "_padding"
            )
            modules.append((name, type(module).__module__, type(module).__qualname__, id(forward), settings))
        return (torch.__version__, torch.version.cuda, self.version_identity, tuple(weights), tuple(modules))

    def __call__(self, x_chunk, cache):
        if x_chunk.device.type != "cuda":
            raise ValueError(f"VAE AOT requires CUDA inputs; got {x_chunk.device}. Disable VAE AOT on CPU/NPU/MPS.")
        if any(entry is not None and entry.device != x_chunk.device for entry in cache):
            raise ValueError("VAE AOT input and all cache tensors must be on the same CUDA device")
        if self.encoder.training:
            raise ValueError("VAE AOT requires an eval() encoder")
        if torch.is_grad_enabled() and (
            x_chunk.requires_grad or any(p.requires_grad for p in self.encoder.parameters())
        ):
            raise ValueError("VAE AOT is inference only; freeze the encoder and use no_grad()")
        # Serialize calls too: compiled functions can reuse their workspace.
        with self._lock, torch.no_grad(), torch.cuda.device(x_chunk.device):
            return self._dispatch(x_chunk, cache)

    def _dispatch(self, x_chunk, cache):
        """Dispatch under the caller's device context, lock and no_grad scope."""
        key = (
            _variant_key(x_chunk, cache),
            self._encoder_identity(),
            torch.is_autocast_enabled("cuda"),
            torch.get_autocast_dtype("cuda"),
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.deterministic,
            torch.are_deterministic_algorithms_enabled(),
            torch.get_float32_matmul_precision(),
        )
        function = self.functions.get(key)
        if function is None and key not in self.failed_variants:
            try:
                function = self._compile(key, x_chunk, cache)
            except Exception as error:  # compile/export/gate errors all fall back once per variant
                self._log(f"compile/gate failed: {error}")
            if function is None:
                self.failed_variants.add(key)
            else:
                self.functions[key] = function
        if function is None:
            return self.encoder(x_chunk, feat_cache=cache, feat_idx=[0])
        try:
            # Run on private storage: an in-place compiled graph must not
            # change input/cache contents visible to another user.
            result = function(_clone_tensor(x_chunk), *[entry for entry in _clone_cache(cache) if entry is not None])
            mask = self.output_none_masks[key]
            expected = 1 + sum(not flag for flag in mask)
            if not isinstance(result, (tuple, list)) or len(result) != expected:
                raise ValueError("compiled output structure changed")
            out = result[0].detach().clone()
            returned = iter(result[1:])
            updated = [None if flag else next(returned).detach().clone() for flag in mask]
            cache[:] = updated
            return out
        except Exception as error:
            self._log(f"runtime failed, disabling variant: {error}")
            self.failed_variants.add(key)
            self.functions.pop(key, None)
            return self.encoder(x_chunk, feat_cache=cache, feat_idx=[0])

    def _log(self, message):
        if self.verbose:
            print(f"[vae-aot] {message}", flush=True)

    def _compile(self, key, x_chunk, cache):
        eager_out, eager_cache = _eager_chunk(self.encoder, x_chunk, cache)
        mask = tuple(entry is None for entry in eager_cache)
        wrapper = _ChunkForAOT(self.encoder, [entry is None for entry in cache])
        if self._package_dir is None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self._package_dir = Path(
                tempfile.mkdtemp(prefix=f"rank{os.environ.get('RANK', '0')}_pid{os.getpid()}_", dir=self.save_dir)
            )
        package = self._package_dir / f"chunk_{hashlib.sha256(repr(key).encode()).hexdigest()}.pt2"
        temporary = package.with_suffix(".building.pt2")
        try:
            with torch.no_grad():
                args = (_clone_tensor(x_chunk), *[entry for entry in _clone_cache(cache) if entry is not None])
                exported = torch.export.export(wrapper, args, strict=False)
                torch._inductor.aoti_compile_and_package(
                    exported,
                    package_path=str(temporary),
                    inductor_configs={
                        "aot_inductor.package_constants_in_so": False,
                        "emulate_precision_casts": True,
                        "comprehensive_padding": False,
                    },
                )
            os.replace(temporary, package)
            function = torch._inductor.aoti_load_package(str(package), device_index=x_chunk.device.index)
            required = set(function.get_constant_fqns())
            constants = {name: value for name, value in wrapper.state_dict().items() if name in required}
            function.load_constants(constants, check_full_update=True, user_managed=True)
            if not self._matches_eager(function, x_chunk, cache, eager_out, eager_cache, mask):
                self._log("variant failed latent/cache correctness gate")
                return None
            self.output_none_masks[key] = mask
            self.constant_tensors[key] = constants
            self._log(f"installed {package.name}")
            return function
        finally:
            temporary.unlink(missing_ok=True)

    def _matches_eager(self, function, x_chunk, cache, eager_out, eager_cache, output_none_mask):
        self.gate_stats = []
        try:
            with torch.no_grad():
                result = function(
                    _clone_tensor(x_chunk), *[entry for entry in _clone_cache(cache) if entry is not None]
                )
            if len(output_none_mask) != len(eager_cache) or tuple(entry is None for entry in eager_cache) != tuple(
                output_none_mask
            ):
                return False
            if not isinstance(result, (tuple, list)) or len(result) != 1 + sum(not flag for flag in output_none_mask):
                return False
            references = [eager_out, *[entry for entry in eager_cache if entry is not None]]
            matches = True
            for index, (actual, reference) in enumerate(zip(result, references)):
                label = (
                    "latent"
                    if index == 0
                    else f"cache[{[i for i, flag in enumerate(output_none_mask) if not flag][index - 1]}]"
                )
                if (
                    not isinstance(actual, torch.Tensor)
                    or actual.shape != reference.shape
                    or actual.dtype != reference.dtype
                    or actual.device != reference.device
                    or actual.layout != reference.layout
                ):
                    return False
                if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
                    return False
                a, b = actual.double(), reference.double()
                if a.numel() == 0:
                    self.gate_stats.append({"tensor": label, "max_abs": 0.0, "relative_l2": 0.0})
                    continue
                rms = b.square().mean().sqrt().item()
                eps = torch.finfo(reference.dtype).eps
                atol = eps * rms if self.gate_atol is None else self.gate_atol
                delta = (a - b).abs()
                diff_rms = delta.square().mean().sqrt().item()
                relative_l2 = diff_rms / max(rms, torch.finfo(torch.float64).tiny)
                max_relative = (delta / b.abs().clamp_min(max(eps * rms, torch.finfo(torch.float64).tiny))).max().item()
                stats = {
                    "tensor": label,
                    "rms": rms,
                    "atol": atol,
                    "max_abs": delta.max().item(),
                    "relative_l2": relative_l2,
                    "max_relative": max_relative,
                }
                self.gate_stats.append(stats)
                self._log(f"gate {stats}")
                budget = 2 * eps if self.gate_relative_l2 is None else self.gate_relative_l2
                matches &= bool(torch.all(delta <= atol + self.gate_rtol * b.abs())) and relative_l2 <= budget
            return matches
        except Exception as error:
            self._log(f"gate rejected malformed/failed output: {error}")
            return False
