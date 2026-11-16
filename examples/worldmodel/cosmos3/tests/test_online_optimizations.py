"""CPU contracts for the verified Cosmos 3 kernel and scheduling optimizations.

Covers:
- ``dit_norm_kernel``'s sum-mean rewrite, bitwise equal in forward and backward on 2-D and 3-D shapes;
- ``vae_norm_kernel`` tile planners, CPU/dtype/bias fallbacks and module patching (with stubbed ``tilelang``);
- ``video_pad_norm`` block sizing, dtype guards and CPU fallback (with stubbed ``tilelang``);
- ``fast_color_jitter`` bitwise hue/pipeline match against torchvision and ``COSMOS3_FAST_JITTER=1`` wiring;
- ``packing.compute_loss`` equivalence after switching ``pow(2)`` to ``square()``;
- ``train_sft`` CLI flags for ``--vae-norm-kernel``, ``--dit-sum-mean-norm`` and ``--prefetch-factor``.
"""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm
from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers.transformer_cosmos3 import Cosmos3NemotronRMSNorm

from examples.worldmodel.cosmos3 import dit_norm_kernel as dsm
from examples.worldmodel.cosmos3 import fast_color_jitter as fcj
from examples.worldmodel.cosmos3 import packing
from examples.worldmodel.cosmos3 import train_sft as entry
from examples.worldmodel.cosmos3.droid_dataset import CAMS, DroidActionPolicyDataset


def _install_tilelang_stub(monkeypatch):
    """Allow importing TileLang kernel modules on CPU-only hosts without tilelang installed."""
    tl = types.ModuleType("tilelang")
    tl.PassConfigKey = SimpleNamespace(TL_ASCEND_AUTO_SYNC="sync", TL_ASCEND_MEMORY_PLANNING="mem")
    tl.jit = lambda *a, **kw: (lambda fn: fn)

    tl_lang = types.ModuleType("tilelang.language")
    tl_lang.Tensor = lambda shape, dtype: (shape, dtype)
    tl_lang.tile = SimpleNamespace()
    tl.language = tl_lang

    monkeypatch.setitem(sys.modules, "tilelang", tl)
    monkeypatch.setitem(sys.modules, "tilelang.language", tl_lang)


@pytest.mark.parametrize("shape", [(16, 128), (8, 4, 128), (4, 2048)])
def test_dit_sum_mean_norm_matches_reference_bitwise(shape):
    torch.manual_seed(3)
    module = Cosmos3NemotronRMSNorm(shape[-1], eps=1e-5).to(dtype=torch.bfloat16)
    with torch.no_grad():
        module.weight.copy_(torch.randn(shape[-1], dtype=torch.float32).clamp(0.5, 1.5))

    x_base = torch.randn(*shape, dtype=torch.bfloat16) * 0.5
    grad_out = torch.randn(*shape, dtype=torch.bfloat16) * 0.05

    x_ref = x_base.detach().clone().requires_grad_(True)
    module.weight.grad = None
    ref_out = module(x_ref)
    ref_out.backward(grad_out)
    ref_gx, ref_gw = x_ref.grad.clone(), module.weight.grad.clone()

    x_tst = x_base.detach().clone().requires_grad_(True)
    module.weight.grad = None
    tst_out = dsm._sum_mean_forward(module, x_tst)
    tst_out.backward(grad_out)

    assert torch.equal(tst_out, ref_out)
    assert torch.equal(x_tst.grad, ref_gx)
    assert torch.equal(module.weight.grad, ref_gw)


def test_enable_sum_mean_norm_patches_only_nemotron_modules():
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = Cosmos3NemotronRMSNorm(32, eps=1e-5)
            self.plain = RMSNorm(32, eps=1e-5, elementwise_affine=True, bias=False)

    model = ToyModel()
    plain_orig = model.plain.forward
    dsm.enable_sum_mean_norm(model)
    assert model.sum_mean_norm_modules == 1
    assert model.n1.forward.__func__ is dsm._sum_mean_forward
    assert model.plain.forward == plain_orig


def test_vae_norm_kernel_planners_fallback_and_patching(monkeypatch):
    _install_tilelang_stub(monkeypatch)
    sys.modules.pop("examples.worldmodel.cosmos3.vae_norm_kernel", None)
    vnk = importlib.import_module("examples.worldmodel.cosmos3.vae_norm_kernel")

    assert vnk._plan(96, 65536) is not None
    assert vnk._plan(96, 64) is None
    assert vnk._plan_two_pass(160, 356864) is not None
    assert vnk._plan_two_pass(160, 64) is None

    def one_pass_bytes(c, chunk):
        # x tile (bf16) + row/sq/acc/norm/g_row (five fp32) + gamma (C-wide fp32).
        return 2 * c * chunk + 20 * chunk + 4 * c

    def two_pass_bytes(c, chunk):
        # row/norm/g_row (three fp32) + half/stage (two bf16) + gamma.
        return 16 * chunk + 4 * c

    # Small C used to plan a tile larger than the 192 KiB UB; the full-footprint bound must fall back
    # or shrink it, while production channel counts keep their historical chunk width.
    for c, s in ((32, 1 << 20), (96, 65536), (160, 65536)):
        plan = vnk._plan(c, s)
        assert plan is not None
        assert one_pass_bytes(c, plan[0]) <= vnk.UB_CAPACITY_BYTES
    assert vnk._plan(96, 65536)[0] == 768
    assert vnk._plan(160, 65536)[0] == 512
    two_pass = vnk._plan_two_pass(160, 356864)
    assert two_pass is not None and two_pass[0] == 8192
    assert two_pass_bytes(160, two_pass[0]) <= vnk.UB_CAPACITY_BYTES

    torch.manual_seed(0)
    norm_cf = WanRMS_norm(16, channel_first=True, images=False).eval().to(torch.bfloat16)
    with torch.no_grad():
        norm_cf.gamma.copy_(torch.randn_like(norm_cf.gamma) * 0.3 + 1.0)
    x = torch.randn(2, 16, 3, 8, 8, dtype=torch.bfloat16)
    with torch.no_grad():
        assert torch.equal(vnk._kernel_forward(norm_cf, x), norm_cf(x))

    class ToyVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cf = WanRMS_norm(8, channel_first=True, images=False)
            self.cl = WanRMS_norm(8, channel_first=False, images=True)

    vae = ToyVAE()
    cl_orig = vae.cl.forward
    vnk.enable_kernel_rmsnorm(vae)
    assert vae.kernel_rmsnorm_modules == 1
    assert vae.cf.forward.__func__ is vnk._kernel_forward
    assert vae.cl.forward == cl_orig


def test_video_pad_norm_block_sizing_and_cpu_fallback(monkeypatch):
    _install_tilelang_stub(monkeypatch)
    sys.modules.pop("examples.worldmodel.cosmos3.video_pad_norm", None)
    vpn = importlib.import_module("examples.worldmodel.cosmos3.video_pad_norm")

    assert vpn._block_for(8 * 3 * 33 * 540 * 640) == 4096
    assert vpn._block_for(4 * 3 * 17 * 544 * 656) == 2048
    assert vpn._block_for(1 * 3 * 33 * 544 * 656) == 512
    assert vpn._block_for(3 * 5 * 7) == 1

    # Odd element count forces block < 64, exercising the reference fallback path on CPU.
    video = torch.randint(0, 256, (1, 3, 1, 5, 7), dtype=torch.uint8)
    got = vpn.pad_normalize(video, pad_h=2, pad_w=3)
    want = vpn.torch_reference(video, pad_h=2, pad_w=3).to(torch.bfloat16)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, want)

    with pytest.raises(TypeError, match="uint8"):
        vpn.normalize_uint8(torch.zeros(1, 3, 1, 4, 4, dtype=torch.float32))


def test_fast_color_jitter_matches_torchvision_bitwise(monkeypatch):
    torch.manual_seed(11)
    x = torch.rand(3, 3, 32, 32)
    assert torch.equal(fcj._hue_math(x.clone(), torch.tensor(0.05)), F.adjust_hue(x.clone(), 0.05))

    ref = T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08)
    fast = fcj.FastColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08)
    torch.manual_seed(99)
    y_ref = ref(x.clone())
    torch.manual_seed(99)
    y_fast = fast(x.clone())
    assert torch.equal(y_fast, y_ref)

    monkeypatch.setenv("COSMOS3_FAST_JITTER", "1")
    ds = DroidActionPolicyDataset.__new__(DroidActionPolicyDataset)
    ds._augmentor = None
    views = {cam: torch.rand(2, 3, 64, 64) for cam in CAMS}
    out = ds._augment(views)
    assert isinstance(ds._augmentor.transforms[-1], fcj.FastColorJitter)
    assert all(out[cam].shape == views[cam].shape for cam in CAMS)


def test_compute_loss_matches_pow2_reference():
    torch.manual_seed(0)
    pred_v = [torch.randn(1, 16, 2, 4, 4, dtype=torch.bfloat16)]
    tgt_v = [torch.randn(1, 16, 2, 4, 4, dtype=torch.float32)]
    mask_v = [torch.randint(0, 2, (1, 1, 2, 4, 4), dtype=torch.bool)]
    pred_a = [torch.randn(8, 12, dtype=torch.bfloat16)]
    tgt_a = [torch.randn(8, 12, dtype=torch.float32)]
    mask_a = [torch.ones(8, 1, dtype=torch.bool)]

    out = SimpleNamespace(sample=pred_v, action=pred_a)
    targets = {
        "vision": tgt_v,
        "vision_noisy_mask": mask_v,
        "action": tgt_a,
        "action_noisy_mask": mask_a,
        "raw_action_dim": 8,
    }
    total, comps = packing.compute_loss(out, targets, return_components=True)
    expected_v = ((pred_v[0].float() - tgt_v[0].float()).pow(2) * mask_v[0].float()).mean() * packing.VISION_LOSS_SCALE
    expected_a = (
        (pred_a[0][:, :8].float() - tgt_a[0][:, :8].float()).pow(2) * mask_a[0].float()
    ).mean() * packing.ACTION_LOSS_WEIGHT
    assert torch.equal(comps["vision"], expected_v)
    assert torch.equal(comps["action"], expected_a)
    assert torch.equal(total, expected_v + expected_a)


def test_compute_loss_square_matches_pow_on_device():
    """The CPU test cannot observe Ascend's aclnnPowTensorScalar path; repeat the comparison on NPU.

    The default path changed from pow(2) to square(), and those are mathematically equal but not necessarily
    bitwise equal across backends.  This test is skipped on hosts without torch_npu; it was validated on an
    Ascend 910B2, where all three tensors match exactly.
    """
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    torch.manual_seed(0)
    device = "npu:0"
    pred_v = [torch.randn(2, 16, 2, 4, 4, dtype=torch.bfloat16, device=device)]
    tgt_v = [torch.randn(2, 16, 2, 4, 4, dtype=torch.float32, device=device)]
    mask_v = [torch.randint(0, 2, (2, 1, 2, 4, 4), dtype=torch.bool, device=device)]
    pred_a = [torch.randn(8, 12, dtype=torch.bfloat16, device=device)]
    tgt_a = [torch.randn(8, 12, dtype=torch.float32, device=device)]
    mask_a = [torch.ones(8, 1, dtype=torch.bool, device=device)]

    out = SimpleNamespace(sample=pred_v, action=pred_a)
    targets = {
        "vision": tgt_v,
        "vision_noisy_mask": mask_v,
        "action": tgt_a,
        "action_noisy_mask": mask_a,
        "raw_action_dim": 8,
    }
    total, comps = packing.compute_loss(out, targets, return_components=True)
    expected_v = ((pred_v[0].float() - tgt_v[0].float()).pow(2) * mask_v[0].float()).mean() * packing.VISION_LOSS_SCALE
    expected_a = (
        (pred_a[0][:, :8].float() - tgt_a[0][:, :8].float()).pow(2) * mask_a[0].float()
    ).mean() * packing.ACTION_LOSS_WEIGHT

    assert torch.equal(comps["vision"], expected_v)
    assert torch.equal(comps["action"], expected_a)
    assert torch.equal(total, expected_v + expected_a)


def test_new_optimization_cli_flags():
    parsed = entry.parse_args(
        [
            "--data",
            "/data",
            "--checkpoint",
            "/ckpt",
            "--vae-norm-kernel",
            "--dit-sum-mean-norm",
            "--prefetch-factor",
            "4",
        ]
    )
    assert parsed.vae_norm_kernel is True
    assert parsed.dit_sum_mean_norm is True
    assert parsed.prefetch_factor == 4
