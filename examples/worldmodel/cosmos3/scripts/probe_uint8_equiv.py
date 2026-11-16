"""Does the uint8 device path reproduce the upstream/current preprocessing, pixel and latent?

Both paths start from the same uint8 pixels (the dataset's upstream-faithful quantisation):
  A (current/upstream order): host float32 /127.5-1 -> per-sample 4-D reflect pad -> bf16
  B (uint8 device path):      uint8 H2D -> aclnn fp32 /127.5-1 -> bf16 -> cat/flip reflect pad
Then the same frozen bf16 VAE encodes both; compare the latent relL2 with the VAE's own bf16 noise.

    python scripts/probe_uint8_equiv.py --checkpoint /mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-Policy-DROID
"""

import argparse
import sys
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--pad", type=int, default=16)
    parser.add_argument("--pad-h", type=int, default=4)
    return parser.parse_args()


def main():
    import torch_npu  # noqa: F401

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from examples.worldmodel.cosmos3.video_pad_norm import pad_normalize_torch, torch_reference
    from examples.worldmodel.cosmos3.vae_fast_encoder import enable_fast_encoder

    args = parse_args()
    torch.npu.set_device(0)
    torch.npu.conv.allow_hf32 = False
    torch.set_float32_matmul_precision("highest")

    shape = (args.batch, 3, 33, 540, 640)
    host_uint8 = torch.randint(0, 256, shape, dtype=torch.uint8)

    reference_fp32 = torch_reference(host_uint8, args.pad_h, args.pad)     # CPU fp32, path A
    device_uint8 = host_uint8.to("npu:0")
    device = pad_normalize_torch(device_uint8, args.pad_h, args.pad)       # NPU bf16, path B
    # Path A reaches the VAE through the same device bf16 cast as path B; that isolates the fp32 division.
    reference = reference_fp32.to("npu:0", torch.bfloat16)
    torch.npu.synchronize()

    ref_cpu = reference.cpu().float()
    dev_cpu = device.cpu().float()
    pixel_max = float((dev_cpu - ref_cpu).abs().max())
    pixel_neq = float((device.cpu() != reference.cpu()).float().mean())
    print(f"[pixel] bf16_neq_fraction={pixel_neq:.3e} max_abs={pixel_max:.3e}")

    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(args.checkpoint / "vae", torch_dtype=torch.bfloat16).eval()
    for parameter in vae.parameters():
        parameter.requires_grad = False
    vae = enable_fast_encoder(vae).to("npu:0")

    def encode(video):
        with torch.no_grad():
            out = vae.encode(video)
            dist = getattr(out, "latent_dist", None)
            return dist.mode() if dist is not None else out

    latent_ref = encode(reference)
    latent_dev = encode(device)
    torch.npu.synchronize()
    a, b = latent_dev.float(), latent_ref.float()
    rel = float(((a - b).norm() / b.norm()).item())
    print(f"[latent] relL2={rel:.3e} max_abs={float((a - b).abs().max()):.3e} "
          f"ref_rms={float(b.pow(2).mean().sqrt()):.3f}")
    print("[context] VAE's own bf16 compute noise vs fp32 is ~2e-2 relL2; the bf16 storage cast ~4e-3.")


if __name__ == "__main__":
    main()
