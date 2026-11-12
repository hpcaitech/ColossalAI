"""Encode a clip with several latent chunks per encoder call, the way the official tokenizer does.

``AutoencoderKLWan._encode`` calls the encoder once per latent chunk: a one-frame prime, then four pixel frames
at a time. The official stack calls that out as the reason the tokenizer under-uses the tensor cores at training
resolutions -- "each kernel launch operates on too little work to saturate the tensor cores" -- and instead
encodes a configurable number of *pixel* frames per call while carrying the causal cache across calls
(technical report Sec. 5.2.6: 68 frames at 256p, 24 at 480p, 12 at 720p; the released code's default is
``{"256": 68, "480": 24, "720": 8}``).

Causal caches preserve the temporal schedule across windows. Backend accumulation
and compiler choices can still change numerical results; validate the checkpoint,
dtype and device used by an experimental recipe. The chunk size must be a
multiple of 4 so stride-2 temporal downsamples keep their phase.

    from vae_chunked import enable_chunked_encode

    enable_chunked_encode(vae, 24)          # 480p, the recipe's operating point
"""

from __future__ import annotations

import torch

# The official dict, keyed by the shorter side of the canvas. Only 480 matters for the DROID recipe.
OFFICIAL_CHUNK_FRAMES = {"256": 68, "480": 24, "720": 8}


def _encode_chunked(vae, x: torch.Tensor, chunk_frames: int, chunk_runner=None) -> torch.Tensor:
    """Encode one independent clip, clearing causal state even on errors/tiling."""
    vae.clear_cache()
    try:
        return _encode_chunked_impl(vae, x, chunk_frames, chunk_runner)
    finally:
        vae.clear_cache()


def _encode_chunked_impl(vae, x: torch.Tensor, chunk_frames: int, chunk_runner=None) -> torch.Tensor:
    """``AutoencoderKLWan._encode`` with a configurable encoder-call window."""
    if not isinstance(chunk_frames, int) or chunk_frames <= 0 or chunk_frames % 4:
        raise ValueError(f"chunk_frames must be a positive multiple of 4, got {chunk_frames}")
    if x.ndim != 5 or x.shape[2] < 1:
        raise ValueError("expected a nonempty [B, C, T, H, W] clip")
    _, _, num_frame, height, width = x.shape
    num_latent = (num_frame - 1) // 4 + 1

    if vae.config.patch_size is not None:
        from diffusers.models.autoencoders.autoencoder_kl_wan import patchify

        x = patchify(x, patch_size=vae.config.patch_size)

    if vae.use_tiling and (width > vae.tile_sample_min_width or height > vae.tile_sample_min_height):
        return vae.tiled_encode(x)

    # Every window after the prime has to hold a whole number of 4-frame groups, or the stride-2 downsamples
    # shift phase and the result stops matching the per-chunk path. DROID's 33 frames need no padding (32 = 24
    # + 8); an arbitrary frame count does, and the extra latent frames are cropped away below.
    ragged = (num_frame - 1) % 4
    if ragged:
        x = torch.cat([x, x[:, :, -1:].expand(-1, -1, 4 - ragged, -1, -1)], dim=2)

    if chunk_runner is None:

        def run(chunk, feat_cache):
            return vae.encoder(chunk, feat_cache=feat_cache, feat_idx=[0])

    else:
        run = chunk_runner

    vae._enc_conv_idx = [0]
    out = run(x[:, :, :1, :, :], vae._enc_feat_map)
    parts = [out]
    for start in range(1, x.shape[2], chunk_frames):
        vae._enc_conv_idx = [0]
        parts.append(run(x[:, :, start : start + chunk_frames, :, :], vae._enc_feat_map))

    out = torch.cat(parts, 2) if len(parts) > 1 else parts[0]
    out = out[:, :, :num_latent]
    return vae.quant_conv(out)


def enable_chunked_encode(
    vae,
    chunk_frames: int,
    aot_dir=None,
    verbose: bool = False,
    *,
    gate_rtol=1e-3,
    gate_atol=None,
    gate_relative_l2=None,
    version_identity=None,
):
    """Install the wider encoder call window on ``vae``. Returns the VAE so this can wrap a constructor.

    Args:
        vae: The Wan VAE.
        chunk_frames: Pixel frames per encoder call, a multiple of 4. Zero keeps diffusers' own loop.
        aot_dir (optional): When given, compile and validate the per-chunk encoder with AOTInductor into this
            directory and dispatch to the compiled graph, falling back to eager per variant. Defaults to None,
            i.e. eager chunks.
        verbose (bool, optional): Print AOT compile/gate lines. Defaults to False.
        gate_rtol, gate_atol, gate_relative_l2: See ``AOTChunkRunner`` for relative,
            absolute and relative-L2 gate budgets. Only used when AOT is enabled.
        version_identity: Optional checkpoint/code identity passed to the runner.
    """
    if chunk_frames == 0 and aot_dir is None:
        return vae
    if not isinstance(chunk_frames, int) or chunk_frames <= 0 or chunk_frames % 4:
        raise ValueError("chunk_frames must be a positive multiple of 4 when chunking/AOT is enabled")

    runner = None
    if aot_dir is not None:
        if __package__:
            from .vae_aot import AOTChunkRunner
        else:
            from vae_aot import AOTChunkRunner

        runner = AOTChunkRunner(
            vae.encoder,
            save_dir=aot_dir,
            verbose=verbose,
            gate_rtol=gate_rtol,
            gate_atol=gate_atol,
            gate_relative_l2=gate_relative_l2,
            version_identity=version_identity,
        )

    def _encode(x):
        return _encode_chunked(vae, x, chunk_frames, runner)

    vae._encode = _encode
    vae.encode_chunk_frames = chunk_frames
    vae._aot_chunk_runner = runner
    return vae
