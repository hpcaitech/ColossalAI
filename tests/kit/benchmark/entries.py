"""Benchmark-only entries: models measured before they have a model_zoo entry. ``config.model.spec_of``
looks a name up in model_zoo first, then here. When a model_zoo entry is added for the model, its ``BenchmarkSpec`` moves
there and the entry here is removed.

They are kept out of model_zoo because other tests select model_zoo entries by substring (``"diffusers"``,
``"transformers_qwen3"``) and would pick these up.
"""

from typing import Dict

from .config.data import LatentTokens, LatentVideo, PackedText, flow_matching_loss
from .config.model import BenchmarkSpec, HFModel
from .metrics.compute import moe_decoder_flops_per_token

QWEN3_MOE = "Qwen/Qwen3-30B-A3B-Base"
QWEN3_MOE_REVISION = "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9"
WAN22_TI2V = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WAN22_TI2V_REVISION = "b8fff7315c768468a5333511427288870b2e9635"

# English web text, for the router of a MoE layer to see the tokens it is trained on
FINEWEB_EDU = dict(
    dataset="HuggingFaceFW/fineweb-edu",
    dataset_revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
    file="sample/10BT/000_00000.parquet",
)

# Wan2.2-TI2V-5B at 480×832, 81 frames: its VAE compresses 4× in time and 16× in space to 48 channels, the
# transformer cuts 1×2×2 patches; the umT5 text encoder gives 512 tokens of 4096
WAN22_TI2V_480P = LatentVideo(channels=48, frames=21, height=30, width=52, patch=(1, 2, 2), text_len=512, text_dim=4096)

ENTRIES: Dict[str, BenchmarkSpec] = {
    # Qwen3-MoE with its published weights and real text, so that the router balances the experts as in
    # training; FLOPs are counted on the devices, where the routing runs
    "qwen3_moe": BenchmarkSpec(
        models={
            "30b-a3b": HFModel(
                QWEN3_MOE,
                QWEN3_MOE_REVISION,
                "transformers.Qwen3MoeForCausalLM",
                config_overrides={"attn_implementation": "sdpa", "use_cache": False},
            )
        },
        seq_len=4096,
        batch_fn=PackedText(tokenizer=QWEN3_MOE, tokenizer_revision=QWEN3_MOE_REVISION, **FINEWEB_EDU),
        flops_fn=moe_decoder_flops_per_token,
        count_on_device=True,
    ),
    # the Wan2.2-TI2V-5B diffusion transformer on latents; no FLOPs formula yet
    "wan2_2_ti2v": BenchmarkSpec(
        models={
            "5b": HFModel(WAN22_TI2V, WAN22_TI2V_REVISION, "diffusers.WanTransformer3DModel", subfolder="transformer")
        },
        seq_len=WAN22_TI2V_480P.tokens,
        batch_fn=WAN22_TI2V_480P,
        tokens_fn=LatentTokens(WAN22_TI2V_480P.patch),
        loss_fn=flow_matching_loss,
        searchable=("micro_batch",),
    ),
}
