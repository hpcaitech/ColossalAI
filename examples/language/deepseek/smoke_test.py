import math
from typing import Tuple

import torch
import torch.distributed as dist
from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import MoeHybridParallelPlugin
from colossalai.shardformer.policies.deepseek_v3 import DeepseekV3ForCausalLMPolicy


class DeepseekV3SmokePolicy(DeepseekV3ForCausalLMPolicy):
    def module_policy(self):
        policy = super().module_policy()
        if self.shard_config.pipeline_stage_manager is None:
            # Keep the current Transformers forward signature while applying
            # ColossalAI's expert-parallel MoE replacement.
            policy.pop("DeepseekV3Model", None)
        return policy


def build_model() -> Tuple[DeepseekV3ForCausalLM, DeepseekV3Config]:
    config = DeepseekV3Config(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=1.0,
        kv_lora_rank=32,
        q_lora_rank=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        qk_nope_head_dim=16,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        max_position_embeddings=128,
        use_cache=False,
        attn_implementation="eager",
    )
    model = DeepseekV3ForCausalLM(config)
    if model.__class__.__name__ != "DeepseekV3ForCausalLM" or not any(
        module.__class__.__name__ == "DeepseekV3MoE" for module in model.modules()
    ):
        raise AssertionError("DeepSeek smoke test must use the real DeepSeek V3 MoE architecture")
    return model, config


def run_smoke_test() -> None:
    world_size = dist.get_world_size()
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    model, config = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    plugin = MoeHybridParallelPlugin(
        ep_size=world_size,
        tp_size=1,
        pp_size=1,
        zero_stage=1,
        precision="bf16",
        enable_fused_normalization=False,
        enable_flash_attention=False,
        overlap_communication=False,
        initial_scale=1,
        custom_policy=DeepseekV3SmokePolicy(),
    )
    booster = Booster(plugin=plugin)
    model, optimizer, _, _, _ = booster.boost(model, optimizer)

    device = torch.device("cuda", torch.cuda.current_device())
    input_ids = torch.randint(0, config.vocab_size, (1, 32), device=device)
    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
    if not torch.isfinite(loss):
        raise AssertionError(f"DeepSeek smoke test produced a non-finite loss: {loss}")

    booster.backward(loss, optimizer)
    optimizer.step()
    grad_norm = optimizer.get_grad_norm()
    if grad_norm is None or not math.isfinite(grad_norm) or grad_norm <= 0:
        raise AssertionError(f"DeepSeek smoke test produced an invalid gradient norm: {grad_norm}")
    optimizer.zero_grad()
    torch.cuda.synchronize()

    if dist.get_rank() == 0:
        print(f"DeepSeek V3 smoke test passed with expert parallel size {world_size}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("DeepSeek smoke test requires CUDA")

    colossalai.launch_from_torch()
    try:
        run_smoke_test()
    finally:
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
