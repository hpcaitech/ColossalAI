import math

import torch
import torch.distributed as dist
from transformers.models.mixtral import MixtralConfig, MixtralForCausalLM

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import MoeHybridParallelPlugin


def run_smoke_test() -> None:
    world_size = dist.get_world_size()
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    config = MixtralConfig(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        attn_implementation="eager",
    )
    model = MixtralForCausalLM(config)
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
    )
    booster = Booster(plugin=plugin)
    model, optimizer, _, _, _ = booster.boost(model, optimizer)

    device = torch.device("cuda", torch.cuda.current_device())
    input_ids = torch.randint(0, config.vocab_size, (1, 32), device=device)
    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
    if not torch.isfinite(loss):
        raise AssertionError(f"Mixtral smoke test produced a non-finite loss: {loss}")

    booster.backward(loss, optimizer)
    optimizer.step()
    grad_norm = optimizer.get_grad_norm()
    if grad_norm is None or not math.isfinite(grad_norm) or grad_norm <= 0:
        raise AssertionError(f"Mixtral smoke test produced an invalid gradient norm: {grad_norm}")
    optimizer.zero_grad()
    torch.cuda.synchronize()

    if dist.get_rank() == 0:
        print(f"Mixtral smoke test passed with expert parallel size {world_size}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Mixtral smoke test requires CUDA")

    colossalai.launch_from_torch()
    try:
        run_smoke_test()
    finally:
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
