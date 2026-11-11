"""``cast_inputs``: whether the plugin may rewrite the dtype of the model's forward arguments.

At ``precision="bf16"`` the wrapper casts every floating-point tensor in ``args`` and ``kwargs`` to bf16. That
is right for activations and wrong for an argument whose magnitude carries meaning: a positional index of
15163.8 comes back as 15168.0, because bf16 has no closer value, and a rotary embedding built from it is
computed at the wrong place in the sequence. ``cast_inputs=False`` leaves the arguments alone and makes the
caller responsible for the dtype, which is what ``LowLevelZeroPlugin`` already means by the same name.

The test records the dtype and the value the module's forward actually receives, rather than inspecting the
wrapper, so it fails if the cast moves somewhere else.
"""

import torch
import torch.nn as nn

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.shardformer.policies.base_policy import Policy
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn

# Large enough that bf16 cannot represent it: in [8192, 16384) bf16's step is 64, so the nearest value is 15168.
A_POSITION = 15163.8


class _Recorder(nn.Module):
    """Records what its forward was handed, and nothing else."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.seen_dtype = None
        self.seen_value = None

    def forward(self, hidden_states, positions):
        self.seen_dtype = positions.dtype
        self.seen_value = float(positions.reshape(-1)[0])
        return self.linear(hidden_states.to(self.linear.weight.dtype)).sum()


class _NoOpPolicy(Policy):
    """ShardFormer refuses a module it has no policy for; this lets a plain nn.Module through unrewritten."""

    def config_sanity_check(self) -> None:
        return None

    def preprocess(self):
        return self.model

    def module_policy(self):
        return {}

    def postprocess(self):
        return self.model


@clear_cache_before_run()
@parameterize("cast_inputs", [True, False])
def check_cast_inputs(cast_inputs: bool):
    model = _Recorder().to(torch.float32)
    plugin = HybridParallelPlugin(
        tp_size=1,
        pp_size=1,
        zero_stage=2,
        precision="bf16",
        cast_inputs=cast_inputs,
        custom_policy=_NoOpPolicy(),
    )
    booster = Booster(plugin=plugin)
    model, optimizer, *_ = booster.boost(model, torch.optim.AdamW(model.parameters(), lr=1e-4))

    device = get_accelerator().get_current_device()
    with torch.no_grad():
        model(
            hidden_states=torch.randn(2, 4, dtype=torch.float32, device=device),
            positions=torch.full((2, 4), A_POSITION, dtype=torch.float32, device=device),
        )

    recorder = model.unwrap()
    if cast_inputs:
        assert recorder.seen_dtype == torch.bfloat16, f"expected the default to cast, got {recorder.seen_dtype}"
        assert recorder.seen_value != A_POSITION, "bf16 cannot hold 15163.8, so the value must have moved"
    else:
        assert (
            recorder.seen_dtype == torch.float32
        ), f"cast_inputs=False must leave a fp32 argument alone, got {recorder.seen_dtype}"
        assert (
            recorder.seen_value == torch.tensor(A_POSITION).item()
        ), f"cast_inputs=False must leave the value alone, got {recorder.seen_value}"


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")
    check_cast_inputs()


@rerun_if_address_is_in_use()
def test_cast_inputs():
    spawn(run_dist, 2)


if __name__ == "__main__":
    test_cast_inputs()
