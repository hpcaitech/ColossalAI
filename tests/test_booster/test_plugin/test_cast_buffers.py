"""``cast_buffers``: whether the mixed-precision cast reaches the model's buffers as well as its parameters.

``Module.to(dtype)`` casts both, which is what the wrapper does at ``precision="bf16"``. That is right for a
buffer holding activations and wrong for one holding a constant whose *value* matters. The case that
motivated the switch is a rotary frequency table: ``inv_freq`` moves by relL2 4.2e-04 in bf16, and multiplied
by a position in the tens of thousands that is a phase error of radians rather than of last bits. PyTorch's
own FSDP keeps the two apart for the same reason -- ``MixedPrecisionPolicy`` takes ``param_dtype`` and
``buffer_dtype`` separately.

The test reads the buffer off the module after the boost, and checks the parameters were cast either way, so
it fails both if the switch stops working and if it starts leaving the parameters alone as well.
"""

import torch
import torch.nn as nn

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.shardformer.policies.base_policy import Policy
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn

# A frequency table of the shape a rotary embedding builds, with values bf16 cannot hold: 1/10000**(i/4) for
# i in 0..3. The second and third are the ones that move.
FREQUENCIES = [1.0, 0.1, 0.01, 0.001]


class _WithFrequencies(nn.Module):
    """One parameter and one float buffer, which is the whole subject of the test."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.register_buffer("inv_freq", torch.tensor(FREQUENCIES, dtype=torch.float32))

    def forward(self, hidden_states):
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
@parameterize("cast_buffers", [True, False])
def check_cast_buffers(cast_buffers: bool):
    model = _WithFrequencies().to(torch.float32)
    plugin = HybridParallelPlugin(
        tp_size=1,
        pp_size=1,
        zero_stage=2,
        precision="bf16",
        cast_buffers=cast_buffers,
        custom_policy=_NoOpPolicy(),
    )
    booster = Booster(plugin=plugin)
    model, *_ = booster.boost(model, torch.optim.AdamW(model.parameters(), lr=1e-4))

    inner = model.unwrap()
    exact = torch.tensor(FREQUENCIES, dtype=torch.float32)
    # Either way the parameters are cast: this switch is about buffers and must not quietly widen.
    assert inner.linear.weight.dtype == torch.bfloat16, f"parameters must be cast, got {inner.linear.weight.dtype}"
    if cast_buffers:
        assert inner.inv_freq.dtype == torch.bfloat16, f"expected the default to cast, got {inner.inv_freq.dtype}"
        assert not torch.equal(
            inner.inv_freq.float().cpu(), exact
        ), "bf16 cannot hold 0.1 or 0.001, so the table must have moved"
    else:
        assert (
            inner.inv_freq.dtype == torch.float32
        ), f"cast_buffers=False must leave a fp32 buffer alone, got {inner.inv_freq.dtype}"
        assert torch.equal(
            inner.inv_freq.float().cpu(), exact
        ), f"cast_buffers=False must leave the values alone, got {inner.inv_freq.float().cpu().tolist()}"


class _OwnInitSubclass(HybridParallelPlugin):
    """A subclass that writes its own ``__init__`` and does not chain to this plugin's.

    ``MoeHybridParallelPlugin`` is that subclass in this repository. An argument added to
    ``HybridParallelPlugin.__init__`` reaches such a subclass only if it forwards it, so without a class-level
    default the first read of ``self.cast_buffers`` raises ``AttributeError`` in code that worked before the
    argument existed.
    """

    def __init__(self) -> None:
        pass


def check_subclass_default():
    """A subclass that never assigns the attribute still reads the pre-switch behaviour."""
    plugin = _OwnInitSubclass()
    assert plugin.cast_buffers is True, "a subclass that does not forward the argument must keep the cast"
    assert plugin.cast_inputs is True, "the same contract the neighbouring argument already has"


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")
    check_cast_buffers()
    check_subclass_default()


@rerun_if_address_is_in_use()
def test_cast_buffers():
    spawn(run_dist, 2)


if __name__ == "__main__":
    test_cast_buffers()
