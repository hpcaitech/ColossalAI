"""Self-tests of the maximum search: the bisection against a known limit, and probes in fresh processes that run out
of memory, on the CPU (the out-of-memory error is raised by the model)."""

import pytest
from torch import nn
from torch.multiprocessing.spawn import ProcessRaisedException

from colossalai.testing import rerun_if_address_is_in_use
from tests.kit.benchmark import BenchmarkSpec
from tests.kit.benchmark.config.setup import Setup
from tests.kit.benchmark.runner.measure import DEVICE_OUT_OF_MEMORY, measure
from tests.kit.benchmark.scenarios.report import build_report, render_markdown
from tests.kit.benchmark.scenarios.search import probe, run_search, search_max
from tests.test_benchmark.test_benchmark_core import tiny_llama

# tokens per micro-batch above which the model "runs out of memory"
LIMIT = 128


class Limited(nn.Module):
    """A tiny Llama that raises CUDA's out-of-memory message for batches above ``LIMIT`` tokens, and a plain error for
    a batch of exactly 7 samples (an error that is not about memory)."""

    def __init__(self):
        super().__init__()
        self.inner = tiny_llama()
        self.config = self.inner.config

    def forward(self, input_ids, labels):
        if input_ids.shape[0] == 7:
            raise ValueError("not a memory problem")
        if input_ids.numel() > LIMIT:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB (simulated)")
        return self.inner(input_ids=input_ids, labels=labels)


LIMITED = BenchmarkSpec(models={"tiny": Limited}, seq_len=32, micro_batch=1, causal_attention=True)


def _setup(**kwargs):
    return Setup(entry="limited", size="tiny", config="ddp_fp32", **{"seq_len": 32, "micro_batch": 1, **kwargs})


class Threshold:
    """A probe that fits up to ``limit`` and counts its calls."""

    def __init__(self, field, limit):
        self.field, self.limit, self.values = field, limit, []

    def __call__(self, setup):
        value = getattr(setup, self.field)
        self.values.append(value)
        return {"fits": True} if value <= self.limit else {"fits": False, "reason": "too big"}


@pytest.mark.parametrize("start, limit", [(256, 5000), (4096, 1000), (1024, 1024), (256, 256), (4096, 300)])
def test_bisection_finds_the_largest_multiple_that_fits(start, limit):
    fits = Threshold("seq_len", limit)
    found = search_max(_setup(seq_len=start), "seq_len", fits, granularity=256)
    assert found["max"] == limit // 256 * 256
    assert [p["value"] for p in found["probes"]] == fits.values
    assert all(v % 256 == 0 for v in fits.values)
    # logarithmic: doubling or halving to bracket, then bisection
    assert len(fits.values) <= 2 * 5 + 2


def test_bounds_and_nothing_that_fits():
    capped = search_max(_setup(seq_len=256), "seq_len", Threshold("seq_len", 10**6), granularity=256, upper=3000)
    assert capped["max"] == 2816 and max(p["value"] for p in capped["probes"]) == 2816
    assert search_max(_setup(seq_len=1024), "seq_len", Threshold("seq_len", 100), granularity=256)["max"] is None
    batch = search_max(_setup(micro_batch=2), "micro_batch", Threshold("micro_batch", 13))
    assert batch["max"] == 13 and batch["granularity"] == 1 and batch["fixed"] == {"seq_len": 32}
    with pytest.raises(ValueError):
        search_max(_setup(), "hidden_size", Threshold("seq_len", 1))


@rerun_if_address_is_in_use()
def test_runs_record_out_of_memory_and_raise_other_errors(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert probe(_setup(seq_len=LIMIT), LIMITED)["fits"]
    assert probe(_setup(seq_len=LIMIT + 16), LIMITED) == {"fits": False, "reason": "out of device memory"}
    assert measure(_setup(seq_len=LIMIT + 16), LIMITED)["status"] == DEVICE_OUT_OF_MEMORY
    with pytest.raises(ProcessRaisedException, match="not a memory problem"):
        probe(_setup(micro_batch=7, seq_len=8), LIMITED)


@rerun_if_address_is_in_use()
def test_search_then_measure_at_the_maximum(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    setup = _setup(seq_len=32, warmup_steps=1, timed_steps=2)
    found = run_search(setup, "seq_len", granularity=16, upper=1024, repeats=1, spec=LIMITED)
    search, measurement = found["search"], found["measurement"]
    assert search["max"] == search["measured_at"] == LIMIT and search["flags"] == []
    assert measurement["setup"]["seq_len"] == LIMIT and measurement["setup"]["timed_steps"] == 2
    assert search["probe_steps"] == 3  # as many as the measurement: 1 warmup + 2 timed
    markdown = render_markdown(build_report(measurement, {}, [], search=search))
    assert "## Maximum sequence length on one device" in markdown
    assert f"Maximum sequence length on one device: **{LIMIT}**." in markdown
    assert "- maximum sequence length on one device" not in markdown
    assert "- maximum micro-batch on one device" in markdown
