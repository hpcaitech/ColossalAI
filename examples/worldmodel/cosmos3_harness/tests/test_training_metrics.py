"""Tests for the structured training metrics (records, aggregation, sinks, anomaly checks).

Everything here runs on CPU without an accelerator, W&B or TensorBoard.
"""

import json
import sys
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training_metrics import (
    AnomalyDetector,
    ConsoleObserver,
    JsonlObserver,
    MetricsAggregator,
    MetricsObserver,
    RankFileReader,
    Stopwatch,
    TrainingMetrics,
    WandbObserver,
    aggregate_records,
    flatten,
    make_event,
    make_record,
)


class _Collector(MetricsObserver):
    def __init__(self):
        self.manifests = []
        self.records = []
        self.events = []
        self.summaries = []

    def on_run_start(self, manifest):
        self.manifests.append(dict(manifest))

    def on_step(self, record):
        self.records.append(dict(record))

    def on_event(self, event):
        self.events.append(dict(event))

    def on_run_end(self, summary):
        self.summaries.append(dict(summary))


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_make_record_drops_missing_sections_and_leaves():
    record = make_record(3, 1, data={"samples": 2, "pad_tokens": None}, loss=None, perf={"step_ms": 5.0})

    assert record["type"] == "record" and record["schema"] == 1
    assert record["step"] == 3 and record["rank"] == 1
    assert record["data"] == {"samples": 2}
    assert "loss" not in record
    assert record["perf"] == {"step_ms": 5.0}


def test_flatten_names_sections_with_a_slash():
    assert flatten({"data": {"samples": 8}, "loss": {"total": 2.0}, "step": 1}) == {
        "data/samples": 8,
        "loss/total": 2.0,
        "step": 1,
    }


def test_aggregate_sums_counts_maximises_durations_and_averages_loss():
    rank0 = make_record(
        5, 0, data={"samples": 8, "supervised_tokens": 20000}, loss={"total": 2.0}, perf={"step_ms": 4000}
    )
    rank1 = make_record(
        5, 1, data={"samples": 8, "supervised_tokens": 22000}, loss={"total": 4.0}, perf={"step_ms": 5000}
    )

    aggregated = aggregate_records([rank0, rank1])

    assert aggregated["data"]["samples"] == 16
    assert aggregated["data"]["supervised_tokens"] == 42000
    assert aggregated["loss"]["total"] == 3.0
    assert aggregated["perf"]["step_ms"] == 5000
    assert aggregated["perf"]["step_ms_mean"] == 4500
    assert aggregated["perf"]["skew"] == pytest.approx(5000 / 4500)
    assert aggregated["ranks"] == [0, 1] and aggregated["world"] == 2
    assert aggregated["type"] == "record" and aggregated["step"] == 5


def test_aggregate_of_one_record_has_no_skew_and_ignores_none_entries():
    record = make_record(7, 0, perf={"step_ms": 1000})

    aggregated = aggregate_records([None, record])

    assert aggregated["world"] == 1
    assert "skew" not in aggregated["perf"]
    assert aggregate_records([]) is None


def test_rank_file_reader_completes_a_partially_written_line(tmp_path):
    path = tmp_path / "rank-000.jsonl"
    path.write_text("", encoding="utf-8")

    first = json.dumps(make_record(1, 0))
    second = json.dumps(make_record(2, 0))
    with path.open("a", encoding="utf-8") as stream:
        stream.write(first + "\n" + second[:20])

    reader = RankFileReader(path)
    reader.read_new()
    record, events = reader.pop(1)
    assert record["step"] == 1 and events == []
    assert reader.pop(2) == (None, [])

    with path.open("a", encoding="utf-8") as stream:
        stream.write(second[20:] + "\n")
    reader.read_new()
    record, _ = reader.pop(2)
    assert record["step"] == 2


def test_rank_file_reader_returns_events_up_to_the_requested_step(tmp_path):
    path = tmp_path / "rank-001.jsonl"
    path.write_text(
        json.dumps(make_record(3, 1)) + "\n" + json.dumps(make_event(2, "loss_spike", "boom")) + "\n",
        encoding="utf-8",
    )

    reader = RankFileReader(path)
    reader.read_new()
    record, events = reader.pop(3)

    assert record["step"] == 3
    assert [event["kind"] for event in events] == ["loss_spike"]


def test_aggregator_reads_other_ranks_and_skips_missing_files(tmp_path):
    own = make_record(4, 0, data={"samples": 2})
    (tmp_path / "rank-001.jsonl").write_text(
        json.dumps(make_record(4, 1, data={"samples": 3})) + "\n", encoding="utf-8"
    )

    aggregator = MetricsAggregator(tmp_path, world_size=3, rank=0)
    records, events = aggregator.collect(4, own_record=own)

    assert sorted(record["rank"] for record in records) == [0, 1]
    assert events == []


def test_aggregator_forwards_events_written_by_other_ranks(tmp_path):
    (tmp_path / "rank-001.jsonl").write_text(
        json.dumps(make_event(3, "nonfinite_loss", "nan", "error")) + "\n", encoding="utf-8"
    )

    aggregator = MetricsAggregator(tmp_path, world_size=2, rank=0)
    records, events = aggregator.collect(5, own_record=None)

    assert [event["kind"] for event in events] == ["nonfinite_loss"]


def test_detector_flags_a_non_finite_loss():
    detector = AnomalyDetector(warmup=0)
    record = aggregate_records([make_record(1, 0, loss={"total": float("nan")}, perf={"step_ms": 1000})])

    events = detector.check(record)

    assert any(event["kind"] == "nonfinite_loss" and event["severity"] == "error" for event in events)


def test_detector_flags_a_loss_spike_after_the_ema_is_seeded():
    detector = AnomalyDetector(warmup=0, loss_spike_ratio=2.0, loss_drop_ratio=0.5)
    detector.check(aggregate_records([make_record(1, 0, loss={"total": 1.0})]))

    events = detector.check(aggregate_records([make_record(2, 0, loss={"total": 5.0})]))

    assert [event["kind"] for event in events] == ["loss_spike"]


def test_detector_flags_step_time_skew_starvation_and_memory_pressure():
    detector = AnomalyDetector(warmup=0, device_capacity_gib=40.0)
    for step in range(1, 7):
        detector.check(aggregate_records([make_record(step, 0, perf={"step_ms": 1000})]))

    record = aggregate_records(
        [
            make_record(
                7,
                0,
                perf={"step_ms": 9000, "skew": 1.5},
                data={"wait_ms": 4000},
                mem={"allocated_gib": 39.0},
            )
        ]
    )
    events = detector.check(record)

    kinds = {event["kind"] for event in events}
    assert {"step_time_jump", "rank_skew", "data_starvation", "memory_pressure"} <= kinds


def test_training_metrics_writes_manifest_records_and_summary(tmp_path):
    collector = _Collector()
    metrics = TrainingMetrics(out_dir=tmp_path, rank=0, world_size=1, log_every=2, aggregate_observers=[collector])
    metrics.start({"git_commit": "abc"})
    metrics.step(1, data={"samples": 4}, loss={"total": 1.0}, perf={"step_ms": 100})
    metrics.step(2, data={"samples": 4}, perf={"step_ms": 100})
    metrics.finish({"steps": 2})

    assert json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))["git_commit"] == "abc"
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["steps"] == 2

    lines = _jsonl(tmp_path / "rank-000.jsonl")
    records = [line for line in lines if line["type"] == "record"]
    assert [record["step"] for record in records] == [1, 2]
    assert lines[-1]["type"] == "summary"

    assert [record["step"] for record in collector.records] == [1, 2]
    assert collector.summaries == [{"steps": 2}]


def test_training_metrics_forwards_an_event_at_the_next_log_step(tmp_path):
    collector = _Collector()
    metrics = TrainingMetrics(out_dir=tmp_path, rank=0, world_size=1, log_every=5, aggregate_observers=[collector])
    metrics.start()

    metrics.event(2, "custom", "boom")
    metrics.step(5, data={"samples": 1})

    assert [event["kind"] for event in collector.events] == ["custom"]
    assert [line["type"] for line in _jsonl(tmp_path / "rank-000.jsonl")] == ["event", "record"]


def test_training_metrics_flushes_an_event_that_arrives_after_the_last_log_step(tmp_path):
    collector = _Collector()
    metrics = TrainingMetrics(out_dir=tmp_path, rank=0, world_size=1, log_every=5, aggregate_observers=[collector])
    metrics.start()

    metrics.step(1, data={"samples": 1})
    metrics.event(2, "custom", "late")
    metrics.finish({})

    assert [event["kind"] for event in collector.events] == ["custom"]
    assert collector.summaries == [{}]


def test_wandb_observer_logs_flattened_records_and_events(monkeypatch):
    fake = types.ModuleType("wandb")
    fake.init_calls, fake.logs, fake.finished = [], [], 0

    def init(**kwargs):
        fake.init_calls.append(kwargs)

    def log(payload, step=None):
        fake.logs.append((payload, step))

    def finish():
        fake.finished += 1

    fake.init, fake.log, fake.finish = init, log, finish
    monkeypatch.setitem(sys.modules, "wandb", fake)

    observer = WandbObserver("project", run_name="run", mode="offline", tags=["t"])
    observer.on_run_start({"env": {"torch": "2.13"}})
    observer.on_step(make_record(3, 0, loss={"total": 1.5}, data={"samples": 2}))
    observer.on_event(make_event(3, "loss_spike", "boom"))
    observer.on_run_end({"steps": 3})

    assert fake.init_calls[0]["project"] == "project"
    assert fake.init_calls[0]["mode"] == "offline"
    assert fake.init_calls[0]["config"]["manifest"] == {"env": {"torch": "2.13"}}
    payload, logged_step = fake.logs[0]
    assert payload["loss/total"] == 1.5 and payload["data/samples"] == 2
    assert logged_step == 3
    assert fake.logs[1] == ({"events/loss_spike": 1}, 3)
    assert fake.finished == 1


def test_wandb_observer_reports_a_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)

    with pytest.raises(RuntimeError, match="pip install wandb"):
        WandbObserver("project").on_run_start({})


def test_console_observer_prints_a_step_line_and_events(capsys):
    observer = ConsoleObserver()
    observer.on_step(
        aggregate_records(
            [
                make_record(
                    12,
                    0,
                    loss={"total": 2.5, "vision": 0.2, "action": 0.03},
                    data={"samples": 8, "supervised_tokens": 29552, "wait_ms": 10.0, "vae_ms": 700.0},
                    perf={"step_ms": 5000.0, "forward_ms": 1800.0, "backward_ms": 2400.0, "optim_ms": 500.0},
                    mem={"allocated_gib": 36.8},
                )
            ]
        )
    )
    observer.on_event(make_event(12, "loss_spike", "boom", "warn", ratio=3.0))

    output = capsys.readouterr().out
    assert "[step 12]" in output and "loss 2.500000" in output and "mem 36.8 GiB" in output
    assert "[event step 12] warn loss_spike: boom" in output


def test_stopwatch_falls_back_to_wall_clock_and_is_free_when_disabled():
    watch = Stopwatch(use_events=False)
    watch.reset(enabled=True)
    with watch.section("work"):
        time.sleep(0.01)

    timings = watch.read_ms()
    assert 5.0 <= timings["work"] < 1000.0

    watch.reset(enabled=False)
    with watch.section("skipped"):
        time.sleep(0.01)
    assert watch.read_ms() == {}


def test_stopwatch_sums_every_microstep_and_labels_dispatch(monkeypatch):
    ticks = iter([0.0, 0.010, 0.020, 0.050, 0.060, 0.065])
    monkeypatch.setattr(
        sys.modules[Stopwatch.__module__], "time", types.SimpleNamespace(perf_counter=lambda: next(ticks))
    )
    watch = Stopwatch(use_events=False)
    watch.reset()
    with watch.section("forward"):
        pass
    with watch.section("forward"):
        pass
    with watch.section("optim"):
        pass
    assert watch.read_ms() == pytest.approx({"forward": 40.0, "optim": 5.0})
    assert watch.read_dispatch_ms() == pytest.approx({"forward": 40.0, "optim": 5.0})
    assert watch.timing_sources == {"forward": "host_dispatch", "optim": "host_dispatch"}
    watch.reset()
    assert watch.read_ms() == {}
    assert watch.timing_sources == {}


def test_jsonl_observer_keeps_events_and_records_in_one_file(tmp_path):
    observer = JsonlObserver(tmp_path / "rank-007.jsonl")
    observer.on_step(make_record(1, 7))
    observer.on_event(make_event(1, "custom", "boom", rank=7))
    observer.on_run_end({})

    lines = _jsonl(tmp_path / "rank-007.jsonl")
    assert [line["type"] for line in lines] == ["record", "event"]
