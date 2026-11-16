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
    ASCEND_910B2_CUBE_PEAK_TFLOPS,
    AnomalyDetector,
    ConsoleObserver,
    Cosmos3FlopsModel,
    JsonlObserver,
    MetricsAggregator,
    MetricsObserver,
    RankFileReader,
    Stopwatch,
    TrainingMetrics,
    WandbObserver,
    aggregate_records,
    compute_mfu,
    cosmos3_step_flops,
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


def test_display_mse_aggregation_weights_samples_and_excludes_cp_followers():
    records = [
        make_record(
            1,
            rank,
            data={"samples": samples},
            loss={"total": value, "vision": value / 2, "action": value / 2, "objective": value * 10},
        )
        for rank, samples, value in [(0, 1, 2.0), (1, 3, 6.0), (2, 0, 1000.0)]
    ]
    loss = aggregate_records(records)["loss"]
    assert loss == {"total": 5.0, "vision": 2.5, "action": 2.5, "objective": 50.0}


def test_console_names_display_mse_objective_and_partial_rank_coverage(capsys):
    record = make_record(7, 0, loss={"total": 0.21, "vision": 0.2, "action": 0.01, "objective": 1.9})
    record.update(ranks=[0, 1], world=2, expected_world=4, complete=False)
    ConsoleObserver().on_step(record)
    text = capsys.readouterr().out
    assert "MSE 0.210000" in text and "a 0.010000" in text
    assert "objective 1.900000" in text and "ranks 2/4 (partial)" in text


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


def test_training_metrics_record_interface(tmp_path):
    collector = _Collector()
    metrics = TrainingMetrics(out_dir=tmp_path, rank=0, world_size=1, log_every=1, aggregate_observers=[collector])
    metrics.start({"test": "record_interface"})
    metrics.record("fwd", 120.5, hbm_gib=32.5)
    metrics.record("bwd", 85.0)
    metrics.record("vae", 25.0)
    metrics.step(1, data={"samples": 4}, loss={"total": 1.25})
    metrics.finish({"status": "done"})

    assert len(collector.records) == 1
    rec = collector.records[0]
    assert rec["perf"]["fwd_ms"] == 120.5
    assert rec["perf"]["forward_ms"] == 120.5
    assert rec["perf"]["bwd_ms"] == 85.0
    assert rec["perf"]["vae_ms"] == 25.0
    assert rec["mem"]["device_used_gib"] == 32.5


def test_telemetry_reduction_with_standard_keys():
    rank0 = make_record(
        step=5,
        rank=0,
        perf={
            "step_ms": 1000.0,
            "fwd_ms": 100.0,
            "bwd_ms": 150.0,
            "vae_ms": 30.0,
            "flops": 120e12,
            "flops_forward": 30e12,
            "flops_backward": 90e12,
            "mfu": 0.10,
            "peak_tflops_per_npu": 353.9,
        },
        mem={"device_used_gib": 32.0, "peak_reserved_gib": 30.0},
    )
    rank1 = make_record(
        step=5,
        rank=1,
        perf={
            "step_ms": 1000.0,
            "fwd_ms": 120.0,
            "bwd_ms": 140.0,
            "vae_ms": 35.0,
            "flops": 110e12,
            "flops_forward": 28e12,
            "flops_backward": 82e12,
            "mfu": 0.09,
            "peak_tflops_per_npu": 353.9,
        },
        mem={"device_used_gib": 34.5, "peak_reserved_gib": 31.0},
    )

    aggregated = aggregate_records([rank0, rank1])

    # Max reduction for straggler latencies and memory high-water marks
    assert aggregated["perf"]["fwd_ms"] == 120.0
    assert aggregated["perf"]["bwd_ms"] == 150.0
    assert aggregated["perf"]["vae_ms"] == 35.0
    assert aggregated["mem"]["device_used_gib"] == 34.5
    assert aggregated["mem"]["peak_reserved_gib"] == 31.0

    # FLOPs sum, but achieved TFLOPS and MFU are recomputed from the cluster totals rather than reduced.
    assert aggregated["perf"]["flops"] == 230e12
    assert aggregated["perf"]["tflops"] == pytest.approx(230.0)
    assert aggregated["perf"]["mfu"] == pytest.approx(230.0 / (2 * 353.9))


def test_mfu_and_flops_follow_the_model_geometry():
    assert ASCEND_910B2_CUBE_PEAK_TFLOPS == 353.9
    config = {
        "num_hidden_layers": 28,
        "hidden_size": 2048,
        "intermediate_size": 9216,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_act": "relu2",
        "vocab_size": 131072,
    }
    model = Cosmos3FlopsModel.from_config(config)
    breakdown = model.breakdown(ar_tokens=1000, dm_tokens=6388, samples=2)
    full = cosmos3_step_flops(model, samples=2, ar_tokens=1000, dm_tokens=6388, checkpointing="full")
    none = cosmos3_step_flops(model, samples=2, ar_tokens=1000, dm_tokens=6388, checkpointing="none")
    selective = cosmos3_step_flops(
        model, samples=2, ar_tokens=1000, dm_tokens=6388, checkpointing="selective", selective_stats={"mode": "mlp"}
    )
    with_vae = cosmos3_step_flops(
        model, samples=2, ar_tokens=1000, dm_tokens=6388, checkpointing="full", include_vae=True
    )

    assert full["flops_forward"] == pytest.approx(breakdown["forward"])
    assert full["flops_backward"] == pytest.approx(2 * breakdown["forward"] + breakdown["layer_forward"])
    assert full["flops"] == pytest.approx(full["flops_forward"] + full["flops_backward"])
    assert none["flops_backward"] == pytest.approx(2 * breakdown["forward"])
    assert none["flops"] < selective["flops"] < full["flops"]
    assert with_vae["flops_vae"] == pytest.approx(2 * 33.0e12)
    assert with_vae["flops"] == pytest.approx(full["flops"] + 2 * 33.0e12)

    # The old helper accepted num_layers and ignored it.  A larger checkpoint must change the scale.
    larger = Cosmos3FlopsModel.from_config(
        {**config, "num_hidden_layers": 36, "hidden_size": 4096, "intermediate_size": 16384}
    )
    larger_breakdown = larger.breakdown(ar_tokens=1000, dm_tokens=6388, samples=2)
    larger_flops = cosmos3_step_flops(larger, samples=2, ar_tokens=1000, dm_tokens=6388)
    assert larger_breakdown["mlp"] > 4 * breakdown["mlp"]
    # The exact total factor also depends on attention scores, the LM head and the AR/DM token mix; the geometry
    # itself is the part the old constant got wrong.
    assert larger_flops["flops"] > 2.5 * full["flops"]

    mfu = compute_mfu(step_ms=1000.0, total_flops=full["flops"], world_size=48, peak_tflops_per_npu=353.9)
    assert 0 < mfu < 1
    assert compute_mfu(step_ms=0, total_flops=full["flops"]) == 0.0
    assert compute_mfu(step_ms=1000.0, total_flops=full["flops"], world_size=0) == 0.0


def test_flops_model_requires_the_checkpoint_geometry():
    with pytest.raises(ValueError, match="num_hidden_layers"):
        Cosmos3FlopsModel.from_config({"hidden_size": 2048})


def test_anomaly_detector_reserved_and_device_memory():
    detector = AnomalyDetector(warmup=0, device_capacity_gib=64.0, memory_fraction=0.85)

    # Below 85% threshold (50 GiB < 54.4 GiB): no warning
    events_safe = detector.check(
        aggregate_records(
            [make_record(1, 0, mem={"allocated_gib": 20.0, "peak_reserved_gib": 45.0, "device_used_gib": 50.0})]
        )
    )
    assert not any(e["kind"] == "memory_pressure" for e in events_safe)

    # peak_reserved_gib exceeds 85% (58 GiB > 54.4 GiB)
    events_reserved = detector.check(
        aggregate_records([make_record(2, 0, mem={"allocated_gib": 20.0, "peak_reserved_gib": 58.0})])
    )
    assert any(e["kind"] == "memory_pressure" for e in events_reserved)

    # device_used_gib exceeds 85% (60 GiB > 54.4 GiB)
    events_device = detector.check(
        aggregate_records([make_record(3, 0, mem={"allocated_gib": 20.0, "device_used_gib": 60.0})])
    )
    assert any(e["kind"] == "memory_pressure" for e in events_device)
