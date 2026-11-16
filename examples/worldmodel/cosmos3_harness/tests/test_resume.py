"""CPU tests for replay weighting and complete collective resume (no model weights)."""

import json
import random
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from examples.worldmodel.cosmos3_harness import train
from examples.worldmodel.cosmos3_harness.checkpoint import (
    _rng_fields,
    apply_master_weights,
    flush_checkpoint,
    load_master_weights,
    master_weights_payload,
    restore_checkpoint,
    save_checkpoint,
    validate_checkpoint,
    validate_master_weights,
)
from examples.worldmodel.cosmos3_harness.loss import normalise_packed_loss, normalise_vision_token_mean
from examples.worldmodel.cosmos3_harness.optim import build_adamw


class Toy(torch.nn.Module):
    def __init__(self, stochastic=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.75))
        self.stochastic = stochastic

    def forward(self, vision_tokens, action_tokens=None, **kwargs):
        noise = torch.rand(()) + random.random() + float(np.random.rand()) if self.stochastic else 0.0
        return SimpleNamespace(
            sample=[self.weight * v + noise for v in vision_tokens],
            action=[self.weight * a + noise for a in (action_tokens or [])],
        )


class CPUBooster:
    """Exercise harness publication with actual DDP weights and AdamW state.

    ``async_=True`` stages the state dicts eagerly and writes them from real background threads
    registered on a ``checkpoint_io``, so the publication path runs exactly as in the GPU flows.
    """

    class _Writer:
        def __init__(self, write):
            self._error = None
            self._thread = threading.Thread(target=self._run, args=(write,), daemon=True)
            self._thread.start()

        def _run(self, write):
            try:
                write()
            except BaseException as exc:
                self._error = exc

        @property
        def done(self):
            return not self._thread.is_alive()

        def sync_before_step(self):
            return None

        def synchronize(self):
            self._thread.join()
            if self._error is not None:
                error, self._error = self._error, None
                raise RuntimeError("Background checkpoint write failed") from error

    class _CheckpointIO:
        def __init__(self):
            self.async_writers = []

        def _sync_io(self):
            writers, self.async_writers = self.async_writers, []
            for writer in writers:
                writer.synchronize()

    def __init__(self, async_=False):
        self.async_ = async_
        self.checkpoint_io = self._CheckpointIO()

    @staticmethod
    def _module(model):
        return model.module if hasattr(model, "module") else model

    def _write(self, write):
        if self.async_:
            self.checkpoint_io.async_writers.append(self._Writer(write))
        else:
            write()

    def save_model(self, model, path, **kwargs):
        state = self._module(model).state_dict() if not dist.is_initialized() or dist.get_rank() == 0 else None

        def write():
            if state is not None:
                Path(path).mkdir()
                torch.save(state, Path(path) / "weights.pt")

        self._write(write)

    def save_optimizer(self, optimizer, path, **kwargs):
        state = optimizer.state_dict() if not dist.is_initialized() or dist.get_rank() == 0 else None

        def write():
            if state is not None:
                Path(path).mkdir()
                torch.save(state, Path(path) / "state.pt")

        self._write(write)

    def load_model(self, model, path):
        self._module(model).load_state_dict(torch.load(Path(path) / "weights.pt", weights_only=True))

    def load_optimizer(self, optimizer, path):
        optimizer.load_state_dict(torch.load(Path(path) / "state.pt", weights_only=True))


def sample(seed, length=1, droid=False):
    value = train._synthetic_sample(2, 8 * length, seed)
    value["vision_tokens"] = [torch.ones(1, 4, 2, 4 * length, 4) * (seed + 1)]
    value["vision_token_shapes"] = [(2, 2 * length, 2)]
    value["vision_target"] = torch.zeros_like(value["vision_tokens"][0])
    if droid:
        value["targets"] = {
            "vision": value["vision_target"],
            "action": torch.zeros(3 * length, 4),
            "vision_noisy_mask": torch.tensor([0.0, 1.0]).view(2, 1, 1),
            "action_noisy_mask": torch.cat([torch.zeros(1, 1), torch.ones(3 * length - 1, 1)]),
            "raw_action_dim": 2,
        }
        # This test exercises the objective independently of the action pack layout.
        value["action_tokens"] = [torch.ones(3 * length, 4) * seed]
    return value


def test_replay_microsteps_cp_and_dp():
    packs = list(range(7))
    for mode in ("replicated", "sharded"):
        one, cursor = train.replay_window(packs, 0, 3, 0, 2, mode)
        two, other_cursor = train.replay_window(packs, 0, 3, 1, 2, mode)
        assert cursor == other_cursor == (6 if mode == "replicated" else 3)
        assert one == ([0, 2, 4] if mode == "replicated" else [0, 1, 2])
        assert two == ([1, 3, 5] if mode == "replicated" else one)
        # Two physical CP peers resolve to each DP coordinate, including replay wrap.
        cp_windows = [train.replay_window(packs, cursor, 3, rank // 2, 2, mode)[0] for rank in range(4)]
        assert cp_windows[0] == cp_windows[1] and cp_windows[2] == cp_windows[3]
        if mode == "replicated":
            assert all(a != b for a, b in zip(cp_windows[0], cp_windows[2]))
    with pytest.raises(ValueError, match="distinct"):
        train.replay_window([1], 0, 2, 0, 2, "replicated")


def test_unequal_length_and_error_objectives():
    model = Toy()
    samples = [sample(1, 1, True), sample(4, 3, True)]
    # Only the output/target objective is under test here; pack input geometry is separate.
    batch = SimpleNamespace(
        meta=None,
        model_inputs={
            "vision_tokens": [s["vision_tokens"][0] for s in samples],
            "action_tokens": [s["action_tokens"][0] for s in samples],
        },
    )
    with patch.object(train, "set_cosmos3_packed_meta"), patch.object(train, "_to_device", return_value=batch):
        loss = train._loss_for_batch(model, batch, samples, 1, objective="droid", global_count=2)
    # Full-size mask denominators, equal sample weights, padded channels excluded.
    expected = (
        sum(10 * ((0.75 * (s + 1)) ** 2 / 2 + (0.75 * s) ** 2 * (3 * l - 1) / (3 * l)) for s, l in ((1, 1), (4, 3))) / 2
    )
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert model.weight.grad.item() == pytest.approx(2 * expected / 0.75)
    # Generic token weighting is intentionally a different answer on unequal lengths/errors.
    for s in samples:
        s.pop("action_tokens")
    batch.model_inputs.pop("action_tokens")
    with patch.object(train, "set_cosmos3_packed_meta"), patch.object(train, "_to_device", return_value=batch):
        generic = train._loss_for_batch(model, batch, samples, 1)
    assert generic.item() == pytest.approx(0.75**2 * (4 + 3 * 25) / 4)
    assert normalise_packed_loss is normalise_vision_token_mean


def test_accumulation_matches_whole_window_token_mean():
    model = Toy()
    window = [[sample(1)], [sample(3, 3), sample(5, 2)]]
    count = train._window_count(window, "generic-vision-token-mean")
    losses = []
    for samples in window:
        batch = train._pack_replay_samples(samples)
        with patch.object(train, "set_cosmos3_packed_meta"):
            contribution = train._loss_for_batch(model, batch, samples, 1, global_count=count)
        contribution.backward()
        losses.append(contribution.detach())
    values = torch.cat([s["vision_tokens"][0].flatten() for samples in window for s in samples])
    expected = (0.75 * values).square().mean()
    assert sum(losses).item() == pytest.approx(expected.item())
    assert model.weight.grad.item() == pytest.approx((1.5 * values.square().mean()).item())


def test_objective_validation_and_cli():
    with pytest.raises(ValueError, match="complete"):
        train._window_count([[sample(1)]], "droid")
    with pytest.raises(ValueError, match="vision-only"):
        train._window_count([[sample(1, droid=True)]], "generic-vision-token-mean")
    args = train.parse_args(["--attn-backend", "naive", "--activation-checkpointing", "none"])
    assert args.backend == "naive" and args.checkpoint_mode == "none"
    with pytest.raises(SystemExit):
        train.parse_args(["--gradient-accumulation-steps", "0"])
    assert build_adamw(Toy().parameters(), 0.01, backend="npu").defaults["foreach"] is False
    with pytest.raises(ValueError, match="NPU"):
        build_adamw(Toy().parameters(), 0.01, backend="npu", mode="fused")


def test_policy_rejects_incompatible_device_before_loading(tmp_path):
    from examples.worldmodel.cosmos3_harness.robolab_policy import DroidWamPolicy

    with pytest.raises(ValueError, match="requires that device"):
        DroidWamPolicy.from_pretrained(tmp_path, transformer_path=None, config={}, device="cpu", backend="cuda")


class _MasterOptimizer:
    """Minimal mixed-precision optimizer surface used by the master-weight helpers."""

    def __init__(self, masters, workings):
        self.param_groups = [{"params": masters}]
        self._map = {id(master): working for master, working in zip(masters, workings)}

    def get_master_to_working_map(self):
        return self._map


def test_master_weights_round_trip_preserves_fp32(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.bfloat16))

    model = Model()
    master = torch.nn.Parameter(torch.tensor([1.0001, 2.0001], dtype=torch.float32))
    optimizer = _MasterOptimizer([master], [model.weight])

    payload = master_weights_payload(model, optimizer)
    assert payload is not None
    assert payload["keys"] == ["weight#0"]
    assert payload["tensors"][0].dtype == torch.float32
    path = tmp_path / "master-00000.pt"
    torch.save(payload, path)
    keys, tensors = load_master_weights(path)
    validate_master_weights(keys, tensors, model, optimizer)

    # This is what booster.load_model() does before the optimizer is restored: it rounds the master
    # parameters to the working bf16 values.  The saved fp32 payload must put the exact values back.
    master.data.copy_(model.weight.data)
    assert master[0].item() == 1.0
    apply_master_weights(tensors, model, optimizer)
    torch.testing.assert_close(master, torch.tensor([1.0001, 2.0001]), rtol=0, atol=0)

    with pytest.raises(ValueError, match="keys"):
        validate_master_weights(["other#0"], tensors, model, optimizer)


def test_checkpoint_integrity_and_atomic_publication(tmp_path):
    model = Toy()
    optimizer = torch.optim.AdamW(model.parameters())
    booster = CPUBooster()
    state = {"step": 1, "epoch": 0, "sampler_cursor": 2, "config": {"mode": "replicated"}}
    save_checkpoint(tmp_path, model, optimizer, state, booster=booster)
    root, saved = validate_checkpoint(tmp_path, expected_config=state["config"])
    assert saved["sampler_cursor"] == 2
    with pytest.raises(ValueError, match="configuration"):
        validate_checkpoint(tmp_path, expected_config={"mode": "sharded"})
    latest = (tmp_path / "latest.json").read_bytes()
    with patch.object(booster, "save_optimizer", side_effect=RuntimeError("interrupted")):
        with pytest.raises(RuntimeError, match="interrupted"):
            save_checkpoint(tmp_path, model, optimizer, state, booster=booster)
    assert (tmp_path / "latest.json").read_bytes() == latest
    validate_checkpoint(tmp_path)
    incomplete = next(tmp_path.glob(".incomplete-*"))
    with pytest.raises(ValueError, match="Incomplete"):
        validate_checkpoint(incomplete)
    (root / "rng-00000.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        restore_checkpoint(tmp_path, model, optimizer, booster=booster)


def _rewrite_rng(root, rank, mutate):
    from examples.worldmodel.cosmos3_harness.checkpoint import file_digest

    path = root / f"rng-{rank:05d}.pt"
    rng = torch.load(path, map_location="cpu", weights_only=False)
    mutate(rng)
    torch.save(rng, path)
    manifest = json.loads((root / "complete.json").read_text())
    manifest["files"][path.name] = file_digest(path)
    (root / "complete.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize(
    "field,value",
    [
        ("torch", torch.ones(7, dtype=torch.float32)),
        ("torch", torch.zeros(2, 2, dtype=torch.uint8)),
        ("torch", torch.zeros(1, dtype=torch.uint8)),
        ("python", (999, (), None)),
        ("numpy", ("MT19937", np.zeros(2, dtype=np.uint32), 0, 0, 0.0)),
        ("accelerator", "unknown"),
        ("missing", None),
    ],
)
def test_checksum_valid_but_invalid_rng_cannot_partially_restore(tmp_path, field, value):
    model, booster = Toy(), CPUBooster()
    optimizer = torch.optim.AdamW(model.parameters())
    save_checkpoint(
        tmp_path, model, optimizer, {"step": 1, "epoch": 0, "sampler_cursor": 1, "config": {}}, booster=booster
    )
    root, _ = validate_checkpoint(tmp_path)
    _rewrite_rng(root, 0, lambda rng: rng.pop("python") if field == "missing" else rng.update({field: value}))
    before = (torch.get_rng_state(), random.getstate(), np.random.get_state())
    with patch.object(booster, "load_model") as model_load, patch.object(booster, "load_optimizer") as optim_load:
        with pytest.raises(ValueError, match="RNG|rng"):
            restore_checkpoint(tmp_path, model, optimizer, booster=booster)
        model_load.assert_not_called()
        optim_load.assert_not_called()
    torch.testing.assert_close(torch.get_rng_state(), before[0], rtol=0, atol=0)
    assert random.getstate() == before[1]
    after_numpy = np.random.get_state()
    assert after_numpy[0] == before[2][0] and after_numpy[2:] == before[2][2:]
    np.testing.assert_array_equal(after_numpy[1], before[2][1])


def test_rng_schema_is_shared_with_the_writer(tmp_path):
    assert set(_rng_fields("cpu")) == {"torch", "python", "numpy", "accelerator"}
    assert set(_rng_fields("npu")) == {"torch", "python", "numpy", "accelerator", "device"}
    model, booster = Toy(), CPUBooster()
    optimizer = torch.optim.AdamW(model.parameters())
    save_checkpoint(
        tmp_path, model, optimizer, {"step": 1, "epoch": 0, "sampler_cursor": 1, "config": {}}, booster=booster
    )
    root, _ = validate_checkpoint(tmp_path)
    rng = torch.load(root / "rng-00000.pt", map_location="cpu", weights_only=False)
    # The writer's fields are a subset of the schema the reader allows; ``extra`` is the optional rank state.
    assert set(rng) <= set(_rng_fields(str(rng["accelerator"]))) | {"extra"}


def test_an_unexpected_rng_field_is_rejected_before_model_load(tmp_path):
    model, booster = Toy(), CPUBooster()
    optimizer = torch.optim.AdamW(model.parameters())
    save_checkpoint(
        tmp_path, model, optimizer, {"step": 1, "epoch": 0, "sampler_cursor": 1, "config": {}}, booster=booster
    )
    root, _ = validate_checkpoint(tmp_path)
    _rewrite_rng(root, 0, lambda rng: rng.update({"future": "unexpected"}))
    with patch.object(booster, "load_model") as model_load, patch.object(booster, "load_optimizer") as optim_load:
        with pytest.raises(ValueError, match="unexpected RNG fields"):
            restore_checkpoint(tmp_path, model, optimizer, booster=booster)
        model_load.assert_not_called()
        optim_load.assert_not_called()


@pytest.mark.parametrize("case", ["missing", "count", "shape", "type"])
def test_device_rng_preflight_before_model_load(tmp_path, case):
    model, booster = Toy(), CPUBooster()
    optimizer = torch.optim.AdamW(model.parameters())
    save_checkpoint(
        tmp_path, model, optimizer, {"step": 1, "epoch": 0, "sampler_cursor": 1, "config": {}}, booster=booster
    )
    root, _ = validate_checkpoint(tmp_path)
    streams = [torch.zeros(8, dtype=torch.uint8)]
    if case == "shape":
        streams = [torch.zeros(7, dtype=torch.uint8), torch.zeros(8, dtype=torch.uint8)]
    elif case == "type":
        streams = [torch.zeros(8), torch.zeros(8)]

    def mutate(rng):
        rng["accelerator"] = "npu"
        if case != "missing":
            rng["device"] = streams

    _rewrite_rng(root, 0, mutate)
    fake = SimpleNamespace(
        name="npu", device_count=lambda: 2, get_rng_state_all=lambda: [torch.zeros(8, dtype=torch.uint8)] * 2
    )
    with (
        patch("examples.worldmodel.cosmos3_harness.checkpoint._accelerator", return_value=fake),
        patch.object(booster, "load_model") as model_load,
        patch.object(booster, "load_optimizer") as optim_load,
    ):
        with pytest.raises(ValueError, match="[Dd]evice RNG"):
            restore_checkpoint(tmp_path, model, optimizer, booster=booster)
        model_load.assert_not_called()
        optim_load.assert_not_called()


def _worker(rank, world, rendezvous, directory):
    from colossalai.accelerator import set_accelerator

    set_accelerator("cpu")
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world)
    try:
        _exercise_window_gradient(rank)
        for mode in ("replicated", "sharded"):
            _exercise_resume(rank, str(Path(directory) / mode), mode)
            _exercise_resume(rank, str(Path(directory) / f"{mode}-async"), mode, async_save=True)
        _exercise_flush_queue(rank, str(Path(directory) / "flush-queue"))
        _exercise_async_publication(rank, str(Path(directory) / "publication"))
        _exercise_failed_publication(rank, str(Path(directory) / "publication-failure"))
    finally:
        dist.destroy_process_group()


def _exercise_window_gradient(rank):
    model = torch.nn.parallel.DistributedDataParallel(Toy())
    # Different sample counts, element counts, and errors on DP peers and microsteps.
    window = [[sample(rank + 1, rank + 1)], [sample(rank + 3, 3)] * (rank + 1)]
    count = train._global_count(train._window_count(window, "generic-vision-token-mean"), torch.device("cpu"))
    square_sum = sum(s["vision_tokens"][0].square().sum() for samples in window for s in samples)
    dist.all_reduce(square_sum)
    expected = 0.75**2 * square_sum / count
    total = torch.zeros(())
    for samples in window:
        batch = train._pack_replay_samples(samples)
        with patch.object(train, "set_cosmos3_packed_meta"):
            loss = train._loss_for_batch(model, batch, samples, 2, global_count=count)
        loss.backward()
        total += loss.detach()
    dist.all_reduce(total)
    torch.testing.assert_close(total / 2, expected)
    torch.testing.assert_close(model.module.weight.grad, 2 * expected / 0.75)


def _exercise_resume(rank, directory, mode, async_save=False):
    def seed():
        torch.manual_seed(111 + rank)
        random.seed(222 + rank)
        np.random.seed(333 + rank)

    offset = rank * 13 if mode == "sharded" else 0
    packs = [[sample(i + 1 + offset, i % 3 + 1)] * (1 + i % 2) for i in range(8)]

    def run(model, optimizer, cursor, start, end):
        history = []
        for step in range(start, end):
            window, cursor = train.replay_window(packs, cursor, 3, rank, 2, mode)
            count = train._global_count(train._window_count(window, "generic-vision-token-mean"), torch.device("cpu"))
            optimizer.zero_grad()
            total = torch.zeros(())
            for samples in window:
                batch = train._pack_replay_samples(samples)
                with patch.object(train, "set_cosmos3_packed_meta"):
                    loss = train._loss_for_batch(model, batch, samples, 2, global_count=count)
                loss.backward()
                total += loss.detach()
            optimizer.step()
            dist.all_reduce(total)
            history.append(total.item() / 2)
        return cursor, history

    seed()
    full = torch.nn.parallel.DistributedDataParallel(Toy(stochastic=True))
    opt_full = torch.optim.AdamW(full.parameters(), lr=0.01)
    cursor_full, losses_full = run(full, opt_full, 0, 0, 4)
    expected_rng = (torch.rand(3), random.random(), np.random.rand())
    seed()
    split = torch.nn.parallel.DistributedDataParallel(Toy(stochastic=True))
    opt_split = torch.optim.AdamW(split.parameters(), lr=0.01)
    cursor, before = run(split, opt_split, 0, 0, 2)
    booster = CPUBooster(async_=async_save)
    pending = save_checkpoint(
        directory,
        split,
        opt_split,
        {"step": 2, "epoch": cursor // len(packs), "sampler_cursor": cursor, "config": {"cp": 1}},
        booster=booster,
        async_=async_save,
    )
    if async_save:
        # The generation is only visible once every rank confirms it.
        flush_checkpoint(pending)
        assert pending.generation.is_dir()
        assert json.loads((Path(directory) / "latest.json").read_text())["generation"] == pending.generation.name
    # Model/optimizer construction and unrelated work consume RNG before restoration.
    torch.rand(17)
    random.random()
    np.random.rand()
    resumed = torch.nn.parallel.DistributedDataParallel(Toy(stochastic=True))
    opt_resumed = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    state = restore_checkpoint(directory, resumed, opt_resumed, booster=booster, expected_config={"cp": 1})
    cursor_resumed, after = run(resumed, opt_resumed, state["sampler_cursor"], state["step"], 4)
    assert cursor_full == cursor_resumed == (24 if mode == "replicated" else 12)
    assert before + after == losses_full
    torch.testing.assert_close(full.module.weight, resumed.module.weight, rtol=0, atol=0)
    for key, expected in opt_full.state[full.module.weight].items():
        torch.testing.assert_close(expected, opt_resumed.state[resumed.module.weight][key], rtol=0, atol=0)
    actual_rng = (torch.rand(3), random.random(), np.random.rand())
    torch.testing.assert_close(expected_rng[0], actual_rng[0], rtol=0, atol=0)
    assert expected_rng[1:] == actual_rng[1:]
    # Rank streams are distinct, each stored separately, shared state has exactly one file.
    root, _ = validate_checkpoint(directory)
    assert len(list(root.glob("rng-*.pt"))) == 2
    assert len(list(root.glob("training_state.pt"))) == 1
    rng = [None, None]
    dist.all_gather_object(rng, actual_rng[0].tolist())
    assert rng[0] != rng[1]
    from examples.worldmodel.cosmos3_harness import checkpoint

    # Only rank 1 has an incompatible local runtime; rank 0 must also stop before loading.
    local_accelerator = SimpleNamespace(name="cuda") if rank == 1 else checkpoint._accelerator()
    with (
        patch.object(checkpoint, "_accelerator", return_value=local_accelerator),
        patch.object(booster, "load_model") as model_load,
        patch.object(booster, "load_optimizer") as optim_load,
    ):
        with pytest.raises(ValueError, match="rank 1") as error:
            restore_checkpoint(directory, resumed, opt_resumed, booster=booster)
        model_load.assert_not_called()
        optim_load.assert_not_called()
    messages = [None, None]
    dist.all_gather_object(messages, str(error.value))
    assert messages[0] == messages[1]

    # The hashes are valid, but a saved peer's stream is not restorable.
    if rank == 0:
        _rewrite_rng(root, 1, lambda saved: saved.update(python=(999, (), None)))
    dist.barrier()
    with patch.object(booster, "load_model") as model_load, patch.object(booster, "load_optimizer") as optim_load:
        with pytest.raises(ValueError, match="rng-00001.pt") as error:
            restore_checkpoint(directory, resumed, opt_resumed, booster=booster)
        model_load.assert_not_called()
        optim_load.assert_not_called()
    dist.all_gather_object(messages, str(error.value))
    assert messages[0] == messages[1]


def _exercise_flush_queue(rank, directory):
    """Every rank queues the same number of generations, so the flush collective cannot diverge.

    The ledger exists on rank zero only.  Keying the queue on the ledger id left the other ranks
    with an empty list and made them skip the collective the publisher's rank was waiting in.
    """
    from examples.worldmodel.cosmos3_harness import train
    from examples.worldmodel.cosmos3_harness.checkpoint_ledger import CheckpointLedger

    model = torch.nn.parallel.DistributedDataParallel(Toy())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    booster = CPUBooster(async_=True)
    model._colossal_booster = booster
    ledger = CheckpointLedger(directory, rank=rank, world_size=2) if rank == 0 else None
    stats = {"count": 0, "bytes": 0}
    pending = []
    try:
        train._save_with_ledger(
            pending,
            ledger,
            stats,
            Path(directory),
            model,
            optimizer,
            1,
            0,
            0,
            {"cp": 1},
            1,
            async_=True,
        )
        assert len(pending) == 1
        train._flush_pending(pending, ledger, stats)
        assert pending == []
    finally:
        if ledger is not None:
            ledger.close()
    if rank == 0:
        assert stats["count"] == 1
        events = [json.loads(line) for line in (Path(directory) / "checkpoints.jsonl").read_text().splitlines()]
        assert events[-1]["event"] == "end" and events[-1]["status"] == "complete"
    dist.barrier()
    assert (Path(directory) / "latest.json").is_file()


def _exercise_async_publication(rank, directory):
    """An asynchronous generation stays invisible until every rank flushes it."""
    from examples.worldmodel.cosmos3_harness import checkpoint

    model = torch.nn.parallel.DistributedDataParallel(Toy())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    state = {"step": 1, "epoch": 0, "sampler_cursor": 0, "config": {"cp": 1}}
    booster = CPUBooster(async_=True)
    entered, release = threading.Event(), threading.Event()
    original = checkpoint.file_digest

    def gated(path):
        entered.set()
        release.wait(timeout=60)
        return original(path)

    if rank == 0:
        # Park the publisher inside the manifest walk: the rename and the pointer have not
        # happened, so neither the generation nor latest.json may be visible.
        with patch.object(checkpoint, "file_digest", side_effect=gated):
            pending = save_checkpoint(directory, model, optimizer, state, booster=booster, async_=True)
            assert entered.wait(timeout=60)
            assert not (Path(directory) / "latest.json").exists()
            assert not pending.generation.exists()
            release.set()
            flush_checkpoint(pending)
    else:
        pending = save_checkpoint(directory, model, optimizer, state, booster=booster, async_=True)
        flush_checkpoint(pending)
    dist.barrier()
    assert pending.generation.is_dir()
    root, saved = validate_checkpoint(directory)
    assert saved["step"] == 1
    assert json.loads((Path(directory) / "latest.json").read_text())["generation"] == pending.generation.name


def _exercise_failed_publication(rank, directory):
    """A failed publisher makes every rank raise together and leaves the pointer untouched."""
    from examples.worldmodel.cosmos3_harness import checkpoint

    model = torch.nn.parallel.DistributedDataParallel(Toy())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    state = {"step": 1, "epoch": 0, "sampler_cursor": 0, "config": None}
    booster = CPUBooster(async_=True)
    if rank == 0:
        with patch.object(checkpoint, "file_digest", side_effect=OSError("disk gone")):
            pending = save_checkpoint(directory, model, optimizer, state, booster=booster, async_=True)
            with pytest.raises(RuntimeError, match="rank 0"):
                flush_checkpoint(pending)
    else:
        pending = save_checkpoint(directory, model, optimizer, state, booster=booster, async_=True)
        with pytest.raises(RuntimeError, match="rank 0"):
            flush_checkpoint(pending)
    assert not (Path(directory) / "latest.json").exists()
    assert not pending.generation.exists()


def _node_local_resume(rank, base):
    """Each rank's ``--out`` is its own disk: rank 0 publishes, rank 1 keeps only the RNG it wrote.

    This is the six-node layout.  Rank 1 has no ``latest.json``, no weights and an unpublished staging
    directory; resume must still find the generation, fetch the shared files from rank 0's node and leave
    a complete local copy behind.
    """
    root = Path(base) / f"node{rank}" / "checkpoints"
    model = torch.nn.parallel.DistributedDataParallel(Toy())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    booster = CPUBooster()
    model._colossal_booster = booster
    pending = save_checkpoint(
        root,
        model,
        optimizer,
        {"step": 5, "epoch": 1, "sampler_cursor": 7, "config": {"cp": 1}},
        booster=booster,
    )
    dist.barrier()
    if rank == 0:
        manifest = json.loads((pending.generation / "complete.json").read_text())
        # Rank files stay on the node that wrote them, but the manifest still vouches for all of them.
        assert "rng-00001.pt" in manifest["files"]
        assert "rng-00001.pt" in manifest["sizes"]
        assert (pending.generation / "model" / "weights.pt").is_file()
    else:
        assert not pending.generation.exists()
        assert not (root / "latest.json").exists()
        assert (root / (".incomplete-" + pending.identifier) / "rng-00001.pt").is_file()
    dist.barrier()

    resumed = torch.nn.parallel.DistributedDataParallel(Toy())
    optimizer_resumed = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    state = restore_checkpoint(root, resumed, optimizer_resumed, booster=booster, expected_config={"cp": 1})
    assert state["step"] == 5
    # Both nodes now hold a published, complete generation for the next resume.
    assert (root / "latest.json").is_file()
    generation = root / pending.generation.name
    assert (generation / "complete.json").is_file()
    assert (generation / "model" / "weights.pt").is_file()
    # Each node holds its own rank file and only its own.
    assert (generation / f"rng-{rank:05d}.pt").is_file()
    assert not (generation / f"rng-{1 - rank:05d}.pt").exists()
    torch.testing.assert_close(resumed.module.weight, model.module.weight, rtol=0, atol=0)


def _node_local_worker(rank, world, rendezvous, base):
    from colossalai.accelerator import set_accelerator

    set_accelerator("cpu")
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world)
    try:
        _node_local_resume(rank, base)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo required")
def test_node_local_disks_fetch_the_generation_from_rank_zero(tmp_path):
    mp.spawn(_node_local_worker, args=(2, str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo required")
def test_two_rank_uninterrupted_equals_resume(tmp_path):
    mp.spawn(_worker, args=(2, str(tmp_path / "rendezvous"), str(tmp_path / "checkpoint")), nprocs=2, join=True)
