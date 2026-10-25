import json
import math
import pickle

import pytest
import torch
from packaging.version import Version
from safetensors.torch import save_file

from colossalai.checkpoint_io import GeneralCheckpointIO
from colossalai.checkpoint_io.index_file import CheckpointIndexFile
from colossalai.checkpoint_io.optimizer_migration import migrate_optimizer_checkpoint
from colossalai.checkpoint_io.utils import (
    load_optimizer_state,
    load_param_groups,
    load_state_dict_shards,
    prepare_pinned_state_dict,
)
from colossalai.utils import _optimizer_safetensors as schema
from colossalai.utils._optimizer_safetensors import LegacyOptimizerCheckpointError
from colossalai.utils.safetensors import (
    HAS_TENSORNVME,
    _flatten_optim_state_dict,
    load_flat,
    prepare,
    save_nested,
    save_optimizer_file,
)


def assert_equal(a, b):
    assert type(a) is type(b)
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    elif isinstance(a, float) and math.isnan(a):
        assert math.isnan(b)
    else:
        assert a == b


def adam():
    params = [torch.nn.Parameter(torch.ones(2, 2)), torch.nn.Parameter(torch.ones(2))]
    opt = torch.optim.Adam(params)
    for param in params:
        param.grad = torch.ones_like(param) * 0.1
    opt.step()
    return opt


def write_v2(path, value):
    flat, metadata = _flatten_optim_state_dict(value)
    save_file(flat, str(path), metadata=metadata)


@pytest.mark.parametrize("kind", ["full", "state_shard", "param_groups"])
def test_typed_roundtrip(tmp_path, kind):
    state = adam().state_dict()
    state["state"][0]["nested"] = [
        None,
        True,
        4,
        "text",
        (0.5, torch.arange(3)),
        {1: -0.0},
    ]
    state["state"][0]["empty"] = torch.empty(0, dtype=torch.bfloat16)
    state["param_groups"][0]["lr"] = torch.tensor(0.001)
    state["param_groups"][0]["special"] = (float("inf"), float("-inf"), float("nan"))
    value = (
        state
        if kind == "full"
        else (state["state"] if kind == "state_shard" else {"param_groups": state["param_groups"]})
    )
    path = tmp_path / "optimizer.safetensors"
    write_v2(path, value)
    assert_equal(value, load_flat(path, expected_kind=kind))


@pytest.mark.parametrize(
    "name",
    [
        "ASGD",
        "Adadelta",
        "Adafactor",
        "Adagrad",
        "Adam",
        "AdamW",
        "Adamax",
        "LBFGS",
        "Muon",
        "NAdam",
        "RAdam",
        "RMSprop",
        "Rprop",
        "SGD",
        "SparseAdam",
        "colossalai.CAME",
        "colossalai.Lamb",
        "colossalai.Lars",
        "colossalai.Adafactor",
    ],
)
def test_optimizer_next_step(tmp_path, name):
    p = torch.nn.Parameter(torch.ones(2, 2))
    if name.startswith("colossalai."):
        import colossalai.nn.optimizer as optimizers

        cls = getattr(optimizers, name.split(".")[1])
    else:
        cls = getattr(torch.optim, name, None)
        if cls is None:
            pytest.skip(f"{name} is unavailable in this torch version")
    kwargs = {"lr": 0.01} if name in ("Muon", "colossalai.CAME") else {}
    opt = cls([p], **kwargs)

    def step(optimizer, param):
        if name == "LBFGS":

            def closure():
                optimizer.zero_grad()
                loss = param.square().sum()
                loss.backward()
                return loss

            optimizer.step(closure)
        else:
            param.grad = torch.full_like(param, 0.1)
            if name == "SparseAdam":
                param.grad = param.grad.to_sparse()
            optimizer.step()

    step(opt, p)
    q = torch.nn.Parameter(p.detach().clone())
    restored = cls([q], **kwargs)
    path = tmp_path / "optimizer.safetensors"
    save_optimizer_file(path, opt.state_dict())
    GeneralCheckpointIO().load_optimizer(restored, str(path))
    assert_equal(opt.state_dict(), restored.state_dict())
    step(opt, p)
    step(restored, q)
    assert torch.equal(p, q)
    assert_equal(opt.state_dict(), restored.state_dict())


def test_prepare_does_not_double_encode_or_mutate():
    tensors, metadata = _flatten_optim_state_dict(adam().state_dict())
    original = dict(metadata)
    prepared, _, _ = prepare(tensors, metadata)
    assert metadata == original
    header = json.loads(prepared.header_bytes)
    assert header["__metadata__"][schema.FORMAT_KEY] == "2"
    assert header["__metadata__"][schema.SCHEMA_KEY] == original[schema.SCHEMA_KEY]


@pytest.mark.parametrize("location", ["state.0.candidate", "param_groups"])
def test_legacy_payload_rejected_before_execution(tmp_path, location):
    marker = tmp_path / "must_not_exist"

    class Payload:
        def __reduce__(self):
            return eval, (f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')",)

    tensors = {location: torch.tensor(list(pickle.dumps(Payload(), protocol=2)), dtype=torch.uint8)}
    path = tmp_path / "legacy.safetensors"
    save_file(tensors, str(path), metadata={"non_tensor_keys": json.dumps([location])})
    for loader in (
        load_flat,
        lambda file: GeneralCheckpointIO().load_optimizer(adam(), str(file)),
    ):
        with pytest.raises(LegacyOptimizerCheckpointError, match="migrat|Migrate"):
            loader(path)
    assert not marker.exists()


def test_torch_optimizer_load_is_explicitly_restricted(tmp_path):
    marker = tmp_path / "must_not_exist"

    class Payload:
        def __reduce__(self):
            return eval, (f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')",)

    path = tmp_path / "optimizer.bin"
    torch.save(Payload(), path)
    error = RuntimeError if Version(torch.__version__) < Version("2.10.0") else pickle.UnpicklingError
    with pytest.raises(error):
        load_optimizer_state(path)
    assert not marker.exists()


def optimizer_file_case(kind):
    from colossalai.checkpoint_io.utils import load_optim_or_model_shard

    state = adam().state_dict()
    if kind == "full":
        return state, load_optimizer_state
    if kind == "param_groups":
        return state["param_groups"], load_param_groups
    return state["state"], lambda path: load_optim_or_model_shard(str(path), True, False)


@pytest.mark.parametrize("kind", ["full", "param_groups", "state_shard"])
@pytest.mark.parametrize("version", ["2.5.1", "2.9.1", "2.10.0rc1"])
def test_legacy_torch_version_rejected_before_deserialization(tmp_path, monkeypatch, kind, version):
    payload, reader = optimizer_file_case(kind)
    path = tmp_path / "legacy.bin"
    torch.save(payload, path)

    def must_not_load(*args, **kwargs):
        pytest.fail("Unsupported torch must be rejected before torch.load")

    monkeypatch.setattr(torch, "__version__", version)
    monkeypatch.setattr(torch, "load", must_not_load)
    with pytest.raises(RuntimeError, match="requires torch >= 2.10.0"):
        reader(path)


@pytest.mark.parametrize("kind", ["full", "param_groups", "state_shard"])
@pytest.mark.parametrize("version", ["2.10.0", "2.11.0+cu128"])
def test_legacy_torch_version_policy_accepts_patched_versions(tmp_path, monkeypatch, kind, version):
    # This checks version dispatch on the installed torch, not compatibility
    # with an actual installation of each version string.
    payload, reader = optimizer_file_case(kind)
    path = tmp_path / "legacy.bin"
    torch.save(payload, path)
    monkeypatch.setattr(torch, "__version__", version)
    assert_equal(payload, reader(path))


@pytest.mark.parametrize("kind", ["full", "param_groups", "state_shard"])
def test_v2_does_not_use_legacy_torch_version_gate(tmp_path, monkeypatch, kind):
    payload, reader = optimizer_file_case(kind)
    path = tmp_path / "optimizer.safetensors"
    save_optimizer_file(path, {"param_groups": payload} if kind == "param_groups" else payload)
    monkeypatch.setattr(torch, "__version__", "2.5.1")
    assert_equal(payload, reader(path))


def test_legacy_tensor_only_shard(tmp_path):
    path = tmp_path / "legacy.safetensors"
    save_file({"state.0.step": torch.tensor(1.0)}, str(path))
    assert_equal({0: {"step": torch.tensor(1.0)}}, load_flat(path, expected_kind="state_shard"))
    with pytest.raises(LegacyOptimizerCheckpointError):
        load_flat(path, expected_kind="full")


@pytest.mark.parametrize(
    "mutation",
    [
        "version",
        "kind",
        "legacy",
        "unknown",
        "duplicate_json",
        "missing_tensor",
        "extra_tensor",
        "shape",
        "dtype",
        "bool_id",
    ],
)
def test_invalid_schema_rejected(tmp_path, mutation):
    tensors, metadata = _flatten_optim_state_dict(adam().state_dict())
    tree = json.loads(metadata[schema.SCHEMA_KEY])

    def nodes(value):
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from nodes(child)
        elif isinstance(value, list):
            for child in value:
                yield from nodes(child)

    if mutation == "version":
        metadata[schema.FORMAT_KEY] = "3"
    elif mutation == "kind":
        metadata["kind"] = "param_groups"
    elif mutation == "legacy":
        metadata["non_tensor_keys"] = "[]"
    elif mutation == "unknown":
        tree["type"] = "python_class"
    elif mutation == "duplicate_json":
        metadata[schema.SCHEMA_KEY] = '{"type":"null","type":"dict"}'
    elif mutation == "missing_tensor":
        tensors.pop(next(iter(tensors)))
    elif mutation == "extra_tensor":
        tensors["unreferenced"] = torch.ones(1)
    elif mutation == "shape":
        next(n for n in nodes(tree) if n.get("type") == "tensor_ref")["shape"] = [100]
    elif mutation == "dtype":
        next(n for n in nodes(tree) if n.get("type") == "tensor_ref")["dtype"] = "torch.int64"
    elif mutation == "bool_id":
        n = next(n for n in nodes(tree) if n.get("type") == "int")
        n["value"] = True
    if mutation != "duplicate_json":
        metadata[schema.SCHEMA_KEY] = json.dumps(tree)
    path = tmp_path / "invalid.safetensors"
    save_file(tensors, str(path), metadata=metadata)
    with pytest.raises(ValueError):
        load_flat(path, expected_kind="full")


def test_reader_schema_byte_boundary_before_tensor_fetch(tmp_path, monkeypatch):
    from colossalai.utils.safetensors import _FileTensors

    state = adam().state_dict()
    state["state"][0]["note"] = "中文"
    tensors, metadata = _flatten_optim_state_dict(state)
    metadata[schema.SCHEMA_KEY] = json.dumps(json.loads(metadata[schema.SCHEMA_KEY]), ensure_ascii=False)
    size = len(metadata[schema.SCHEMA_KEY].encode("utf-8"))
    assert size > len(metadata[schema.SCHEMA_KEY])
    path = tmp_path / "byte_boundary.safetensors"
    save_file(tensors, str(path), metadata=metadata)
    monkeypatch.setattr(schema, "MAX_SCHEMA_BYTES", size)
    assert_equal(state, load_flat(path, expected_kind="full"))

    def must_not_fetch(self, key):
        pytest.fail("Oversized metadata must be rejected before fetching tensors")

    monkeypatch.setattr(schema, "MAX_SCHEMA_BYTES", size - 1)
    monkeypatch.setattr(_FileTensors, "__getitem__", must_not_fetch)
    with pytest.raises(ValueError, match="oversized"):
        load_flat(path, expected_kind="full")


@pytest.mark.parametrize("limit,boundary", [("MAX_DEPTH", 6), ("MAX_NODES", 12)])
def test_reader_container_budget_boundary(tmp_path, monkeypatch, limit, boundary):
    # Root -> groups list -> group dict -> three nested lists -> int: depth 6.
    # Including dictionary keys and params=[0], this payload has 12 typed nodes.
    state = {"param_groups": [{"params": [0], "extra": [[[1]]]}]}
    path = tmp_path / "container_boundary.safetensors"
    save_optimizer_file(path, state)
    monkeypatch.setattr(schema, limit, boundary)
    assert_equal(state, load_flat(path, expected_kind="param_groups"))
    monkeypatch.setattr(schema, limit, boundary - 1)
    with pytest.raises(ValueError, match="depth or node limit"):
        load_flat(path, expected_kind="param_groups")


def test_budget_and_unsupported_writer_inputs(monkeypatch):
    state = adam().state_dict()
    state["state"][0]["object"] = object()
    with pytest.raises(TypeError, match="Unsupported"):
        _flatten_optim_state_dict(state)
    state["state"][0].pop("object")
    state["state"][0]["cycle"] = state
    with pytest.raises(ValueError, match="Cyclic"):
        _flatten_optim_state_dict(state)
    state["state"][0].pop("cycle")
    monkeypatch.setattr(schema, "MAX_NODES", 2)
    with pytest.raises(ValueError, match="node limit"):
        _flatten_optim_state_dict(state)


@pytest.mark.parametrize("prefetch", [False, True])
def test_shards_groups_and_prefetch(tmp_path, prefetch):
    state = adam().state_dict()
    paths = []
    keys = []
    for param_id, values in state["state"].items():
        path = tmp_path / f"{param_id}.safetensors"
        shard = {param_id: values}
        flat, _ = _flatten_optim_state_dict(shard)
        keys.append(set(flat))
        write_v2(path, shard)
        paths.append(str(path))
    assert keys[0].isdisjoint(keys[1])
    loaded = {}
    for shard in load_state_dict_shards(paths, True, True, low_cpu_mem_mode=not prefetch):
        loaded.update(shard)
    assert_equal(state["state"], loaded)
    groups = tmp_path / "groups.safetensors"
    save_optimizer_file(groups, {"param_groups": state["param_groups"]})
    assert_equal(state["param_groups"], load_param_groups(groups))


def test_pinned_cache_accepts_new_keys_shapes_and_dtypes(monkeypatch):
    from colossalai.checkpoint_io import utils

    # Buffer selection is a CPU unit; real pinned transfers run in GPU integration.
    monkeypatch.setattr(utils, "_pin_tensor", lambda tensor: torch.empty_like(tensor))
    cache = {}
    first = prepare_pinned_state_dict({"a": torch.ones(1)}, cache)
    second = prepare_pinned_state_dict({"a": torch.ones(1), "b": torch.ones(2)}, cache)
    assert first["a"] is second["a"]
    third = prepare_pinned_state_dict({"a": torch.ones(3, dtype=torch.int64)}, cache)
    assert third["a"].shape == (3,) and third["a"].dtype == torch.int64


@pytest.mark.parametrize("escape", ["../outside.bin", "/outside.bin"])
def test_index_path_escape_rejected(tmp_path, escape):
    path = tmp_path / "optimizer.index.json"
    path.write_text(json.dumps({"metadata": {}, "weight_map": {"0": escape}}))
    with pytest.raises(ValueError, match="root|relative"):
        CheckpointIndexFile.from_file(path, is_optimizer=True)


def test_model_snapshot_links_to_external_blobs(tmp_path):
    snapshot = tmp_path / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    expected = torch.nn.Linear(2, 2).state_dict()
    blob = blobs / "opaque_hash"
    save_file(expected, str(blob))
    (snapshot / "model.safetensors").symlink_to(blob)
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {name: "model.safetensors" for name in expected},
            }
        )
    )
    restored = torch.nn.Linear(2, 2)
    GeneralCheckpointIO().load_model(restored, str(snapshot), strict=True)
    assert_equal(expected, restored.state_dict())


@pytest.mark.parametrize("location", ["param_groups", "weight_map"])
@pytest.mark.parametrize("reference", ["absolute", "parent", "symlink"])
def test_optimizer_api_rejects_external_index_paths(tmp_path, location, reference):
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"never decoded")
    if reference == "absolute":
        name = str(outside)
    elif reference == "parent":
        name = "../outside.safetensors"
    else:
        (directory / "linked.safetensors").symlink_to(outside)
        name = "linked.safetensors"
    metadata = {"param_groups": name if location == "param_groups" else "groups.safetensors"}
    weights = {"0": name} if location == "weight_map" else {}
    (directory / "optimizer.index.json").write_text(json.dumps({"metadata": metadata, "weight_map": weights}))
    with pytest.raises(ValueError, match="relative|root"):
        GeneralCheckpointIO().load_optimizer(adam(), str(directory))


def test_optimizer_index_preserves_internal_link_names(tmp_path, cpu_sync):
    expected = adam().state_dict()
    save_optimizer_file(tmp_path / "groups_blob", {"param_groups": expected["param_groups"]})
    save_optimizer_file(tmp_path / "states_blob", expected["state"])
    (tmp_path / "groups.safetensors").symlink_to(tmp_path / "groups_blob")
    (tmp_path / "states.safetensors").symlink_to(tmp_path / "states_blob")
    index = tmp_path / "optimizer.index.json"
    index.write_text(
        json.dumps(
            {
                "metadata": {"param_groups": "groups.safetensors"},
                "weight_map": {str(key): "states.safetensors" for key in expected["state"]},
            }
        )
    )
    loaded_index = CheckpointIndexFile.from_file(index)
    assert loaded_index.get_param_group_filename().endswith("groups.safetensors")
    assert loaded_index.get_checkpoint_filenames()[0][0].endswith("states.safetensors")
    restored = adam()
    GeneralCheckpointIO().load_optimizer(restored, str(tmp_path))
    assert_equal(expected, restored.state_dict())


@pytest.fixture
def cpu_sync(monkeypatch):
    from colossalai.accelerator import get_accelerator

    accelerator = get_accelerator()
    if accelerator.name == "cpu":
        monkeypatch.setattr(accelerator, "synchronize", lambda: None)


@pytest.mark.skipif(
    Version(torch.__version__) < Version("2.10.0"), reason="Legacy optimizer loading requires torch >= 2.10"
)
def test_general_sync_checkpoint_next_step(tmp_path, cpu_sync):
    original = adam()
    restored = adam()
    io = GeneralCheckpointIO()
    io.save_optimizer(original, str(tmp_path), shard=True, size_per_shard=1)
    io.load_optimizer(restored, str(tmp_path))
    assert_equal(original.state_dict(), restored.state_dict())


def test_migration_requires_explicit_trust_and_preserves_input(tmp_path):
    state = adam().state_dict()
    flat = {f"state.{pid}.{key}": tensor for pid, values in state["state"].items() for key, tensor in values.items()}
    flat["param_groups"] = torch.tensor(list(pickle.dumps(state["param_groups"])), dtype=torch.uint8)
    source = tmp_path / "legacy.safetensors"
    output = tmp_path / "new.safetensors"
    save_file(flat, str(source), metadata={"non_tensor_keys": '["param_groups"]'})
    original_bytes = source.read_bytes()
    with pytest.raises(ValueError, match="Explicit"):
        migrate_optimizer_checkpoint(source, output)
    assert not output.exists()
    report = migrate_optimizer_checkpoint(source, output, trust_legacy_pickle=True)
    assert report["format_version"] == 2 and source.name in report["inputs"]
    assert output.name in report["outputs"]
    assert_equal(state, load_flat(output))
    assert source.read_bytes() == original_bytes
    with pytest.raises(FileExistsError):
        migrate_optimizer_checkpoint(source, output, trust_legacy_pickle=True)


def test_sharded_migration(tmp_path, cpu_sync):
    state = adam().state_dict()
    source = tmp_path / "old"
    source.mkdir()
    torch.save(state["param_groups"], source / "groups.bin")
    torch.save(state["state"], source / "states.bin")
    (source / "optimizer.index.json").write_text(
        json.dumps(
            {
                "metadata": {"param_groups": "groups.bin"},
                "weight_map": {str(key): "states.bin" for key in state["state"]},
            }
        )
    )
    output = tmp_path / "new"
    migrate_optimizer_checkpoint(source, output, trust_legacy_pickle=True)
    restored = adam()
    GeneralCheckpointIO().load_optimizer(restored, str(output))
    assert_equal(state, restored.state_dict())


@pytest.mark.parametrize("suffix", [".bin", ".safetensors"])
def test_migrate_legacy_nested_tensors(tmp_path, suffix):
    state = adam().state_dict()
    state["param_groups"][0]["lr"] = torch.tensor(0.001)
    state["state"][0]["history"] = [torch.arange(3), (None, torch.tensor(0.5))]
    source = tmp_path / ("legacy" + suffix)
    if suffix == ".bin":
        torch.save(state, source)
    else:
        flat, objects = {}, []
        values = {
            f"state.{pid}.{key}": value for pid, fields in state["state"].items() for key, value in fields.items()
        }
        values["param_groups"] = state["param_groups"]
        for key, value in values.items():
            if not isinstance(value, torch.Tensor):
                objects.append(key)
                value = torch.tensor(list(pickle.dumps(value)), dtype=torch.uint8)
            flat[key] = value
        save_file(flat, str(source), metadata={"non_tensor_keys": json.dumps(objects)})
    output = tmp_path / "migrated.safetensors"
    migrate_optimizer_checkpoint(source, output, trust_legacy_pickle=True)
    assert_equal(state, load_flat(output, expected_kind="full"))


@pytest.mark.parametrize("suffix", [".bin", ""])
def test_migration_rejects_destination_suffix_before_decoding(tmp_path, monkeypatch, suffix):
    from colossalai.checkpoint_io import optimizer_migration

    source = tmp_path / "old.bin"
    torch.save(adam().state_dict(), source)
    destination = tmp_path / ("new" + suffix)

    def must_not_decode(path):
        pytest.fail("Invalid destinations must be rejected before legacy decoding")

    monkeypatch.setattr(optimizer_migration, "_load_trusted_legacy", must_not_decode)
    with pytest.raises(ValueError, match="destination must end with .safetensors"):
        migrate_optimizer_checkpoint(source, destination, trust_legacy_pickle=True)
    assert not destination.exists()


def test_moe_loads_migrated_shards_at_format_boundary(tmp_path, cpu_sync):
    from types import SimpleNamespace

    from colossalai.checkpoint_io.moe_checkpoint import MoECheckpointIO
    from colossalai.interface import OptimizerWrapper

    original = adam().state_dict()
    source = tmp_path / "old"
    source.mkdir()
    torch.save(original["param_groups"], source / "groups.bin")
    torch.save(original["state"], source / "states.bin")
    (source / "optimizer.index.json").write_text(
        json.dumps(
            {
                "metadata": {"param_groups": "groups.bin"},
                "weight_map": {"0": "states.bin", "1": "states.bin"},
            }
        )
    )
    destination = tmp_path / "new"
    migrate_optimizer_checkpoint(source, destination, trust_legacy_pickle=True)
    opt = adam()
    wrapped = OptimizerWrapper(opt)
    params = [param for group in opt.param_groups for param in group["params"]]
    wrapped.get_master_to_working_map = lambda: None
    wrapped.param_info = {
        "param2id": {id(param): index for index, param in enumerate(params)},
        "param2shape": {id(param): param.shape for param in params},
    }
    io = object.__new__(MoECheckpointIO)
    GeneralCheckpointIO.__init__(io)
    io.coordinator = SimpleNamespace(is_master=lambda: False)
    io.verbose = False
    # Isolate file dispatch here; expert redistribution is tested on CUDA.
    io.shard_from_complete_optimizer_state = lambda state, **kwargs: state
    io.load_optimizer(wrapped, str(destination))
    assert_equal(original, opt.state_dict())


@pytest.mark.parametrize(
    "entry",
    [
        "general",
        "low_level_zero",
        "fsdp",
        "hybrid_full",
        "hybrid_shard",
        "shard_helper",
    ],
)
def test_all_direct_loaders_reject_legacy_objects(tmp_path, entry):
    from types import SimpleNamespace

    from colossalai.booster.plugin.low_level_zero_plugin import LowLevelZeroCheckpointIO
    from colossalai.booster.plugin.torch_fsdp_plugin import FSDPOptimizerWrapper, TorchFSDPCheckpointIO
    from colossalai.checkpoint_io.hybrid_parallel_checkpoint_io import HybridParallelCheckpointIO
    from colossalai.checkpoint_io.utils import load_optim_or_model_shard
    from colossalai.interface import OptimizerWrapper

    path = tmp_path / "legacy.safetensors"
    save_file(
        {"state.0.step": torch.zeros(1, dtype=torch.uint8)},
        str(path),
        metadata={"non_tensor_keys": '["state.0.step"]'},
    )
    opt = adam()
    # Only exercise each real loader's format-rejection boundary here. Plugin
    # training and state distribution are covered by GPU integration tests.
    if entry == "general":
        call = lambda: GeneralCheckpointIO().load_unsharded_optimizer(opt, str(path))
    elif entry == "low_level_zero":
        io = object.__new__(LowLevelZeroCheckpointIO)
        GeneralCheckpointIO.__init__(io)
        call = lambda: io.load_unsharded_optimizer(opt, str(path))
    elif entry == "fsdp":
        io = object.__new__(TorchFSDPCheckpointIO)
        GeneralCheckpointIO.__init__(io)
        wrapped = object.__new__(FSDPOptimizerWrapper)
        call = lambda: io.load_unsharded_optimizer(wrapped, str(path))
    elif entry == "shard_helper":
        call = lambda: load_optim_or_model_shard(str(path), True, True)
    else:
        io = object.__new__(HybridParallelCheckpointIO)
        GeneralCheckpointIO.__init__(io)
        io.coordinator = SimpleNamespace(is_master=lambda: False)
        wrapped = OptimizerWrapper(opt)
        wrapped.get_master_to_working_map = lambda: None
        if entry == "hybrid_full":
            call = lambda: io.load_unsharded_optimizer(wrapped, str(path))
        else:
            params = [p for group in opt.param_groups for p in group["params"]]
            wrapped.param_info = {"param2id": {id(p): i for i, p in enumerate(params)}}
            save_optimizer_file(
                tmp_path / "groups.safetensors",
                {"param_groups": opt.state_dict()["param_groups"]},
            )
            index = tmp_path / "optimizer.index.json"
            index.write_text(
                json.dumps(
                    {
                        "metadata": {"param_groups": "groups.safetensors"},
                        "weight_map": {"0": path.name},
                    }
                )
            )
            call = lambda: io.load_sharded_optimizer(wrapped, str(index), "")
    with pytest.raises(LegacyOptimizerCheckpointError):
        call()


def test_migration_cli_requires_flag(tmp_path):
    from click.testing import CliRunner

    from colossalai.cli.cli import cli

    source = tmp_path / "old.bin"
    expected = adam().state_dict()
    torch.save(expected, source)
    destination = tmp_path / "new.safetensors"
    runner = CliRunner()
    missing = runner.invoke(cli, ["checkpoint", "migrate-optimizer", str(source), str(destination)])
    assert missing.exit_code != 0 and not destination.exists()
    result = runner.invoke(
        cli,
        [
            "checkpoint",
            "migrate-optimizer",
            "--trust-legacy-pickle",
            str(source),
            str(destination),
        ],
    )
    assert result.exit_code == 0, result.output
    assert_equal(expected, load_flat(destination))


@pytest.mark.skipif(
    not HAS_TENSORNVME or not torch.cuda.is_available(),
    reason="TensorNVMe requires a guarded CUDA device",
)
def test_real_async_writer(tmp_path):
    state = adam().state_dict()
    path = tmp_path / "optimizer.safetensors"
    writer = save_nested(str(path), state)
    writer.sync_before_step()
    writer.synchronize()
    assert_equal(state, load_flat(path))


@pytest.mark.skipif(
    not HAS_TENSORNVME or not torch.cuda.is_available(),
    reason="Requires a guarded CUDA device",
)
@pytest.mark.parametrize("shard", [False, True])
def test_general_repeated_async_save(tmp_path, shard):
    def build():
        return torch.optim.Adam(
            [
                {"params": [torch.nn.Parameter(torch.ones(2, 2))], "lr": 0.01},
                {"params": [torch.nn.Parameter(torch.ones(2))], "lr": 0.02},
            ]
        )

    optimizer = build()
    io = GeneralCheckpointIO()
    path = tmp_path / ("shards" if shard else "full.safetensors")
    # Save before state is initialized, then after initialization, then with a
    # longer parameter-group field. Pinned buffers must not encode object sizes.
    for step in range(3):
        if step:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    param.grad = torch.ones_like(param)
            optimizer.step()
            optimizer.param_groups[0]["note"] = "updated" * (step * 100)
        io.save_optimizer(optimizer, str(path), shard=shard, use_async=True, size_per_shard=0.00001)
        io._sync_d2h()
        io._sync_io()
        restored = build()
        io.load_optimizer(restored, str(path))
        assert_equal(optimizer.state_dict(), restored.state_dict())
