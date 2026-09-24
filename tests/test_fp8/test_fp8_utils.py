import pytest

from colossalai.quantization import fp8


@pytest.mark.parametrize(
    "local_world_size_var",
    ["LOCAL_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_SIZE", "SLURM_TASKS_PER_NODE"],
)
@pytest.mark.parametrize(
    "group_ranks, expected",
    [([0, 1, 2, 3], True), ([0, 1, 4, 5], False)],
)
def test_process_group_is_intranode(monkeypatch, local_world_size_var, group_ranks, expected):
    for var in ["LOCAL_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_SIZE", "SLURM_TASKS_PER_NODE"]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(local_world_size_var, "4")
    monkeypatch.setattr(fp8.dist, "get_process_group_ranks", lambda _: group_ranks)

    assert fp8.process_group_is_intranode(object()) is expected


def test_process_group_is_intranode_prefers_torchrun_local_world_size(monkeypatch):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_SIZE", "2")
    monkeypatch.setenv("SLURM_TASKS_PER_NODE", "1")
    monkeypatch.setattr(fp8.dist, "get_process_group_ranks", lambda _: [0, 3])

    assert fp8.process_group_is_intranode(object()) is True
