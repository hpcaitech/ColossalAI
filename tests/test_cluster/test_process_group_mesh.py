import pytest
import torch.distributed as dist

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.testing import spawn


def check_process_group_mesh_with_gpc():
    from colossalai.legacy.context import ParallelMode
    from colossalai.legacy.core import global_context as gpc

    DP_DIM, PP_DIM, TP_DIM = 0, 1, 2
    pg_mesh = ProcessGroupMesh(1, 2, 2)

    # check world size
    assert gpc.get_world_size(ParallelMode.TENSOR) == pg_mesh.size(
        TP_DIM
    ), f"{gpc.get_world_size(ParallelMode.TENSOR)} != {pg_mesh.size(TP_DIM)}"
    assert gpc.get_world_size(ParallelMode.PIPELINE) == pg_mesh.size(PP_DIM)
    assert gpc.get_world_size(ParallelMode.DATA) == pg_mesh.size(DP_DIM)

    # check locak rank (coordinate)
    assert gpc.get_local_rank(ParallelMode.TENSOR) == pg_mesh.coordinate(
        TP_DIM
    ), f"{gpc.get_local_rank(ParallelMode.TENSOR)} != {pg_mesh.coordinate(TP_DIM)}"
    assert gpc.get_local_rank(ParallelMode.PIPELINE) == pg_mesh.coordinate(PP_DIM)
    assert gpc.get_local_rank(ParallelMode.DATA) == pg_mesh.coordinate(DP_DIM)

    # check ranks in group
    tp_group = pg_mesh.get_group_along_axis(TP_DIM)
    assert gpc.get_ranks_in_group(ParallelMode.TENSOR) == pg_mesh.get_ranks_in_group(tp_group)
    pp_group = pg_mesh.get_group_along_axis(PP_DIM)
    assert gpc.get_ranks_in_group(ParallelMode.PIPELINE) == pg_mesh.get_ranks_in_group(pp_group)
    dp_group = pg_mesh.get_group_along_axis(DP_DIM)
    assert gpc.get_ranks_in_group(ParallelMode.DATA) == pg_mesh.get_ranks_in_group(dp_group)

    # check prev rank
    coord = pg_mesh.coordinate()
    if not gpc.is_first_rank(ParallelMode.TENSOR):
        assert coord[TP_DIM] != 0
        prev_coord = coord[:TP_DIM] + (coord[TP_DIM] - 1,) + coord[TP_DIM + 1 :]
        assert gpc.get_prev_global_rank(ParallelMode.TENSOR) == pg_mesh.ravel(prev_coord, pg_mesh.shape)
    if not gpc.is_first_rank(ParallelMode.PIPELINE):
        assert coord[PP_DIM] != 0
        prev_coord = coord[:PP_DIM] + (coord[PP_DIM] - 1,) + coord[PP_DIM + 1 :]
        assert gpc.get_prev_global_rank(ParallelMode.PIPELINE) == pg_mesh.ravel(prev_coord, pg_mesh.shape)

    # check next rank
    if not gpc.is_last_rank(ParallelMode.TENSOR):
        assert coord[TP_DIM] != pg_mesh.size(TP_DIM) - 1
        next_coord = coord[:TP_DIM] + (coord[TP_DIM] + 1,) + coord[TP_DIM + 1 :]
        assert gpc.get_next_global_rank(ParallelMode.TENSOR) == pg_mesh.ravel(next_coord, pg_mesh.shape)
    if not gpc.is_last_rank(ParallelMode.PIPELINE):
        assert coord[PP_DIM] != pg_mesh.size(PP_DIM) - 1
        next_coord = coord[:PP_DIM] + (coord[PP_DIM] + 1,) + coord[PP_DIM + 1 :]
        assert gpc.get_next_global_rank(ParallelMode.PIPELINE) == pg_mesh.ravel(next_coord, pg_mesh.shape)


def check_process_group_mesh_with_cases():
    DP_DIM, PP_DIM, TP_DIM = 0, 1, 2
    DP_SIZE, PP_SIZE, TP_SIZE = 1, 2, 2
    RANK_TO_COORDINATE = {
        0: (0, 0, 0),
        1: (0, 0, 1),
        2: (0, 1, 0),
        3: (0, 1, 1),
    }
    TP_RANKS_IN_GROUP = {
        0: [0, 1],
        1: [0, 1],
        2: [2, 3],
        3: [2, 3],
    }
    PP_RANKS_IN_GROUP = {
        0: [0, 2],
        1: [1, 3],
        2: [0, 2],
        3: [1, 3],
    }
    DP_RANKS_IN_GROUP = {
        0: [0],
        1: [1],
        2: [2],
        3: [3],
    }
    TPxPP_RANKS_IN_GROUP = {
        0: [0, 1, 2, 3],
        1: [0, 1, 2, 3],
        2: [0, 1, 2, 3],
        3: [0, 1, 2, 3],
    }
    DPxTP_RANKS_IN_GROUP = {
        0: [0, 1],
        1: [0, 1],
        2: [2, 3],
        3: [2, 3],
    }
    TPxPP_PARTIAL_INDICES = {
        0: [[0, 1], [0]],
        1: [[1], [0, 1]],
        2: [[0], [0, 1]],
        3: [[0, 1], [1]],
    }
    TPxPP_RANKS_IN_GROUP_PARTIAL = {
        0: [0, 1],
        1: [1, 3],
        2: [0, 2],
        3: [2, 3],
    }

    pg_mesh = ProcessGroupMesh(DP_SIZE, PP_SIZE, TP_SIZE)

    rank = dist.get_rank()
    assert rank == pg_mesh.rank

    # check world size
    assert pg_mesh.size(TP_DIM) == 2
    assert pg_mesh.size(PP_DIM) == 2
    assert pg_mesh.size(DP_DIM) == 1

    # check coordinate
    assert pg_mesh.coordinate(TP_DIM) == RANK_TO_COORDINATE[rank][TP_DIM]
    assert pg_mesh.coordinate(PP_DIM) == RANK_TO_COORDINATE[rank][PP_DIM]
    assert pg_mesh.coordinate(DP_DIM) == RANK_TO_COORDINATE[rank][DP_DIM]

    # check ranks in group
    tp_group = pg_mesh.get_group_along_axis(TP_DIM)
    assert pg_mesh.get_ranks_in_group(tp_group) == TP_RANKS_IN_GROUP[rank]
    pp_group = pg_mesh.get_group_along_axis(PP_DIM)
    assert pg_mesh.get_ranks_in_group(pp_group) == PP_RANKS_IN_GROUP[rank]
    dp_group = pg_mesh.get_group_along_axis(DP_DIM)
    assert pg_mesh.get_ranks_in_group(dp_group) == DP_RANKS_IN_GROUP[rank]
    dpxtp_group = pg_mesh.create_group_along_axis([DP_DIM, TP_DIM])
    assert pg_mesh.get_ranks_in_group(dpxtp_group) == DPxTP_RANKS_IN_GROUP[rank]
    tpxpp_group = pg_mesh.create_group_along_axis([TP_DIM, PP_DIM])
    assert pg_mesh.get_ranks_in_group(tpxpp_group) == TPxPP_RANKS_IN_GROUP[rank]
    tpxpp_group_partial = pg_mesh.create_group_along_axis([TP_DIM, PP_DIM], TPxPP_PARTIAL_INDICES[rank])
    assert pg_mesh.get_ranks_in_group(tpxpp_group_partial) == TPxPP_RANKS_IN_GROUP_PARTIAL[rank]

    # check prev rank
    if RANK_TO_COORDINATE[rank][TP_DIM] != 0:
        prev_coord = (
            RANK_TO_COORDINATE[rank][:TP_DIM]
            + (RANK_TO_COORDINATE[rank][TP_DIM] - 1,)
            + RANK_TO_COORDINATE[rank][TP_DIM + 1 :]
        )
        prev_rank = TP_RANKS_IN_GROUP[rank][TP_RANKS_IN_GROUP[rank].index(rank) - 1]
        assert pg_mesh.ravel(prev_coord, pg_mesh.shape) == prev_rank
    if RANK_TO_COORDINATE[rank][PP_DIM] != 0:
        prev_coord = (
            RANK_TO_COORDINATE[rank][:PP_DIM]
            + (RANK_TO_COORDINATE[rank][PP_DIM] - 1,)
            + RANK_TO_COORDINATE[rank][PP_DIM + 1 :]
        )
        prev_rank = PP_RANKS_IN_GROUP[rank][PP_RANKS_IN_GROUP[rank].index(rank) - 1]
        assert pg_mesh.ravel(prev_coord, pg_mesh.shape) == prev_rank

    # check next rank
    if RANK_TO_COORDINATE[rank][TP_DIM] != TP_SIZE - 1:
        next_coord = (
            RANK_TO_COORDINATE[rank][:TP_DIM]
            + (RANK_TO_COORDINATE[rank][TP_DIM] + 1,)
            + RANK_TO_COORDINATE[rank][TP_DIM + 1 :]
        )
        next_rank = TP_RANKS_IN_GROUP[rank][TP_RANKS_IN_GROUP[rank].index(rank) + 1]
        assert pg_mesh.ravel(next_coord, pg_mesh.shape) == next_rank
    if RANK_TO_COORDINATE[rank][PP_DIM] != PP_SIZE - 1:
        next_coord = (
            RANK_TO_COORDINATE[rank][:PP_DIM]
            + (RANK_TO_COORDINATE[rank][PP_DIM] + 1,)
            + RANK_TO_COORDINATE[rank][PP_DIM + 1 :]
        )
        next_rank = PP_RANKS_IN_GROUP[rank][PP_RANKS_IN_GROUP[rank].index(rank) + 1]
        assert pg_mesh.ravel(next_coord, pg_mesh.shape) == next_rank


def run_dist(rank, world_size, port):
    colossalai.launch(
        config=dict(parallel=dict(data=1, pipeline=2, tensor=dict(mode="1d", size=2))),
        rank=rank,
        world_size=world_size,
        port=port,
        host="localhost",
    )
    # TODO(ver217): this function should be removed when gpc is removed
    # check_process_group_mesh_with_gpc()
    check_process_group_mesh_with_cases()


@pytest.mark.dist
def test_process_group_mesh():
    spawn(run_dist, 4)


if __name__ == "__main__":
    test_process_group_mesh()
