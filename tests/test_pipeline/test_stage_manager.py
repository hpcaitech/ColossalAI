import pytest
import torch.distributed as dist

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.testing import rerun_if_address_is_in_use, spawn


def check_stage_manager():
    DP_DIM, PP_DIM = 0, 1
    DP_SIZE, PP_SIZE = 2, 2
    RANK_TO_COORDINATE = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, 1),
    }
    PP_RANKS_IN_GROUP = {
        0: [0, 1],
        1: [0, 1],
        2: [2, 3],
        3: [2, 3],
    }
    pg_mesh = ProcessGroupMesh(DP_SIZE, PP_SIZE)
    stage_manager = PipelineStageManager(pg_mesh, PP_DIM)
    rank = dist.get_rank()

    # check stage info
    assert stage_manager.num_stages == PP_SIZE
    assert stage_manager.stage == RANK_TO_COORDINATE[rank][PP_DIM]

    # check is_first_stage
    ranks_in_group = PP_RANKS_IN_GROUP[rank]
    is_first_stage = ranks_in_group.index(rank) == 0
    assert stage_manager.is_first_stage() == is_first_stage

    # check is_last_stage
    is_last_stage = ranks_in_group.index(rank) == len(ranks_in_group) - 1
    assert stage_manager.is_last_stage() == is_last_stage

    # check prev rank
    if not is_first_stage:
        prev_rank = ranks_in_group[ranks_in_group.index(rank) - 1]
        assert stage_manager.get_prev_rank() == prev_rank

    # check next rank
    if not is_last_stage:
        next_rank = ranks_in_group[ranks_in_group.index(rank) + 1]
        assert stage_manager.get_next_rank() == next_rank

    # check p2p groups
    for prev, cur in zip(ranks_in_group[:-1], ranks_in_group[1:]):
        if rank in [prev, cur]:
            group = stage_manager.get_p2p_process_group()
            dist.barrier(group=group)

    # check stage groups
    pg_mesh = ProcessGroupMesh(4)
    stage_manager = PipelineStageManager(pg_mesh, 0)
    group = stage_manager.init_process_group_by_stages([0, 2])
    if rank in [0, 2]:
        dist.barrier(group=group)


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")
    check_stage_manager()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_pipeline_stage_manager():
    spawn(run_dist, 4)


if __name__ == "__main__":
    test_pipeline_stage_manager()
