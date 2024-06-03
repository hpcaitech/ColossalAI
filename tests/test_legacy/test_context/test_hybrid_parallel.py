#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from pathlib import Path

import torch

from colossalai.legacy import launch
from colossalai.legacy.context import reset_seeds
from colossalai.legacy.context.parallel_mode import ParallelMode
from colossalai.legacy.core import global_context as gpc
from colossalai.legacy.global_variables import tensor_parallel_env as tp_env
from colossalai.testing import free_port, rerun_if_address_is_in_use, spawn

CONFIG_PATH_LIST = list(Path(__file__).parent.glob("configs/*.py"))


def check_data_parallel_rank(rank):
    global_world_size = gpc.get_world_size(ParallelMode.GLOBAL)
    mp_size = gpc.get_world_size(ParallelMode.MODEL)
    num_dp_groups = global_world_size // mp_size
    dp_local_rank = gpc.get_local_rank(ParallelMode.DATA)

    assert gpc.get_world_size(ParallelMode.DATA) == num_dp_groups

    for group_idx in range(num_dp_groups):
        ranks_in_dp_group = range(group_idx * mp_size, (group_idx + 1) * mp_size)
        if rank in ranks_in_dp_group:
            assert dp_local_rank == group_idx


def check_pipeline_parallel_rank(rank):
    mp_world_size = gpc.get_world_size(ParallelMode.MODEL)
    tp_world_size = gpc.get_world_size(ParallelMode.TENSOR)
    num_pipeline_stage = mp_world_size // tp_world_size
    pipeline_local_rank = gpc.get_local_rank(ParallelMode.PIPELINE)

    for stage_idx in range(num_pipeline_stage):
        ranks_in_current_stage = range(stage_idx * tp_world_size, (stage_idx + 1) * tp_world_size)
        if rank in ranks_in_current_stage:
            assert stage_idx == pipeline_local_rank


def check_model_parallel_rank(rank):
    mp_size = gpc.get_world_size(ParallelMode.MODEL)
    rank_within_mp_group = rank % mp_size
    mp_local_rank = gpc.get_local_rank(ParallelMode.MODEL)
    assert rank_within_mp_group == mp_local_rank


def check_tensor_parallel_rank(rank):
    if tp_env.mode == "2d":
        check_2d_tensor_parallel_rank(rank)
    elif tp_env == "2.5d":
        check_2p5d_tensor_parallel_rank(rank)
    elif tp_env == "3d":
        check_3d_tensor_parallel_rank(rank)


def get_tp_info():
    global_world_size = gpc.get_world_size(ParallelMode.GLOBAL)
    tp_world_size = gpc.get_world_size(ParallelMode.TENSOR)
    num_tp_groups = global_world_size // tp_world_size
    tp_local_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    return tp_local_rank, tp_world_size, num_tp_groups


def check_2d_tensor_parallel_rank(rank):
    tp_local_rank, tp_world_size, num_tp_groups = get_tp_info()

    for group_id in range(num_tp_groups):
        ranks_in_current_tp_group = range(group_id * tp_world_size, (group_id + 1) * tp_world_size)

        if rank in ranks_in_current_tp_group:
            col_local_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2D_COL)
            row_local_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2D_ROW)

            assert col_local_rank == tp_local_rank // tp_env.summa_dim
            assert row_local_rank == tp_local_rank % tp_env.summa_dim


def check_2p5d_tensor_parallel_rank(rank):
    tp_local_rank, tp_world_size, num_tp_groups = get_tp_info()

    for group_id in range(num_tp_groups):
        ranks_in_current_tp_group = range(group_id * tp_world_size, (group_id + 1) * tp_world_size)

        if rank in ranks_in_current_tp_group:
            rp_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_ROW)
            cp_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL)
            dp_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_DEP)
            xp_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_XZ)

            assert rp_rank == tp_local_rank % tp_env.summa_dim
            assert cp_rank == tp_local_rank // tp_env.tesseract_dim
            assert dp_rank == tp_local_rank // (tp_env.summa_dim**2)
            assert xp_rank == tp_local_rank // tp_env.summa_dim


def check_3d_tensor_parallel_rank(rank):
    tp_local_rank, tp_world_size, num_tp_groups = get_tp_info()

    for group_id in range(num_tp_groups):
        ranks_in_current_tp_group = range(group_id * tp_world_size, (group_id + 1) * tp_world_size)

        if rank in ranks_in_current_tp_group:
            ip_rank = gpc.get_local_rank(ParallelMode.PARALLEL_3D_INPUT)
            wp_rank = gpc.get_local_rank(ParallelMode.PARALLEL_3D_WEIGHT)
            op_rank = gpc.get_local_rank(ParallelMode.PARALLEL_3D_OUTPUT)

            assert ip_rank == tp_local_rank % tp_env.depth_3d
            assert wp_rank == tp_local_rank // tp_env.depth_3d
            assert op_rank == tp_local_rank // (tp_env.depth_3d**2)


def init_context(config_path, rank, world_size, backend, port, host):
    dist_args = dict(
        config=config_path, rank=rank, world_size=world_size, backend=backend, port=port, host=host, verbose=True
    )
    launch(**dist_args)

    check_tensor_parallel_rank(rank)
    check_data_parallel_rank(rank)
    check_pipeline_parallel_rank(rank)
    check_model_parallel_rank(rank)
    gpc.destroy()
    torch.cuda.empty_cache()


def run_dist(rank, world_size, port, backend, port_list, host):
    for config_path, current_port in zip(CONFIG_PATH_LIST, port_list):
        init_context(
            config_path=config_path, rank=rank, world_size=world_size, backend=backend, port=current_port, host=host
        )
        reset_seeds()


@rerun_if_address_is_in_use()
def test_context():
    """
    As no computation or communication is done, we can run this test on CPU.
    """
    world_size = 32
    port_list = []

    for _ in range(len(CONFIG_PATH_LIST)):
        while True:
            port = free_port()
            if port not in port_list:
                port_list.append(port)
                break

    spawn(run_dist, world_size, backend="gloo", port_list=port_list, host="localhost")


if __name__ == "__main__":
    test_context()
