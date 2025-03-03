from typing import List

from colossalai.pipeline.schedule.dualpipe_schedule import DUALPIPE_NODETYPE, DualPipeGraph
from colossalai.pipeline.schedule.v_schedule import DualVPipelineGraph, PipelineGraph, ScheduledNode
from colossalai.testing import parameterize


def print_pipeline_details(
    pipeline_schedule: List[List[ScheduledNode]],
    chunk_mode: bool = False,
    mbs_mode: bool = False,
    empty_bubble_str_mode: bool = False,
):
    assert not (
        chunk_mode and mbs_mode
    ), "Only one mode is supported at the same time, please choose from chunk_mode and mbs_mode"
    schedule_str = ""
    for stage in range(len(pipeline_schedule)):
        stage_nodes = []
        for node in pipeline_schedule[stage]:
            if node.type in DUALPIPE_NODETYPE:
                if node.type == "EMPTY_BUBBLE":
                    if empty_bubble_str_mode:
                        stage_nodes.append("E")
                    else:
                        stage_nodes.append(" ")
                else:
                    if chunk_mode:
                        stage_nodes.append(node.type + str(node.chunk))
                    elif mbs_mode:
                        stage_nodes.append(node.type + str(node.minibatch))
                    else:
                        stage_nodes.append(node.type)
        stage_str = "".join([_ for _ in stage_nodes])
        schedule_str += "\n" + stage_str
    print(schedule_str)


@parameterize(
    "test_config",
    [
        {
            "n_stage": 16,
        },
    ],
)
def test_dualpipe_schedule(test_config):
    dualpipe = DualPipeGraph(
        n_stage=test_config["n_stage"],
        n_micro=(test_config["n_stage"] + 2) * 2,
    )
    dualpipe_schedule = dualpipe.get_dualpipe_schedule()
    # print(dualpipe_schedule)
    dualpipe.print_details(
        dualpipe_schedule,
        chunk_mode=True,
        # mbs_mode=True,
        empty_bubble_str_mode=True,
    )


@parameterize(
    "test_config",
    [
        {
            "n_stage": 4,
        },
    ],
)
def test_dualpipeV_schedule(test_config):
    mem_f = 34 * 4096 + 5 * 24 * 4096
    mem_w = -32 * 4096
    mem_b = -mem_w - mem_f
    # zbv
    zbv_schedule = PipelineGraph(
        n_stage=test_config["n_stage"],
        n_micro=8,
        f_cost=1000,
        b_cost=1000,
        w_cost=1000,
        c_cost=1,
        f_mem=mem_f * 1.5,
        b_mem=mem_b * 1.5,
        w_mem=mem_w * 1.5,
    ).get_v_schedule()
    print_pipeline_details(
        zbv_schedule,
        mbs_mode=True,
    )

    # dual V
    dualV_graph = DualVPipelineGraph(
        n_stage=test_config["n_stage"],
        n_micro=8,
        f_cost=1000,
        b_cost=1000,
        w_cost=1000,
        c_cost=1,
        f_mem=mem_f * 1.5,
        b_mem=mem_b * 1.5,
        w_mem=mem_w * 1.5,
    )
    dualV_schedule = dualV_graph.get_v_schedule()
    dualV_schedule = dualV_graph.convert_to_dualV(dualV_schedule)
    print_pipeline_details(
        dualV_schedule,
        mbs_mode=True,
    )


if __name__ == "__main__":
    # test_dualpipe_schedule()
    test_dualpipeV_schedule()
