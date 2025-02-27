from colossalai.pipeline.schedule.dualpipe_schedule import DualPipeGraph
from colossalai.testing import parameterize


@parameterize(
    "test_config",
    [
        # {
        #     "n_stage": 8,
        # },
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
        # chunk_mode=True,
        mbs_mode=True,
        empty_bubble_str_mode=True,
    )


if __name__ == "__main__":
    test_dualpipe_schedule()
