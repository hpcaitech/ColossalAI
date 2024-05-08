import pytest

import colossalai
from colossalai.inference.config import InferenceConfig
from colossalai.inference.struct import RequestStatus, Sequence
from colossalai.testing import rerun_if_address_is_in_use, spawn


def check_config_and_inference():
    config = InferenceConfig()
    assert config.max_batch_size == 8
    sequence = Sequence(
        request_id=1,
        prompt="abc",
        input_token_id=[1, 2, 3],
        block_size=16,
        sample_params=None,
        eos_token_id=2,
        pad_token_id=2,
        max_output_len=256,
    )

    sequence.mark_running()
    assert sequence.status == RequestStatus.RUNNING
    sequence.recycle()
    assert sequence.status == RequestStatus.RECYCLED

    assert sequence.sentence_len == 3
    assert sequence.input_len == 3
    assert sequence.output_len == 0
    assert sequence.check_finish() == False


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")
    check_config_and_inference()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_config_and_inference():
    spawn(run_dist, 1)


if __name__ == "__main__":
    test_config_and_inference()
