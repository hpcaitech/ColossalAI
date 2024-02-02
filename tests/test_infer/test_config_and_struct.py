import pytest

import colossalai
from colossalai.inference.config import InferenceConfig
from colossalai.inference.struct import BatchInfo, RequestStatus, Sequence
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
        block_table=None,
        eos_token_id=2,
        pad_token_id=2,
        max_output_len=256,
    )

    sequence2 = Sequence(
        request_id=2,
        prompt="bcd",
        input_token_id=[4, 5, 6],
        block_size=16,
        sample_params=None,
        block_table=None,
        eos_token_id=2,
        pad_token_id=2,
        max_output_len=256,
    )

    sequence3 = Sequence(
        request_id=3,
        prompt="efg",
        input_token_id=[7, 8, 9],
        block_size=16,
        sample_params=None,
        block_table=None,
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

    batch = BatchInfo(
        max_batch_size=8,
        kv_max_split_num=16,
        num_heads=2,
        head_dim=128,
    )
    batch.add_seqs([sequence])
    batch.add_seqs([sequence2, sequence3])

    # add duplicated sequence to test that it will not be counted twice
    batch.add_seqs([sequence])

    assert batch.is_empty == False
    assert batch.get_batch_size() == 3
    batch.update_batch_tokens([1, 2, 3])
    seq = batch.abort_seq(sequence)
    seq2 = batch.fliter_batch()[0]

    assert batch.get_batch_size() == 1
    assert seq.output_len == 1
    assert seq.output_token_id == [1]
    assert seq2.output_len == 1
    assert seq2.output_token_id == [2]

    batch.clear_batch()
    assert batch.is_empty == True


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, port=port, host="localhost")
    check_config_and_inference()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_config_and_inference():
    spawn(run_dist, 1)


if __name__ == "__main__":
    test_config_and_inference()
