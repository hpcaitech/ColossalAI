from colossalai.inference.config import InferenceConfig
from colossalai.inference.struct import BatchInfo, RequsetStatus, Sequence


def test_config_and_inferenceData():
    config = InferenceConfig("/llama")
    assert config.max_batch_size
    sequence = Sequence(
        request_id=1,
        prompt="abc",
        token_id=[1, 2, 3],
        block_size=16,
        sample_params=None,
        block_table_index=1,
    )

    sequence2 = Sequence(
        request_id=2,
        prompt="bcd",
        token_id=[4, 5, 6],
        block_size=16,
        sample_params=None,
        block_table_index=2,
    )

    assert sequence.get_sentence_len() == 3
    assert sequence.get_input_len() == 3
    assert sequence.get_output_len() == 0
    assert sequence.check_finish() == False

    batch = BatchInfo.init_batch([sequence])
    assert batch.block_table[sequence.request_id] == sequence.block_table_index
    sequence.status = RequsetStatus.COMPLETED
    batch.fliter_batch()
    assert batch.block_table == {}
    batch.add_seqs([sequence2])
    assert batch.block_table[sequence2.request_id] == sequence2.block_table_index
    batch.clear_batch()
    assert batch.block_table == {}


if __name__ == "__main__":
    test_config_and_inferenceData()
