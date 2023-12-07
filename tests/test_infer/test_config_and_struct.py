from colossalai.inference.core.config import InferenceConfig
from colossalai.inference.core.inference_struct import BatchHandler, Sequence


def test_config_and_struct():
    InferenceConfig("/llama")
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

    batch = BatchHandler.init_batch([sequence])
    batch.fliter_batch()
    batch.add_seqs([sequence2])
    batch.clear_batch()


if __name__ == "__main__":
    test_config_and_struct()
