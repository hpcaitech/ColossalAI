from colossalai.inference.config import InferenceConfig
from colossalai.inference.core.request_handler import RequestHandler, RunningList
from colossalai.inference.struct import Sequence


def test_running_list():
    """
    Test the RunningList Structure.
    """
    running_list = RunningList(ratio=1.2)
    seq1 = Sequence(
        request_id=1,
        prompt="abc",
        token_id=[1, 2, 3],
        block_size=16,
        sample_params=None,
        block_table_index=1,
    )

    running_list.append(seq1)
    assert running_list.ready_for_prefill()
    assert running_list.decoding == [] and running_list.prefill[0] == seq1

    seq = running_list.find_seq(seq1.request_id)
    assert seq == seq1

    running_list.remove(seq1)
    assert running_list.is_empty()


def test_request_handler():
    """
    Test main function of RequestHandler
    """
    config = InferenceConfig(
        max_input_len=10,
    )
    RequestHandler()


if __name__ == "__main__":
    test_running_list()
    test_request_handler()
