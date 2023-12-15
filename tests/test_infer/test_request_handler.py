import torch
from transformers.models.llama import LlamaConfig

from colossalai.inference.config import InferenceConfig
from colossalai.inference.core.request_handler import RequestHandler, RunningList
from colossalai.inference.struct import RequsetStatus, Sequence


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
    inference_config = InferenceConfig(
        model="",
        max_input_len=10,
        max_output_len=10,
        block_size=8,
    )
    model_config = LlamaConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    request_handler = RequestHandler(inference_config, model_config)
    seq1 = Sequence(
        request_id=1,
        prompt="abc",
        token_id=[1, 2, 3, 4, 5],
        block_size=16,
        sample_params=None,
        block_table_index=torch.tensor([0, 0]),
    )
    request_handler.add_sequence(seq1)
    # the priority should be 1
    assert request_handler.waiting_list[1][0] == seq1
    assert request_handler._has_waiting()

    request_handler.abort_sequence(seq1.request_id)
    assert not request_handler._has_waiting()
    seq1.status = RequsetStatus.WAITING
    request_handler.add_sequence(seq1)
    request_handler.schedule()


if __name__ == "__main__":
    # test_running_list()
    test_request_handler()
