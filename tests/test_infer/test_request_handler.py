import pytest
import torch
from transformers.models.llama import LlamaConfig

import colossalai
from colossalai.inference.config import InferenceConfig
from colossalai.inference.core.request_handler import RequestHandler, RunningList
from colossalai.inference.struct import RequestStatus, Sequence
from colossalai.testing import spawn


def check_running_list():
    """
    Test the RunningList Structure.
    """
    running_list = RunningList(prefill_ratio=1.2)
    seq1 = Sequence(
        request_id=1,
        prompt="abc",
        input_token_id=[1, 2, 3],
        block_size=16,
        eos_token_id=0,
        sample_params=None,
        block_table=1,
    )

    running_list.append(seq1)
    assert running_list.ready_for_prefill()
    assert running_list.decoding == [] and running_list.prefill[0] == seq1

    seq = running_list.find_seq(seq1.request_id)
    assert seq == seq1

    running_list.remove(seq1)
    assert running_list.is_empty()


def check_request_handler():
    """
    Test main function of RequestHandler
    """
    inference_config = InferenceConfig(
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
        input_token_id=[1, 2, 3, 4, 5],
        block_size=16,
        eos_token_id=0,
        sample_params=None,
        block_table=torch.tensor([0, 0]),
    )
    request_handler.add_sequence(seq1)
    # the priority should be 1
    assert request_handler.waiting_list[1][0] == seq1
    assert request_handler._has_waiting()

    request_handler.abort_sequence(seq1.request_id)
    assert not request_handler._has_waiting()
    seq1.status = RequestStatus.WAITING
    request_handler.add_sequence(seq1)
    request_handler.schedule()


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, port=port, host="localhost")
    check_running_list()
    check_request_handler()


@pytest.mark.dist
def test_running_list_and_request_handler():
    spawn(run_dist, 1)


if __name__ == "__main__":
    test_running_list_and_request_handler()
