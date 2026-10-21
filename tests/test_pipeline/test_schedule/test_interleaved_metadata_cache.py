from contextlib import contextmanager

import torch

from colossalai.pipeline.schedule.interleaved_pp import InterleavedSchedule


class _StageManager:
    def __init__(self):
        self.model_chunk_id = None
        self.first_by_chunk = {}
        self.last_by_chunk = {}

    @contextmanager
    def switch_model_chunk_id(self, model_chunk_id):
        self.model_chunk_id = model_chunk_id
        yield
        self.model_chunk_id = None

    def is_first_stage(self):
        return self.first_by_chunk.get(self.model_chunk_id, False)

    def is_last_stage(self):
        return self.last_by_chunk.get(self.model_chunk_id, False)


class _Communication:
    def __init__(self):
        self.forward_recv_metadata = []
        self.backward_recv_metadata = []
        self.forward_send_metadata = []
        self.backward_send_metadata = []
        self.forward_send_recv_metadata = []
        self.backward_send_recv_metadata = []

    def recv_forward(self, _prev_rank, metadata_recv):
        self.forward_recv_metadata.append(metadata_recv)
        return torch.ones(1), []

    def recv_backward(self, _next_rank, metadata_recv):
        self.backward_recv_metadata.append(metadata_recv)
        return torch.ones(1), []

    def send_forward(self, _output_tensor, _next_rank, send_metadata):
        self.forward_send_metadata.append(send_metadata)
        return []

    def send_backward(self, _input_tensor_grad, _prev_rank, send_metadata):
        self.backward_send_metadata.append(send_metadata)
        return []

    def send_forward_recv_forward(self, _output_tensor, _is_send, _is_recv, send_metadata, metadata_recv, send_first):
        del send_first
        self.forward_send_recv_metadata.append((send_metadata, metadata_recv))
        return torch.ones(1), []

    def send_backward_recv_backward(
        self, _input_tensor_grad, _is_send, _is_recv, send_metadata, metadata_recv, send_first
    ):
        del send_first
        self.backward_send_recv_metadata.append((send_metadata, metadata_recv))
        return torch.ones(1), []


def test_interleaved_metadata_cache_is_chunk_local():
    schedule = InterleavedSchedule(_StageManager(), num_model_chunks=2, num_microbatch=2)
    communication = _Communication()
    schedule.comm = communication

    schedule.recv_forward(0)
    schedule.recv_forward(1)
    assert communication.forward_recv_metadata == [None, None]
    assert schedule.tensor_metadata_recv[0] is not None
    assert schedule.tensor_metadata_recv[1] is not None
    schedule.recv_forward(0)
    assert communication.forward_recv_metadata[2] is schedule.tensor_metadata_recv[0]

    schedule.recv_backward(0)
    schedule.recv_backward(1)
    assert communication.backward_recv_metadata == [None, None]
    assert schedule.grad_metadata_recv[0] is not None
    assert schedule.grad_metadata_recv[1] is not None
    schedule.recv_backward(0)
    assert communication.backward_recv_metadata[2] is schedule.grad_metadata_recv[0]

    tensor = torch.ones(1)
    schedule.send_forward(0, tensor)
    schedule.send_forward(1, tensor)
    schedule.send_forward(0, tensor)
    assert communication.forward_send_metadata == [True, True, False]

    schedule.send_backward(0, tensor)
    schedule.send_backward(1, tensor)
    schedule.send_backward(0, tensor)
    assert communication.backward_send_metadata == [True, True, False]

    schedule = InterleavedSchedule(_StageManager(), num_model_chunks=2, num_microbatch=2)
    communication = _Communication()
    schedule.comm = communication
    schedule.send_forward_recv_forward(0, 1, tensor)
    schedule.send_forward_recv_forward(1, 0, tensor)
    assert communication.forward_send_recv_metadata[0] == (True, None)
    assert communication.forward_send_recv_metadata[1] == (True, None)
    assert schedule.send_tensor_metadata == [False, False]
    assert all(metadata is not None for metadata in schedule.tensor_metadata_recv)

    schedule.send_backward_recv_backward(0, 1, tensor)
    schedule.send_backward_recv_backward(1, 0, tensor)
    assert communication.backward_send_recv_metadata[0] == (True, None)
    assert communication.backward_send_recv_metadata[1] == (True, None)
    assert schedule.send_grad_metadata == [False, False]
    assert all(metadata is not None for metadata in schedule.grad_metadata_recv)


def test_chunk_metadata_structure_and_no_send_state():
    stage_manager = _StageManager()
    schedule = InterleavedSchedule(stage_manager, num_model_chunks=2, num_microbatch=2)
    communication = _Communication()
    communication.recv_forward = lambda rank, metadata_recv: (
        (
            {"hidden": torch.ones(2)}
            if len(communication.forward_recv_metadata) == 1
            else {"hidden": torch.ones(3), "router": (torch.ones(1),)}
        ),
        [],
    )
    original_recv = communication.recv_forward

    def record_recv(rank, metadata_recv):
        communication.forward_recv_metadata.append(metadata_recv)
        return original_recv(rank, metadata_recv)

    communication.recv_forward = record_recv
    schedule.comm = communication
    schedule.recv_forward(0)
    schedule.recv_forward(1)
    assert schedule.tensor_metadata_recv[0] != schedule.tensor_metadata_recv[1]

    # A combined op that does not send must not consume this chunk's first-send metadata state.
    stage_manager.last_by_chunk[0] = True
    schedule.send_forward_recv_forward(0, 1, torch.ones(1))
    assert schedule.send_tensor_metadata[0] is True
    stage_manager.last_by_chunk[0] = False
    schedule.send_forward_recv_forward(0, 1, torch.ones(1))
    assert schedule.send_tensor_metadata[0] is False


def test_metadata_cache_disabled_keeps_sending():
    schedule = InterleavedSchedule(_StageManager(), num_model_chunks=2, num_microbatch=2, enable_metadata_cache=False)
    communication = _Communication()
    schedule.comm = communication
    tensor = torch.ones(1)
    schedule.send_forward(0, tensor)
    schedule.send_forward(0, tensor)
    schedule.send_backward(1, tensor)
    schedule.send_backward(1, tensor)
    assert communication.forward_send_metadata == [True, True]
    assert communication.backward_send_metadata == [True, True]


def test_batch_size_change_invalidates_all_chunks():
    schedule = InterleavedSchedule(_StageManager(), num_model_chunks=2, microbatch_size=2)
    schedule.forward_only = True
    schedule.last_batch_size = 4
    schedule.send_tensor_metadata = [False, False]
    schedule.send_grad_metadata = [False, False]
    schedule.tensor_metadata_recv = [object(), object()]
    schedule.grad_metadata_recv = [object(), object()]
    schedule.load_batch(iter([{"input": torch.ones(6, 3)}]))
    assert schedule.enable_metadata_cache is False
    assert schedule.send_tensor_metadata == [True, True]
    assert schedule.send_grad_metadata == [True, True]
    assert schedule.tensor_metadata_recv == [None, None]
    assert schedule.grad_metadata_recv == [None, None]


def test_backward_no_send_keeps_first_metadata():
    manager = _StageManager()
    manager.first_by_chunk[0] = True
    schedule = InterleavedSchedule(manager, num_model_chunks=2, num_microbatch=2)
    schedule.comm = _Communication()
    schedule.send_backward_recv_backward(0, 1, torch.ones(1))
    assert schedule.send_grad_metadata[0] is True
    manager.first_by_chunk[0] = False
    schedule.send_backward_recv_backward(0, 1, torch.ones(1))
    assert schedule.send_grad_metadata[0] is False
