import pytest
import torch

from colossalai.testing import clear_cache_before_run
from colossalai.zero.legacy.gemini.stateful_tensor import StatefulTensor, TensorState


@pytest.mark.dist
@clear_cache_before_run()
def test_gemini_manager():
    # reset the manager, in case that there exists memory information left
    manager = StatefulTensor.GST_MGR
    manager.reset()

    # occupation 8
    st1 = StatefulTensor(torch.empty(2, 2, dtype=torch.float16, device='cuda'))
    # occupation 60
    st2 = StatefulTensor(torch.empty(3, 5, dtype=torch.float32, device='cpu'))

    # occupation 28
    t1 = torch.empty(7, device='cuda')
    # occupation 12
    t2 = torch.empty(3, device='cpu')
    st3 = StatefulTensor(t1, TensorState.HOLD_AFTER_FWD)
    st4 = StatefulTensor(None, TensorState.FREE)

    assert manager.total_number == 4
    assert manager.total_mem['cpu'] == 60
    assert manager.total_mem['cuda'] == 36
    assert manager.state_mem['cpu'][TensorState.HOLD] == 60
    assert manager.state_mem['cuda'][TensorState.HOLD] == 8
    assert manager.state_mem['cuda'][TensorState.HOLD_AFTER_FWD] == 28

    st4.payload_reset(t2)
    st3.payload_reset(t2)

    assert manager.total_number == 4
    assert manager.total_mem['cpu'] == 84
    assert manager.total_mem['cuda'] == 8
    assert manager.state_mem['cpu'][TensorState.HOLD] == 72
    assert manager.state_mem['cuda'][TensorState.HOLD] == 8
    assert manager.state_mem['cpu'][TensorState.HOLD_AFTER_FWD] == 12
    assert manager.state_mem['cuda'][TensorState.HOLD_AFTER_FWD] == 0

    st1.move_to(torch.device('cpu'))
    st2.move_to(torch.device('cpu'))
    st3.move_to(torch.device('cuda', 0))

    assert manager.total_number == 4
    assert manager.total_mem['cpu'] == 80
    assert manager.total_mem['cuda'] == 12
    assert manager.state_mem['cpu'][TensorState.HOLD] == 80
    assert manager.state_mem['cuda'][TensorState.HOLD] == 0
    assert manager.state_mem['cpu'][TensorState.HOLD_AFTER_FWD] == 0
    assert manager.state_mem['cuda'][TensorState.HOLD_AFTER_FWD] == 12

    st1.trans_state(TensorState.COMPUTE)
    st2.trans_state(TensorState.COMPUTE)
    st2.trans_state(TensorState.HOLD_AFTER_BWD)

    assert manager.total_number == 4
    assert manager.total_mem['cpu'] == 80
    assert manager.total_mem['cuda'] == 12
    assert manager.state_mem['cpu'][TensorState.HOLD] == 12
    assert manager.state_mem['cuda'][TensorState.HOLD] == 0
    assert manager.state_mem['cpu'][TensorState.HOLD_AFTER_FWD] == 0
    assert manager.state_mem['cuda'][TensorState.HOLD_AFTER_FWD] == 12
    assert manager.state_mem['cpu'][TensorState.HOLD_AFTER_BWD] == 60
    assert manager.state_mem['cuda'][TensorState.HOLD_AFTER_BWD] == 0
    assert manager.state_mem['cpu'][TensorState.COMPUTE] == 8
    assert manager.state_mem['cuda'][TensorState.COMPUTE] == 0


if __name__ == '__main__':
    test_gemini_manager()
