from enum import Enum


class TensorState(Enum):
    """
    TensorState represents the state of a tensor in Elixir.
    There are five states of a tensor: free, compute, hold, hold_after_bwd, ready_for_reduce.
    """
    FREE = 0
    COMPUTE = 1
    HOLD = 2
    HOLD_AFTER_BWD = 3
    READY_FOR_REDUCE = 4


# this includes the possible state transition in tensor state:
# the item in the list is in the format of (old_state, new_state)
# the complete state transtition is:
# free -> hold -> compute -> hold ->
# -> compute -> hold_after_bwd -> ready_for_reduce
LEGAL_TENSOR_STATE_UPDATE_LIST = [(TensorState.FREE, TensorState.HOLD), (TensorState.FREE, TensorState.COMPUTE),
                                  (TensorState.HOLD, TensorState.FREE), (TensorState.HOLD, TensorState.COMPUTE),
                                  (TensorState.COMPUTE, TensorState.HOLD),
                                  (TensorState.COMPUTE, TensorState.HOLD_AFTER_BWD),
                                  (TensorState.HOLD_AFTER_BWD, TensorState.COMPUTE),
                                  (TensorState.HOLD_AFTER_BWD, TensorState.READY_FOR_REDUCE),
                                  (TensorState.READY_FOR_REDUCE, TensorState.HOLD)]


def validate_tensor_state_update(old_state: TensorState, new_state: TensorState, raise_exception: bool = False) -> bool:
    """
    Validate the tensor state update is legal or not.

    Args:
        old_state (TensorState): the old state of the tensor
        new_state (TensorState): the new state of the tensor
        raise_exception (bool, optional): whether to raise exception when the state update is illegal. Defaults to False.

    Returns:
        bool: whether the state update is legal or not.
    """
    if (old_state, new_state) not in LEGAL_TENSOR_STATE_UPDATE_LIST:
        if raise_exception:
            raise RuntimeError(f'Found illegal tensor state updating: {old_state} -> {new_state}')
        else:
            return False
    return True
