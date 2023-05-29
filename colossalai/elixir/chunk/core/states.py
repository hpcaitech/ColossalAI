from enum import Enum


class TensorState(Enum):
    FREE = 0
    COMPUTE = 1
    HOLD = 2
    HOLD_AFTER_BWD = 3
    READY_FOR_REDUCE = 4


# expected: free -> hold -> compute -> hold ->
# -> compute -> hold_after_bwd -> ready_for_reduce
legal_ts_update_list = [(TensorState.FREE, TensorState.HOLD), (TensorState.FREE, TensorState.COMPUTE),
                        (TensorState.HOLD, TensorState.FREE), (TensorState.HOLD, TensorState.COMPUTE),
                        (TensorState.COMPUTE, TensorState.HOLD), (TensorState.COMPUTE, TensorState.HOLD_AFTER_BWD),
                        (TensorState.HOLD_AFTER_BWD, TensorState.COMPUTE),
                        (TensorState.HOLD_AFTER_BWD, TensorState.READY_FOR_REDUCE),
                        (TensorState.READY_FOR_REDUCE, TensorState.HOLD)]


def ts_update_sanity_check(old_state, new_state) -> bool:
    if (old_state, new_state) not in legal_ts_update_list:
        raise RuntimeError(f'illegal tensor state updating: {old_state} -> {new_state}')
    return True
