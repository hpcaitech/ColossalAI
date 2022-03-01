from colossalai.constants import INPUT_GROUP_3D, WEIGHT_GROUP_3D, OUTPUT_GROUP_3D
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.global_variables import tensor_parallel_env as env
from torch import Tensor


def get_depth_from_env() -> int:
    try:
        depth = env.depth_3d
        assert depth > 0, 'DEPTH must be greater than zero'
        return depth

    except KeyError as e:
        raise EnvironmentError('DEPTH is not found in the current environment, '
                               'please make sure that you have used the correct process group initializer')


def get_parallel_mode_from_env(group):
    assert group in [INPUT_GROUP_3D, WEIGHT_GROUP_3D, OUTPUT_GROUP_3D], \
        f'{group} is not valid for 3D tensor parallelism.'
    return getattr(env, group)


def get_last_group(a, b):
    mapping = {
        ParallelMode.PARALLEL_3D_INPUT: 'A',
        ParallelMode.PARALLEL_3D_WEIGHT: 'B',
        ParallelMode.PARALLEL_3D_OUTPUT: 'C',
    }

    res = chr(ord('A') + ord('B') + ord('C') - ord(mapping[a]) - ord(mapping[b]))

    if res == 'A':
        return ParallelMode.PARALLEL_3D_INPUT
    elif res == 'B':
        return ParallelMode.PARALLEL_3D_WEIGHT
    elif res == 'C':
        return ParallelMode.PARALLEL_3D_OUTPUT


def swap_in_out_group():
    env.input_group_3d, env.output_group_3d = env.output_group_3d, env.input_group_3d


def dbg_check_shape(tensor: Tensor, shape: tuple):
    rank = gpc.get_global_rank()
    if rank == 0:
        print(tensor.shape)
    assert tensor.shape == shape, \
        '{} does not match {}'.format(tensor.shape, shape)
