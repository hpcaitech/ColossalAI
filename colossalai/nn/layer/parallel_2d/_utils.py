from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.global_variables import tensor_parallel_env as env


def get_summa_dim_from_env() -> int:
    try:
        summa_dim = env.summa_dim
        assert summa_dim > 0, 'SUMMA_DIM must be larger than zero'
        return summa_dim

    except KeyError as e:
        raise EnvironmentError('SUMMA_DIM is not found in the current environment, '
                               'please make sure that you have used the correct process group initializer')


def assert_summa_initialization():
    assert gpc.is_initialized(ParallelMode.PARALLEL_2D_COL) and \
           gpc.is_initialized(ParallelMode.PARALLEL_2D_ROW), \
        'Both TWO_DIMENSION_COL and TWO_DIMENSION_ROW must be initialized by the process group initializer'
