import os

from colossalai.context.parallel_mode import ParallelMode
from colossalai.context.process_group_initializer.initializer_2d import SUMMA_DIM
from colossalai.core import global_context as gpc


def get_summa_dim_from_env() -> int:
    try:
        summa_dim = os.environ[SUMMA_DIM]
        summa_dim = int(summa_dim)
        assert summa_dim > 0, 'SUMMA_DIM must be larger than zero'
        return summa_dim

    except KeyError as e:
        raise EnvironmentError('SUMMA_DIM is not found in the current environment, '
                               'please make sure that you have used the correct process group initializer')


def assert_summa_initialization():
    assert gpc.is_initialized(ParallelMode.PARALLEL_2D_COL) and \
           gpc.is_initialized(ParallelMode.PARALLEL_2D_ROW), \
        'Both TWO_DIMENSION_COL and TWO_DIMENSION_ROW must be initialized by the process group initializer'
