from colossalai.legacy.context.parallel_mode import ParallelMode
from colossalai.legacy.core import global_context as gpc
from colossalai.legacy.global_variables import tensor_parallel_env as env


def get_tesseract_dim_dep_from_env():
    try:
        tesseract_dim = env.tesseract_dim
        tesseract_dep = env.tesseract_dep
        assert tesseract_dim > 0, "TESSERACT_DIM must be larger than zero"
        assert tesseract_dep > 0, "TESSERACT_DEP must be larger than zero"
        return tesseract_dim, tesseract_dep

    except KeyError:
        raise EnvironmentError(
            "TESSERACT_DIM or TESSERACT_DEP is not found in the current environment, "
            "please make sure that you have used the correct process group initializer"
        )


def assert_tesseract_initialization():
    assert (
        gpc.is_initialized(ParallelMode.PARALLEL_2P5D_COL)
        and gpc.is_initialized(ParallelMode.PARALLEL_2P5D_ROW)
        and gpc.is_initialized(ParallelMode.PARALLEL_2P5D_DEP)
        and gpc.is_initialized(ParallelMode.PARALLEL_2P5D_XZ)
    ), (
        "Both PARALLEL_2P5D_COL, PARALLEL_2P5D_ROW, PARALLEL_2P5D_DEP and PARALLEL_2P5D_XZ "
        "must be initialized by the process group initializer"
    )
