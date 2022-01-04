import torch.cuda
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from ._helper import add_seed, get_seeds, set_mode


def moe_set_seed(seed):
    if torch.cuda.is_available():

        moe_tp_rank = gpc.get_local_rank(ParallelMode.MOE_TENSOR)
        moe_tp_seed = seed + moe_tp_rank
        add_seed(ParallelMode.MOE_TENSOR, moe_tp_seed)

        global_rank = gpc.get_global_rank()
        add_seed(ParallelMode.TENSOR, global_rank, True)
        print(f"moe seed condition: {global_rank} with moe seed {moe_tp_seed}, ",
              f"tensor seed {global_rank}", flush=True)
