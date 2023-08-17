from colossalai.tensor import ProcessGroup


class MoeParallelInfo:
    """Moe parallelism information, storing parallel sizes and groups.
    """

    def __init__(self, ep_size: int, dp_size: int):
        self.ep_size = ep_size
        self.dp_size = dp_size
        self.pg = ProcessGroup(tp_degree=ep_size, dp_degree=dp_size)
        self.ep_group = self.pg.tp_process_group()
        self.dp_group = self.pg.dp_process_group()
