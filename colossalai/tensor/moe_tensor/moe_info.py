from colossalai.cluster import ProcessGroupMesh


class MoeParallelInfo:
    """Moe parallelism information, storing parallel sizes and groups.
    """

    def __init__(self, ep_size: int, dp_size: int):
        self.dp_axis = 0
        self.dp_size = dp_size
        self.ep_axis = 1
        self.ep_size = ep_size
        self.pg = ProcessGroupMesh(self.dp_size, self.ep_size)
        self.ep_group = self.pg.get_group_along_axis(self.ep_axis)
        self.ep_group_ranks = self.pg.get_ranks_in_group(self.ep_group)
        self.dp_group = self.pg.get_group_along_axis(self.dp_axis)
        self.dp_group_ranks = self.pg.get_ranks_in_group(self.dp_group)
