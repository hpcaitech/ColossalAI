from colossalai.cluster import ProcessGroupMesh


class MoeParallelInfo:
    """Moe parallelism information, storing parallel sizes and groups."""

    def __init__(self, ep_inside: bool, ep_size: int, dp_size: int, pp_size: int = 1):
        """
        init MoeParallelInfo with ep_size, dp_size and pp_size

        Args:
            ep_size (int): expert parallel size
            dp_size (int): data parallel (zero) size
            pp_size (int, optional): pipeline parallel size. Defaults to 1.
            ep_inside (bool, optional): Use ep inside dp if True, dp inside ep if False. Defaults to True.
        """
        self.pp_size, self.dp_size, self.ep_size = pp_size, dp_size, ep_size
        if ep_inside:
            self.pp_axis, self.dp_axis, self.ep_axis = 0, 1, 2
            self.pg = ProcessGroupMesh(self.pp_size, self.dp_size, self.ep_size)
        else:
            self.pp_axis, self.ep_axis, self.dp_axis = 0, 1, 2
            self.pg = ProcessGroupMesh(self.pp_size, self.ep_size, self.dp_size)

        self.ep_group = self.pg.get_group_along_axis(self.ep_axis)
        self.ep_group_ranks = self.pg.get_ranks_in_group(self.ep_group)
        self.dp_group = self.pg.get_group_along_axis(self.dp_axis)
        self.dp_group_ranks = self.pg.get_ranks_in_group(self.dp_group)
        self.ep_rank = self.pg.coordinate(self.ep_axis)
        self.dp_rank = self.pg.coordinate(self.dp_axis)
