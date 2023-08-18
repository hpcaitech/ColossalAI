from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch.distributed as dist
from torch.distributed import ProcessGroup

from colossalai.cluster import ProcessGroupMesh


class PipelineStageManager:
    """PipelineStageManager is a helper class to manage pipeline stages.

    Args:
        pg_mesh (ProcessGroupMesh): Process group mesh.
        pipeline_axis (int): The axis along which the pipeline is constructed.

    Attributes:
        num_stages (int): Number of stages in the pipeline.
        stage (int): The current stage.
    """

    def __init__(self, pg_mesh: ProcessGroupMesh, pipeline_axis: int, is_virtual: bool = False) -> None:
        self.pg_mesh = pg_mesh
        self.pipeline_axis = pipeline_axis
        self.prev_rank: Optional[Tuple[int, ...]] = None
        self.next_rank: Optional[Tuple[int, ...]] = None
        self.p2p_groups: Dict[Tuple[int, int], ProcessGroup] = {}
        # init prev and next coord
        coord = self.pg_mesh.coordinate()
        # the prev rank of rank0 is the last rank
        prev_coord = coord[: self.pipeline_axis] + \
            (coord[self.pipeline_axis] - 1,) + coord[self.pipeline_axis + 1:]
        self.prev_rank = self.pg_mesh.ravel(prev_coord, self.pg_mesh.shape, mode='wrap')
        # the next rank of the last rank is rank0
        next_coord = coord[: self.pipeline_axis] + \
            (coord[self.pipeline_axis] + 1,) + coord[self.pipeline_axis + 1:]
        self.next_rank = self.pg_mesh.ravel(next_coord, self.pg_mesh.shape, mode='wrap')

        # init p2p process groups
        stages = list(range(self.num_stages))
        for prev, cur in zip(stages[:-1], stages[1:]):
            group = self.pg_mesh.get_group_along_axis(self.pipeline_axis, [prev, cur])
            if self.stage in [prev, cur]:
                ranks_in_group = self.pg_mesh.get_ranks_in_group(group)
                self.p2p_groups[tuple(ranks_in_group)] = group

        if is_virtual:
            # add the process group of the first rank and the last rank
            # only used in interleaved pipeline for now
            group = self.pg_mesh.get_group_along_axis(self.pipeline_axis, [stages[0], stages[-1]])
            if self.stage in [stages[0], stages[-1]]:
                ranks_in_group = self.pg_mesh.get_ranks_in_group(group)
                self.p2p_groups[tuple(ranks_in_group)] = group

    def is_first_stage(self) -> bool:
        """Is the current stage the first stage.

        Returns:
            bool: Whether the current stage is the first stage.
        """
        return self.stage == 0

    def is_last_stage(self) -> bool:
        """Is the current stage the last stage.

        Returns:
            bool: Whether the current stage is the last stage.
        """
        return self.stage == self.num_stages - 1

    @property
    def num_stages(self) -> int:
        """Number of stages in the pipeline.

        Returns:
            int: Number of stages in the pipeline.
        """
        return self.pg_mesh.size(self.pipeline_axis)

    @property
    def stage(self) -> int:
        """Current stage.

        Returns:
            int: Current stage.
        """
        return self.pg_mesh.coordinate(self.pipeline_axis)

    def get_rank(self) -> int:
        """Get the rank of the current process.

        Returns:
            int: Rank of the current process.
        """
        return dist.get_rank()

    def get_prev_rank(self) -> int:
        """Get the rank of the previous stage.

        Returns:
            int: Rank of the previous stage.
        """
        return self.prev_rank

    def get_next_rank(self) -> int:
        """Get the rank of the next stage.

        Returns:
            int: Rank of the next stage.
        """
        return self.next_rank

    def get_p2p_process_group(self, first_rank: int, second_rank: int) -> ProcessGroup:
        """Get the p2p process group between two ranks. The order of the two ranks does not matter.

        Args:
            first_rank (int): The first rank.
            second_rank (int): The second rank.

        Returns:
            ProcessGroup: P2P process group between the two ranks.
        """
        if first_rank > second_rank:
            first_rank, second_rank = second_rank, first_rank
        return self.p2p_groups[(first_rank, second_rank)]

    def init_process_group_by_stages(self, stages: List[int]) -> ProcessGroup:
        """Get the process group of the given stages.

        Args:
            stages (List[int]): List of stages.

        Returns:
            ProcessGroup: Process group of the given stages.
        """
        return self.pg_mesh.get_group_along_axis(self.pipeline_axis, stages)
