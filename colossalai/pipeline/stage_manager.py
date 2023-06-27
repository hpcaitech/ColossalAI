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
        num_virtual_stages (int): Number of virtual stages in the pipeline.
        virtual_stage (int): The current virtual stage.
    """

    def __init__(self, pg_mesh: ProcessGroupMesh, pipeline_axis: int) -> None:
        self.pg_mesh = pg_mesh
        self.pipeline_axis = pipeline_axis
        self.num_virtual_stages: Optional[int] = None
        self.virtual_stage: Optional[int] = None
        self.prev_rank: Optional[Tuple[int, ...]] = None
        self.next_rank: Optional[Tuple[int, ...]] = None
        self.p2p_groups: Dict[Tuple[int, int], ProcessGroup] = {}
        # init prev and next coord
        coord = self.pg_mesh.coordinate()
        if self.stage > 0:
            prev_coord = coord[: self.pipeline_axis] + \
                (coord[self.pipeline_axis] - 1,) + coord[self.pipeline_axis + 1:]
            self.prev_rank = self.pg_mesh.ravel(prev_coord, self.pg_mesh.shape)
        if self.stage < self.num_stages - 1:
            next_coord = coord[: self.pipeline_axis] + \
                (coord[self.pipeline_axis] + 1,) + coord[self.pipeline_axis + 1:]
            self.next_rank = self.pg_mesh.ravel(next_coord, self.pg_mesh.shape)

        # init p2p process groups
        stages = list(range(self.num_stages))
        for prev, cur in zip(stages[:-1], stages[1:]):
            group = self.pg_mesh.get_group_along_axis(self.pipeline_axis, [prev, cur])
            if self.stage in [prev, cur]:
                ranks_in_group = self.pg_mesh.get_ranks_in_group(group)
                self.p2p_groups[tuple(ranks_in_group)] = group

    def is_first_stage(self, virtual: bool = False) -> bool:
        """Is the current stage the first stage.

        Args:
            virtual (bool, optional): Whether to consider virtual stages. Defaults to False.

        Returns:
            bool: Whether the current stage is the first stage.
        """
        if virtual:
            assert self.num_virtual_stages is not None
            return self.virtual_stage == 0
        return self.stage == 0

    def is_last_stage(self, virtual: bool = False) -> bool:
        """Is the current stage the last stage.

        Args:
            virtual (bool, optional): Whether to consider virtual stages. Defaults to False.

        Returns:
            bool: Whether the current stage is the last stage.
        """
        if virtual:
            assert self.num_virtual_stages is not None
            return self.virtual_stage == self.num_virtual_stages - 1
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
        assert not self.is_first_stage(), "Cannot get previous rank in the first stage."
        return self.prev_rank

    def get_next_rank(self) -> int:
        """Get the rank of the next stage.

        Returns:
            int: Rank of the next stage.
        """
        assert not self.is_last_stage(), "Cannot get next rank in the last stage."
        return self.next_rank

    def set_num_virtual_stages(self, num_virtual_stages: int) -> None:
        """Set the number of virtual stages.

        Args:
            num_virtual_stages (int): Number of virtual stages.
        """
        self.num_virtual_stages = num_virtual_stages

    def set_virtual_stage(self, virtual_stage: int) -> None:
        """Set the virtual stage.

        Args:
            virtual_stage (int): Virtual stage.
        """
        self.virtual_stage = virtual_stage

    @contextmanager
    def switch_virtual_stage(self, virtual_stage: int) -> None:
        """A context manager to switch virtual stage.

        Args:
            virtual_stage (int): Target virtual stage.
        """
        old_stage = self.virtual_stage
        try:
            self.set_virtual_stage(virtual_stage)
            yield
        finally:
            self.set_virtual_stage(old_stage)

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
