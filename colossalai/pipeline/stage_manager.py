import contextlib
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch.distributed as dist
from torch.distributed import ProcessGroup

from colossalai.cluster import ProcessGroupMesh


class PipelineStageManager:
    """PipelineStageManager is a helper class to manage pipeline stages.

    Args:
        pg_mesh (ProcessGroupMesh): Process group mesh.
        pipeline_axis (int): The axis along which the pipeline is constructed.
        is_virtual (bool): Whether to use circle p2p communication, it will make the first and last stage communicate with each other.

    Attributes:
        num_stages (int): Number of stages in the pipeline.
        stage (int): The current stage.
    """

    def __init__(
        self,
        pg_mesh: ProcessGroupMesh,
        pipeline_axis: int,
        enable_interleave: bool = False,
        use_zbv: bool = False,
        num_model_chunks: int = 1,
        num_layers_per_stage: Optional[List[int]] = None,
    ) -> None:
        assert enable_interleave or num_model_chunks == 1, "num_model_chunks must be 1 when enable_interleave is False"

        self.pg_mesh = pg_mesh
        self.pipeline_axis = pipeline_axis
        self.prev_rank: Optional[Tuple[int, ...]] = None
        self.next_rank: Optional[Tuple[int, ...]] = None
        self.p2p_groups: Dict[Tuple[int, ...], ProcessGroup] = {}
        if num_layers_per_stage is not None:
            assert len(num_layers_per_stage) == self.num_stages
        self.num_layers_per_stage = num_layers_per_stage

        # init prev and next coord
        coord = self.pg_mesh.coordinate()
        # the prev rank of rank0 is the last rank
        prev_coord = coord[: self.pipeline_axis] + (coord[self.pipeline_axis] - 1,) + coord[self.pipeline_axis + 1 :]
        self.prev_rank = self.pg_mesh.ravel(prev_coord, self.pg_mesh.shape, mode="wrap")
        # the next rank of the last rank is rank0
        next_coord = coord[: self.pipeline_axis] + (coord[self.pipeline_axis] + 1,) + coord[self.pipeline_axis + 1 :]
        self.next_rank = self.pg_mesh.ravel(next_coord, self.pg_mesh.shape, mode="wrap")
        self.is_interleave = enable_interleave
        self.use_zbv = use_zbv
        # for interleaved pipeline parallel, each device is responsible for multiple chunk of layers
        self.num_model_chunks: int = num_model_chunks
        # for shardformer, hold stage indices of model
        self.stage_indices: List[Tuple[int, int]]
        # for shardformer, hold model chunk id
        self.model_chunk_id: Optional[int] = None
        self.p2p_group = self.pg_mesh.get_group_along_axis(self.pipeline_axis)

    def get_stage_index(
        self,
        layers_per_stage: List[int],
        stage: Optional[int] = None,
        num_model_chunks: Optional[int] = None,
        num_stages: Optional[int] = None,
    ) -> Union[Tuple[int, int], List[Tuple[int, int]]]:
        """
        Get the start index and end index of layers for each stage.

        Args:
            layers_per_stage (List[int]): number of layers for each stage
            stage (int): the stage index
            num_stages (int): number of stages
            num_model_chunks (int): number of model chunks

        Returns:
            - Tuple[int, int]: the start index and end index of this stage
            - List[Tuple[int, int]]: the start index and end index of this stage for each model chunk

        """
        stage = self.stage if stage is None else stage
        num_model_chunks = self.num_model_chunks if num_model_chunks is None else num_model_chunks
        num_stages = self.num_stages if num_stages is None else num_stages

        num_layers_per_stage_accumulated = np.insert(np.cumsum(layers_per_stage), 0, 0)

        stage_indices = []
        if self.use_zbv:
            stage_indices.append([num_layers_per_stage_accumulated[stage], num_layers_per_stage_accumulated[stage + 1]])
            stage_indices.append(
                [
                    num_layers_per_stage_accumulated[2 * num_stages - stage - 1],
                    num_layers_per_stage_accumulated[2 * num_stages - stage],
                ]
            )
            return stage_indices

        for model_chunk in range(num_model_chunks):
            start_idx = num_layers_per_stage_accumulated[stage + model_chunk * num_stages]
            end_idx = num_layers_per_stage_accumulated[stage + model_chunk * num_stages + 1]
            stage_indices.append([start_idx, end_idx])

        return stage_indices[0] if num_model_chunks == 1 else stage_indices

    def is_first_stage(self, ignore_chunk: bool = False) -> bool:
        """Is the current stage the first stage.

        NOTE:
            1. if using interleaved pipeline parallel, the first stage is the first chunk of the first device.
            2. invoke is_first_stage() with ignore_chunk=True is equivalent to invoke is_first_device()

        Returns:
            bool: Whether the current stage is the first stage.
        """
        assert isinstance(ignore_chunk, bool)
        assert not self.is_interleave or (ignore_chunk or self.model_chunk_id is not None)
        if not self.is_interleave or ignore_chunk:
            return self.stage == 0
        else:
            return self.stage == 0 and self.model_chunk_id == 0

    def is_last_stage(self, ignore_chunk: bool = False) -> bool:
        """Is the current stage the last stage.

        NOTE:
            1. if using interleaved pipeline parallel, the last stage is the last chunk of the last device.
            2. invoke is_last_stage() with ignore_chunk=True is equivalent to invoke is_last_device()

        Returns:
            bool: Whether the current stage is the last stage.
        """
        assert isinstance(ignore_chunk, bool)
        assert not self.is_interleave or (ignore_chunk or self.model_chunk_id is not None)
        if not self.is_interleave or ignore_chunk:
            return self.stage == self.num_stages - 1
        else:
            # use zero bubble pipeline
            if self.use_zbv:
                return self.stage == 0 and self.model_chunk_id == self.num_model_chunks - 1
            else:
                return self.stage == self.num_stages - 1 and self.model_chunk_id == self.num_model_chunks - 1

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

    def get_p2p_process_group(self) -> ProcessGroup:
        """Get the p2p process group between two ranks. The order of the two ranks does not matter.
        Returns:
            ProcessGroup: P2P process group between the two ranks.
        """
        return self.p2p_group

    def init_process_group_by_stages(self, stages: List[int]) -> ProcessGroup:
        """Get the process group of the given stages.

        Args:
            stages (List[int]): List of stages.

        Returns:
            ProcessGroup: Process group of the given stages.
        """
        return self.pg_mesh.get_group_along_axis(self.pipeline_axis, stages)

    @contextlib.contextmanager
    def switch_model_chunk_id(self, model_chunk_id: int):
        old_model_chunk_id = self.model_chunk_id
        self.model_chunk_id = model_chunk_id
        yield
        self.model_chunk_id = old_model_chunk_id

    def distribute_layers(
        self, num_layers: int, num_stages: Optional[int] = None, num_model_chunks: Optional[int] = None
    ) -> List[int]:
        if self.num_layers_per_stage is not None:
            assert sum(self.num_layers_per_stage) == num_layers
            return self.num_layers_per_stage

        num_stages = self.num_stages if num_stages is None else num_stages
        num_model_chunks = self.num_model_chunks if num_model_chunks is None else num_model_chunks
        quotient = num_layers // (num_stages * num_model_chunks)
        remainder = num_layers % (num_stages * num_model_chunks)

        # calculate the num_layers per stage
        layers_per_stage = [quotient] * num_stages * num_model_chunks
        # deal with the rest layers
        if remainder > 0:
            start_position = (num_stages * num_model_chunks) // 2 - remainder // 2
            for i in range(start_position, start_position + remainder):
                layers_per_stage[i] += 1
        return layers_per_stage
