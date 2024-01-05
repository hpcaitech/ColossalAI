import gc
import itertools
from functools import reduce
from operator import mul
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch.distributed as dist
from torch.distributed import ProcessGroup


def prod(nums: List[int]) -> int:
    """Product of a list of numbers.

    Args:
        nums (List[int]): A list of numbers.

    Returns:
        int: The product of the numbers.
    """
    return reduce(mul, nums)


class ProcessGroupMesh:
    """A helper class to manage the process group mesh. It only describes how to organize process groups, and it's decoupled with parallel method.
    It just initialize process groups and cache them. The parallel method should manage them and use them to do the parallel computation.

    We use a ND-tuple to represent the process group mesh. And a ND-coordinate is to represent each process.
    For example, ``(0, 1, 0)`` represents the process whose rank is 2 in a 3D process group mesh with size ``(2, 2, 2)``.

    Args:
        *size (int): The size of each dimension of the process group mesh. The product of the size must be equal to the world size.

    Attributes:
        shape (Tuple[int, ...]): The shape of the process group mesh.
        rank (int): The rank of the current process.
    """

    def __init__(self, *size: int) -> None:
        assert dist.is_initialized(), "Please initialize torch.distributed first."
        assert prod(size) == dist.get_world_size(), "The product of the size must be equal to the world size."
        self._shape = size
        self._rank = dist.get_rank()
        self._coord = ProcessGroupMesh.unravel(self._rank, self._shape)
        self._ranks_to_group: Dict[Tuple[int, ...], ProcessGroup] = {}
        self._group_to_ranks: Dict[ProcessGroup, Tuple[int, ...]] = {}

    def destroy_mesh_process_groups(self):
        r"""
        Destructor method for the ProcessGroupMesh class.

        When the ProcessGroupMesh object is deleted or goes out of scope, this method is called. It is responsible for
        cleaning up any process groups that were created during the lifetime of the object.

        Note:
            All process groups in PyTorch are represented as global variables, and they may not be automatically destroyed
            when the ProcessGroupMesh's lifetime ends. This method manually destroys the process groups to release
            system resources.
        """
        for group in self._ranks_to_group.values():
            dist.destroy_process_group(group)

        # Manually clear all process groups to save memory
        gc.collect()

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def rank(self) -> int:
        return self._rank

    def size(self, dim: Optional[int] = None) -> Union[int, Tuple[int, ...]]:
        """Get the size of the process group mesh.

        Args:
            dim (Optional[int], optional): Dimension of the process group mesh. `None` means all dimensions. Defaults to None.

        Returns:
            Union[int, Tuple[int, ...]]: Size of the target dimension or the whole process group mesh.
        """
        if dim is None:
            return self._shape
        else:
            return self._shape[dim]

    def coordinate(self, dim: Optional[int] = None) -> Union[int, Tuple[int, ...]]:
        """Get the coordinate of the process group mesh.

        Args:
            dim (Optional[int], optional): Dimension of the process group mesh. `None` means all dimensions. Defaults to None.

        Returns:
            Union[int, Tuple[int, ...]]: Coordinate of the target dimension or the whole process group mesh.
        """
        if dim is None:
            return self._coord
        else:
            return self._coord[dim]

    @staticmethod
    def unravel(rank: int, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Convert a rank to a coordinate.

        Args:
            rank (int): Rank to be converted.
            shape (Tuple[int, ...]): Shape of the process group mesh.

        Returns:
            Tuple[int, ...]: Coordinate of the rank.
        """
        return np.unravel_index(rank, shape)

    @staticmethod
    def ravel(coord: Tuple[int, ...], shape: Tuple[int, ...], mode: str = "raise") -> int:
        """Convert a coordinate to a rank.
           mode: ['raise', 'wrap', 'clip'], see https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html.
           with wrap, index out of range would be wrapped around.
           For instance, ravel((0, i, 0), (1, 2, 1), 'wrap') returns (i % 2)

        Args:
            coords (Tuple[int, ...]): Coordinate to be converted.
            shape (Tuple[int, ...]): Shape of the process group mesh.
            mode (Optional[str]): The mode for numpy.ravel_multi_index.

        Returns:
            int: Rank of the coordinate.
        """

        assert mode in ["raise", "wrap", "clip"]
        return np.ravel_multi_index(coord, shape, mode)

    def get_group(self, ranks_in_group: List[int], backend: Optional[str] = None) -> ProcessGroup:
        """Get the process group with the given ranks. It the process group doesn't exist, it will be created.

        Args:
            ranks_in_group (List[int]): Ranks in the process group.
            backend (Optional[str], optional): Backend of the process group. Defaults to None.

        Returns:
            ProcessGroup: The process group with the given ranks.
        """
        ranks_in_group = sorted(ranks_in_group)
        if tuple(ranks_in_group) not in self._group_to_ranks:
            group = dist.new_group(ranks_in_group, backend=backend)
            self._ranks_to_group[tuple(ranks_in_group)] = group
            self._group_to_ranks[group] = tuple(ranks_in_group)
        return self._ranks_to_group[tuple(ranks_in_group)]

    def get_ranks_in_group(self, group: ProcessGroup) -> List[int]:
        """Get the ranks in the given process group. The process group must be created by this class.

        Args:
            group (ProcessGroup): The process group.

        Returns:
            List[int]: Ranks in the process group.
        """
        return list(self._group_to_ranks[group])

    @staticmethod
    def get_coords_along_axis(
        base_coord: Tuple[int, ...], axis: int, indices_at_axis: List[int]
    ) -> List[Tuple[int, ...]]:
        """Get coordinates along the given axis.

        Args:
            base_coord (Tuple[int, ...]): Base coordinate which the coordinates along the axis are based on.
            axis (int): Axis along which the coordinates are generated.
            indices_at_axis (List[int]): Indices at the axis.

        Returns:
            List[Tuple[int, ...]]: Coordinates along the axis.
        """
        coords_in_group = []
        for idx in indices_at_axis:
            coords_in_group.append(base_coord[:axis] + (idx,) + base_coord[axis + 1 :])
        return coords_in_group

    def create_group_along_axis(
        self, axis: int, indices_at_axis: Optional[List[int]] = None, backend: Optional[str] = None
    ) -> ProcessGroup:
        """Create all process groups along the given axis, and return the one which the current process belongs to.

        Args:
            axis (int): Axis along which the process groups are created.
            indices_at_axis (Optional[List[int]], optional): Indices at the axis. Defaults to None.
            backend (Optional[str], optional): Backend of the process group. Defaults to None.

        Returns:
            ProcessGroup: The process group along the given axis which the current process belongs to.
        """
        indices_at_axis = indices_at_axis or list(range(self._shape[axis]))
        reduced_shape = list(self._shape)
        # the choices on the axis are reduced to 1, since it's determined by `indices_at_axis`
        reduced_shape[axis] = 1
        target_group = None
        # use Cartesian product to generate all combinations of coordinates
        for base_coord in itertools.product(*[range(s) for s in reduced_shape]):
            coords_in_group = ProcessGroupMesh.get_coords_along_axis(base_coord, axis, indices_at_axis)
            ranks_in_group = tuple([ProcessGroupMesh.ravel(coord, self._shape) for coord in coords_in_group])
            group = self.get_group(ranks_in_group, backend=backend)
            if self._rank in ranks_in_group:
                target_group = group
        return target_group

    def get_group_along_axis(
        self, axis: int, indices_at_axis: Optional[List[int]] = None, backend: Optional[str] = None
    ) -> ProcessGroup:
        """Get the process group along the given axis which the current process belongs to. If the process group doesn't exist, it will be created.

        Args:
            axis (int): Axis along which the process groups are created.
            indices_at_axis (Optional[List[int]], optional): Indices at the axis. Defaults to None.
            backend (Optional[str], optional): Backend of the process group. Defaults to None.

        Returns:
            ProcessGroup: The process group along the given axis which the current process belongs to.
        """
        indices_at_axis = indices_at_axis or list(range(self._shape[axis]))
        coords_in_group = ProcessGroupMesh.get_coords_along_axis(self._coord, axis, indices_at_axis)
        ranks_in_group = tuple([ProcessGroupMesh.ravel(coord, self._shape) for coord in coords_in_group])
        if ranks_in_group not in self._ranks_to_group:
            # no need to cache it explicitly, since it will be cached in `create_group_along_axis`
            return self.create_group_along_axis(axis, indices_at_axis, backend=backend)
        return self._ranks_to_group[ranks_in_group]
