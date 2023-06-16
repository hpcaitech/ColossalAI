from dataclasses import dataclass

from colossalai.cluster.dist_coordinator import DistCoordinator

__all__ = ['ShardConfig']


@dataclass
class ShardConfig:
    r"""
    The config for sharding the huggingface model

    Args:
        data_parallel_size (int): The size of data parallel
        tensor_parallel_size (int): The size of tensor parallel
        pipeline_parallel_size (int): The size of pipeline parallel
        tensor_parallel_mode (List): The mode of tensor parallel, choose from `['1d','2d','2.5d','3d']
        inference_only (bool): Whether to use the inference only mode, when setting to `True`, the model
            will not calculate the loss and just return the output.
        gather_output (bool): Whether to gather the output of the model of the last layer
    """
    tensor_parallel_size: int

    # TODO: add support for tensor parallel
    # pipeline_parallel_size: int
    # data_parallel_size: int
    # tensor_parallel_mode: Literal['1d', '2d', '2.5d', '3d']
    # inference_only: bool = True
    # gather_output: bool = True

    def __post_init__(self):
        coordinator = DistCoordinator()

        # ensure the parallel size can match the world size
        world_size = coordinator.world_size
        self.data_parallel_size = world_size // self.tensor_parallel_size
        assert world_size == self.data_parallel_size * self.tensor_parallel_size, \
        f"The world size ({world_size}) should be divisible by the data parallel size {self.data_parallel_size} and tensor parallel size {self.tensor_parallel_size}"
