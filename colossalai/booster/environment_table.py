from typing import List

__all__ = ['EnvironmentTable']


class EnvironmentTable:

    def __init__(self, intra_op_world_sizes: List[int]):
        # TODO: implement this method
        pass

    @property
    def is_master(self) -> bool:
        # TODO: implement this method
        pass

    # TODO: implement more utility methods as given in
    # https://github.com/hpcaitech/ColossalAI/issues/3051
