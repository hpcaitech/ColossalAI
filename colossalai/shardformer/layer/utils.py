from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


class Randomizer:
    """
    Randomizer enables the program to be executed under a different seed within the context.

    Example:

    ```python
    randomizer = Randomizer(seed=1024)

    with randomizer.fork():
        # do something here with seed 1024
        do_something()
    ```

    Args:
        seed (int): The random seed to set.
        enable_cpu (bool): fork the CPU RNG state as well.
        with_index (bool): whether to use the index of the randomizer.
    """

    _INDEX = 0

    def __init__(self, seed: int):
        # TODO: remove colossalai.context.random

        self.seed = seed

        # Handle CUDA rng state
        # 1. get the current rng state
        # 2. set the seed and store the rng state
        # 3. recover the original rng state
        cuda_original_rng_state = torch.cuda.get_rng_state()
        torch.cuda.manual_seed(seed)
        self.cuda_rng_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(cuda_original_rng_state)

        # to the same for cpu rng state
        cpu_original_rng_state = torch.get_rng_state()
        torch.manual_seed(seed)
        self.cpu_rng_state = torch.get_rng_state()
        torch.set_rng_state(cpu_original_rng_state)

    def _set_cuda_rng_state(self, rng_state):
        torch.cuda.set_rng_state(rng_state)

    def _get_cuda_rng_state(self):
        current_state = torch.cuda.get_rng_state()
        return current_state

    def _set_cpu_rng_state(self, rng_state):
        torch.set_rng_state(rng_state)

    def _get_cpu_rng_state(self):
        current_state = torch.get_rng_state()
        return current_state

    @contextmanager
    def fork_rng(self, enable_cpu: bool = False):
        """
        This is a context manager to change the dropout state and recover the original state.

        Usage:
        ::
            >>> with _seed_manager.dropout_mode():
            >>>     input = super().forward(input)
        """
        try:
            current_cuda_rng_state = self._get_cuda_rng_state()
            self._set_cuda_rng_state(self.cuda_rng_state)

            if enable_cpu:
                current_cpu_rng_state = self._get_cpu_rng_state()
                self._set_cpu_rng_state(self.cpu_rng_state)
            yield
        finally:
            self.cuda_rng_state = self._get_cuda_rng_state()
            self._set_cuda_rng_state(current_cuda_rng_state)

            if enable_cpu:
                self.cpu_rng_state = self._get_cpu_rng_state()
                self._set_cpu_rng_state(current_cpu_rng_state)

    @staticmethod
    def index():
        """
        Return the index of the randomizer. The index is useful when the user wants
        to introduce some randomness in the program.

        Note:
        The index will increment by one each time this method is called.

        Example:

        ```python
        # assume we need a randomizer to init the weight of different layers
        # we can use the index of the randomizer to do so that
        # each layer has its own randomizer with a different seed
        base_seed = torch.random.initial_seed()
        seed = base_seed + Randomizer.index()
        randomizer = Randomizer(seed)

        with randomizer.fork():
            init_weights()
        ```

        """
        idx = Randomizer._INDEX
        Randomizer._INDEX += 1
        return idx


def create_randomizer_with_offset(seed: int, process_group: ProcessGroup = None):
    """
    Create a randomizer with an offset. The offset is equal to the rank of the process and the index of the randomizer.

    Args:
        seed (int): The base random seed to set.
        enable_cpu (bool): fork the CPU RNG state as well.
        process_group (ProcessGroup): the process group to get the rank from.

    Returns:
        Randomizer: the randomizer with offset.
    """
    offset = Randomizer.index()

    if dist.is_initialized():
        rank = dist.get_rank(process_group)
        offset += rank

    seed += offset
    return Randomizer(seed=seed)
