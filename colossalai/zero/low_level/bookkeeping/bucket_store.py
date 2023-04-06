from torch.distributed import ProcessGroup

from .base_store import BaseStore


class BucketStore(BaseStore):

    def __init__(self, torch_pg: ProcessGroup):
        super().__init__(torch_pg)
        self._params = dict()
        self._num_elements_in_bucket = dict()

        self.reset()

    def num_elements_in_bucket(self, reduce_rank: int = None):
        return self._num_elements_in_bucket[reduce_rank]

    def add_num_elements_in_bucket(self, num_elements, reduce_rank: int = None):
        self._num_elements_in_bucket[reduce_rank] += num_elements

    def add_param(self, tensor, reduce_rank: int = None):
        self._params[reduce_rank].append(tensor)

    def reset(self):
        keys = [None] + list(range(self._world_size))
        self._params = {rank: [] for rank in keys}
        self._num_elements_in_bucket = {rank: 0 for rank in keys}

    def reset_by_rank(self, reduce_rank=None):
        self._params[reduce_rank] = []
        self._num_elements_in_bucket[reduce_rank] = 0

    def get_grad(self, reduce_rank: int = None):
        param_list = self.get_param(reduce_rank)
        for param in param_list:
            # the param must have grad for reduction
            assert param.grad is not None, f'Parameter of size ({param.size()}) has None grad, cannot be reduced'
        return [param.grad for param in param_list]

    def get_param(self, reduce_rank: int = None):
        return self._params[reduce_rank]
