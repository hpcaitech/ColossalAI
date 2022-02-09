from .base_store import BaseStore


class BucketStore(BaseStore):

    def __init__(self):
        super().__init__()
        self._grads = []
        self._params = dict()
        self._num_elements_in_bucket = 0

    @property
    def num_elements_in_bucket(self):
        return self._num_elements_in_bucket

    @num_elements_in_bucket.setter
    def num_elements_in_bucket(self, num_elements):
        self._num_elements_in_bucket = num_elements

    def add_grad(self, tensor):
        self._grads.append(tensor)

    def add_param(self, tensor, group_id):
        if group_id not in self._params:
            self._params[group_id] = []
        self._params[group_id].append(tensor)

    def reset(self):
        self._grads = []
        self._params = dict()
        self._num_elements_in_bucket = 0

    def get_grad(self):
        return self._grads

    def get_param(self):
        return self._params
