from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class TensorBucket:

    def __init__(self, size):
        self._max_size = size
        self._current_size = 0
        self._bucket = []

    @property
    def max_size(self):
        return self._max_size

    @property
    def current_size(self):
        return self._current_size

    def is_full_or_oversized(self):
        return self._current_size >= self._max_size

    def is_empty(self):
        return len(self._bucket) == 0

    def add_to_bucket(self, tensor, allow_oversize=False):
        tensor_size = tensor.numel()

        if not allow_oversize and self.will_exceed_max_size(tensor_size):
            msg = f"The param bucket max size {self._max_size} is exceeded" \
                + f"by tensor (size {tensor_size})"
            raise RuntimeError(msg)

        self._bucket.append(tensor)
        self._current_size += tensor_size

    def will_exceed_max_size(self, tensor_size):
        expected_size = self._current_size + tensor_size
        return expected_size > self._max_size

    def get_bucket(self):
        return self._bucket

    def empty(self):
        self._bucket = []
        self._size = 0

    def flatten(self):
        return _flatten_dense_tensors(self._bucket)

    def unflatten_and_copy(self, flat_tensor):
        unflattened_tensor_list = _unflatten_dense_tensors(flat_tensor, self._bucket)
        for old, new in zip(self._bucket, unflattened_tensor_list):
            old.copy_(new)
