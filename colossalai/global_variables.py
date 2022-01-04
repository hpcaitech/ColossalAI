

class MoeEnv:
    """Moe enviroment variable.
    """

    def __init__(self):
        self.data_parallel_size = None
        self.tensor_parallel_size = None
        self.aux_loss = None

    def setup(self, moe_tensor_size):
        from .core import global_context as gpc
        if gpc.tensor_parallel_size > 1 or gpc.pipeline_parallel_size > 1:
            raise NotImplementedError("Moe is not compatible with tensor or pipeline parallel")

        assert gpc.data_parallel_size % moe_tensor_size == 0, \
            "The size of data parallel needs to be divided by moe tensor size"

        self.data_parallel_size = gpc.data_parallel_size // moe_tensor_size
        self.tensor_parallel_size = moe_tensor_size

    def is_initialized(self):
        return self.tensor_parallel_size is not None

    def reset_loss(self):
        self.aux_loss = 0

    def add_loss(self, loss):
        self.aux_loss += loss

    def get_loss(self):
        return self.aux_loss


moe_env = MoeEnv()
