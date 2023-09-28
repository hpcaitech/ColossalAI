from typing import Optional


class TensorParallelEnv(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, *args, **kwargs):
        self.load(*args, **kwargs)

    def load(
        self,
        mode: Optional[str] = None,
        vocab_parallel: bool = False,
        parallel_input_1d: bool = False,
        summa_dim: int = None,
        tesseract_dim: int = None,
        tesseract_dep: int = None,
        depth_3d: int = None,
        input_group_3d=None,
        weight_group_3d=None,
        output_group_3d=None,
        input_x_weight_group_3d=None,
        output_x_weight_group_3d=None,
    ):
        self.mode = mode
        self.vocab_parallel = vocab_parallel
        self.parallel_input_1d = parallel_input_1d
        self.summa_dim = summa_dim
        self.tesseract_dim = tesseract_dim
        self.tesseract_dep = tesseract_dep
        self.depth_3d = depth_3d
        self.input_group_3d = input_group_3d
        self.weight_group_3d = weight_group_3d
        self.output_group_3d = output_group_3d
        self.input_x_weight_group_3d = input_x_weight_group_3d
        self.output_x_weight_group_3d = output_x_weight_group_3d

    def save(self):
        return dict(
            mode=self.mode,
            vocab_parallel=self.vocab_parallel,
            parallel_input_1d=self.parallel_input_1d,
            summa_dim=self.summa_dim,
            tesseract_dim=self.tesseract_dim,
            tesseract_dep=self.tesseract_dep,
            depth_3d=self.depth_3d,
            input_group_3d=self.input_group_3d,
            weight_group_3d=self.weight_group_3d,
            output_group_3d=self.output_group_3d,
            input_x_weight_group_3d=self.input_x_weight_group_3d,
            output_x_weight_group_3d=self.output_x_weight_group_3d,
        )


tensor_parallel_env = TensorParallelEnv()
