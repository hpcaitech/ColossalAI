import torch
from colossalai.utils.memory_tracer.model_data_memtracer import ModelDataTracer


def col_move_to_cpu(t: torch.Tensor):
    assert isinstance(t, torch.Tensor)
    if t.device.type == 'cpu':
        return

    ModelDataTracer().delete_tensor(t)
    t.data = t.data.cpu()


def col_modeldata_allocate(device: torch.device) -> torch.Tensor:
    pass


def col_modeldata_release(t: torch.Tensor):
    pass
