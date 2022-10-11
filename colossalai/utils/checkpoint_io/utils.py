from typing import List, Optional, Dict, Any
from torch import Tensor
from torch.optim import Optimizer


def get_param_to_os(model_state_dict: Dict[str, Tensor], optimizer: Optimizer) -> Dict[str, int]:
    # ensure all params in optimizer are in model state dict
    params_set = set(id(p) for p in model_state_dict.values())
    for p in optimizer.param_groups['params']:
        assert id(p) in params_set
    param_mappings = {}
    start_index = 0

    def get_group_mapping(group):
        nonlocal start_index
        param_mappings.update(
            {id(p): i for i, p in enumerate(group['params'], start_index) if id(p) not in param_mappings})
        start_index += len(group['params'])

    for g in optimizer.param_groups:
        get_group_mapping(g)
    return {k: param_mappings[id(p)] for k, p in model_state_dict.items()}


def compute_optimizer_state_size(state: Dict[str, Any]) -> int:
    size = 0
    for v in state.values():
        if isinstance(v, Tensor):
            size += v.numel() * v.element_size()
    return size


def shard_checkpoint(max_shard_size: int,
                     model_state_dict: Dict[str, Tensor],
                     optimizer_state_dict: Optional[dict] = None,
                     param_to_os: Optional[dict] = None) -> List[dict]:
    has_optimizer: bool = False
    if optimizer_state_dict is not None:
        assert param_to_os is not None
        os_to_param = {v: k for k, v in param_to_os.items()}
        for os_key in optimizer_state_dict['state'].keys():
            assert os_key in os_to_param
            assert os_to_param[os_key] in model_state_dict
        has_optimizer = True
    shards = []
    buffer = {'model': {}}
    if has_optimizer:
        buffer['optimizer'] = {'state': {}, 'param_groups': optimizer_state_dict['param_groups']}
    buffer_size = 0
    for k, tensor in model_state_dict.items():
        if buffer_size >= max_shard_size:
            shards.append(buffer)
            buffer = {'model': {}}
            if has_optimizer:
                buffer['optimizer'] = {'state': {}}
            buffer_size = 0
        buffer['model'][k] = tensor
        buffer_size += tensor.numel() * tensor.element_size()
        if has_optimizer:
            buffer['optimizer']['state'][param_to_os[k]] = optimizer_state_dict['state'][param_to_os[k]]
            buffer_size += compute_optimizer_state_size(optimizer_state_dict['state'][param_to_os[k]])
    if len(buffer['model']) > 0:
        shards.append(buffer)
    return shards
