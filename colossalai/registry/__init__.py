import torch.distributed.optim as dist_optim
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tv_models
from torchvision.transforms import transforms

from .registry import Registry

LAYERS = Registry('layers', third_party_library=[nn])
LOSSES = Registry('losses')
MODELS = Registry('models', third_party_library=[tv_models])
OPTIMIZERS = Registry('optimizers', third_party_library=[optim, dist_optim])
OPTIMIZER_WRAPPERS = Registry('optimizer_wrappers')
DATASETS = Registry('datasets')
DIST_GROUP_INITIALIZER = Registry('dist_group_initializer')
GRADIENT_HANDLER = Registry('gradient_handler')
LOSSES = Registry('losses', third_party_library=[nn])
HOOKS = Registry('hooks')
TRANSFORMS = Registry('transforms', third_party_library=[transforms])
PIPE_ALLOC_POLICY = Registry('pipeline_allocation_policy')
SAMPLERS = Registry('samplers')
LR_SCHEDULERS = Registry('lr_schedulers')
SCHEDULE = Registry('schedules')
