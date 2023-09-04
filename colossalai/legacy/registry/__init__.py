import torch.distributed.optim as dist_optim
import torch.nn as nn
import torch.optim as optim

from .registry import Registry

LAYERS = Registry("layers", third_party_library=[nn])
MODELS = Registry("models")
OPTIMIZERS = Registry("optimizers", third_party_library=[optim, dist_optim])
DATASETS = Registry("datasets")
DIST_GROUP_INITIALIZER = Registry("dist_group_initializer")
GRADIENT_HANDLER = Registry("gradient_handler")
LOSSES = Registry("losses", third_party_library=[nn])
HOOKS = Registry("hooks")
TRANSFORMS = Registry("transforms")
DATA_SAMPLERS = Registry("data_samplers")
LR_SCHEDULERS = Registry("lr_schedulers")
SCHEDULE = Registry("schedules")
OPHOOKS = Registry("ophooks")
