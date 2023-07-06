from typing import Dict, Union

import torch.nn as nn

from colossalai.shardformer.layer import DropoutForReplicatedInput, DropoutForParallelInput, FusedLayerNorm, Linear1D_Col, Linear1D_Row

from .basepolicy import ModulePolicyDescription, Policy, SubModuleReplacementDescription



