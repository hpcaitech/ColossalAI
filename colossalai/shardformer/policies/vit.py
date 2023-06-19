rom typing import Dict, Union

import torch.nn as nn

from transformers.models.vit.modeling_vit import ViTModel

from colossalai.shardformer.layer.layers import Linear1D_Col, Linear1D_Row, VocabParallelEmbedding1D

from .basepolicy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

