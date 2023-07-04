from typing import Any, Dict, List, Optional, Tuple, Type

from torch import Tensor
from torch.nn import Module, Parameter

from colossalai.pipeline.stage_manager import PipelineStageManager

from .base import Policy
from .bert import BertModel, BertModelPolicy

POLICY_MAP: Dict[Type[Module], Type[Policy]] = {
    BertModel: BertModelPolicy,
}


def pipeline_parallelize(
        model: Module,
        stage_manager: PipelineStageManager) -> Tuple[Dict[str, Parameter], Dict[str, Tensor], List[Dict[int, Tensor]]]:
    if type(model) not in POLICY_MAP:
        raise NotImplementedError(f"Policy for {type(model)} not implemented")
    policy = POLICY_MAP[type(model)](stage_manager)
    return policy.parallelize_model(model)
