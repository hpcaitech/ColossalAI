from torch import nn

from colossalai.inference.config import RPC_PARAM
from colossalai.inference.modeling.models.diffusion import DiffusionPipe
from colossalai.inference.modeling.models.pixart_alpha import pixart_alpha_forward
from colossalai.shardformer.policies.base_policy import Policy


class PixArtAlphaInferPolicy(Policy, RPC_PARAM):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        policy = {}
        self.append_or_create_method_replacement(
            description={"forward": pixart_alpha_forward}, policy=policy, target_key=DiffusionPipe
        )
        return policy

    def preprocess(self) -> nn.Module:
        return self.model

    def postprocess(self):
        return self.model

    def config_sanity_check(self):
        pass

    def to_rpc_param(self) -> str:
        return __class__.__name__

    @staticmethod
    def from_rpc_param() -> "PixArtAlphaInferPolicy":
        return PixArtAlphaInferPolicy()
