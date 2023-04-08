from typing import Optional

from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaForCausalLM

from ..base import Actor

class RoBERTaActor(Actor):
    """
    RoBERTa Actor model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (RoBERTaConfig): Model config.
        checkpoint (bool): Enable gradient checkpointing.
        lora_rank (int): Rank of the low-rank approximation.
        lora_train_bias (str): LoRA bias training mode.
    """


    def __init__(self,
                 pretrained: Optional[str] = None,
                 config: Optional[RobertaConfig] = None,
                 checkpoint: bool = False,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none') -> None:
        if pretrained is not None:
            model = RobertaForCausalLM.from_pretrained(pretrained)
        elif config is not None:
            model = RobertaForCausalLM(config)
        else:
            model = RobertaForCausalLM(RobertaConfig())
        if checkpoint:
            model.gradient_checkpointing_enable()
        super().__init__(model, lora_rank, lora_train_bias)
