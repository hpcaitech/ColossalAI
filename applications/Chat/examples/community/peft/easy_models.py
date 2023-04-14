from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from coati.models.generation import generate
from coati.models.utils import log_probs_from_logits, masked_mean
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.nn.modules import Module
from transformers import BloomConfig, BloomForCausalLM,AutoModel
from coati.models.base.reward_model import RewardModel
import os

class ChatGLMRM(Module):
    """
    ChatGLMRM Reward model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (BloomConfig): Model config.
        checkpoint (bool): Enable gradient checkpointing.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self,
                 pretrained: str = None,
                 lora_path :str = None,
                 lora_rank :int = 0) -> None:
        super().__init__()
        if pretrained is not None:
            model = AutoModel.from_pretrained(
                pretrained,
                trust_remote_code=True,
            ).half() # load model to cpu and half 
            if lora_path is not None and os.path.exists(lora_path+'/adapter_config.json') \
            and os.path.exists(lora_path+'/adapter_model.bin'):
                print('load lora from ',lora_path)
                model = PeftModel.from_pretrained(model, lora_path).half().cpu()
                self.model = model
            else:
                #we'll use peft lora library to do the lora
                lora_rank = lora_rank if lora_rank > 0 else 32
                #config lora with rank of lora_rank
                lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                        inference_mode=False,
                                        r=lora_rank,
                                        lora_alpha=32,
                                        lora_dropout=0.1)
                model = get_peft_model(model, lora_config)
                self.model = model
        else:
            raise ValueError("No pretrained model provided!")
        value_head = nn.Linear(model.config.hidden_size, 1)
        if lora_path is not None and os.path.exists(os.path.join(lora_path,'value_head.bin')):
            print('load value_head from ',os.path.exists(os.path.join(lora_path,'value_head.bin')))
            value_head.load_state_dict(torch.load(os.path.join(lora_path,'value_head.bin')))
        else:
            value_head.weight.data.normal_(mean=0.0, std=1 / (model.config.hidden_size + 1))
        self.value_head = value_head


    def forward(self, sequences: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(sequences, attention_mask=None,return_dict=True, output_hidden_states=True)
        last_hidden_states = outputs['hidden_states'][-1]
        # print(last_hidden_states.shape)
        values = self.value_head(last_hidden_states)[:-1, :]
        # print(values.shape)
        values = values.transpose(0,1)#change from (seq,B) to (B,seq)
        value = values.mean(dim=1).squeeze(1)    # ensure shape is (B)
        # print(value.shape)
        return value
    
    def get_base_model(self):
        return self.model
    
    def save_pretrained(self,save_directory):
        self.model.save_pretrained(save_directory)
        torch.save(self.value_head.state_dict(),os.path.join(save_directory,'value_head.bin'))


class ChatGLMCritic(ChatGLMRM):

    def __init__(self, pretrained: str = None, lora_path: str = None, lora_rank: int = 0,use_action_mask: bool = True,pad_token_id :int=3) -> None:
        super().__init__(pretrained, lora_path, lora_rank)
        self.use_action_mask = use_action_mask
        self.pad_token_id = pad_token_id

    def forward(self,
                sequences: torch.LongTensor,
                action_mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(sequences, attention_mask=None,return_dict=True, output_hidden_states=True)
        last_hidden_states = outputs['hidden_states'][-1]

        values = self.value_head(last_hidden_states).squeeze(-1)
        values = values.transpose(0,1)#change from (seq,B) to (B,seq)

        if action_mask is not None and self.use_action_mask:
            num_actions = action_mask.size(1)
            #create a prompt_mask with is the same size as the values
            prompt_mask = attention_mask[:, :-num_actions]
            values = values[:, :-num_actions]
            value = masked_mean(values, prompt_mask, dim=1)
            return value

        values = values[:, :-1]
        value = values.mean(dim=1)
        return value
    
class Actor(Module):
    """
    Actor model base class.

    Args:
        model (nn.Module): Actor Model.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        return_action_mask: bool = True,
        **kwargs
    ) -> Union[Tuple[torch.LongTensor, torch.LongTensor], Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]]:
        sequences = generate(self.model, input_ids, **kwargs)
        attention_mask = None
        pad_token_id = kwargs.get('pad_token_id', None)
        if pad_token_id is not None:
            attention_mask = sequences.not_equal(pad_token_id).to(dtype=torch.long, device=sequences.device)
        if not return_action_mask:
            return sequences, attention_mask, None
        input_len = input_ids.size(1)
        eos_token_id = kwargs.get('eos_token_id', None)
        if eos_token_id is None:
            action_mask = torch.ones_like(sequences, dtype=torch.bool)
        else:
            # left padding may be applied, only mask action
            action_mask = (sequences[:, input_len:] == eos_token_id).cumsum(dim=-1) == 0
            action_mask = F.pad(action_mask, (1 + input_len, -1), value=True)    # include eos token and input
        action_mask[:, :input_len] = False
        action_mask = action_mask[:, 1:]
        return sequences, attention_mask, action_mask[:, -(sequences.size(1) - input_len):]

    def forward(self,
                sequences: torch.LongTensor,
                num_actions: int,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns action log probs
        """
        output = self.model(sequences, attention_mask=attention_mask)
        logits = output['logits']
        log_probs = log_probs_from_logits(logits[:, :-1, :], sequences[:, 1:])
        return log_probs[:, -num_actions:]#shape is (B,num_actions)

    def get_base_model(self):
        return self.model
    
    def save_pretrained(self,save_directory):
        self.model.save_pretrained(save_directory)

class ChatGlmActor(Actor):
    def __init__(self,
                 pretrained: str = None,
                 lora_path: str = None) -> None:
        if pretrained is not None:
            model = AutoModel.from_pretrained(
                pretrained,
                trust_remote_code=True,
            ).half().cpu() # load model to cpu and half precision
        if lora_path is not None:
            model = PeftModel.from_pretrained(model, lora_path).half().cpu()
        super().__init__(model)
    def forward(self,
                sequences: torch.LongTensor,
                num_actions: int,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ignor attention_mask
        """
        return super().forward(sequences, num_actions,attention_mask=None)

    def print_trainable_parameters(self):
        self.get_base_model().print_trainable_parameters()

class BLOOMActor(Actor):
    """
    BLOOM Actor model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (BloomConfig): Model config.
        checkpoint (bool): Enable gradient checkpointing.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self,
                 pretrained: str = None,
                 config: Optional[BloomConfig] = None,
                 checkpoint: bool = False,
                 lora_path: str = None) -> None:
        if pretrained is not None:
            model = BloomForCausalLM.from_pretrained(pretrained)
        elif config is not None:
            model = BloomForCausalLM(config)
        else:
            model = BloomForCausalLM(BloomConfig())
        if lora_path is not None:
            model = PeftModel.from_pretrained(model, lora_path)
        if checkpoint:
            model.gradient_checkpointing_enable()
        super().__init__(model)

    def print_trainable_parameters(self):
        self.get_base_model().print_trainable_parameters()
