import torch
from coati.models.utils import compute_reward, normalize, compute_approx_kl
import torch.nn.functional as F
from .base import Experience, ExperienceMaker
from typing import Any, Callable, Optional
import torch.distributed as dist
from coati.models.base import ActorCritic
import numpy as np

class MultiStepExperienceMaker(ExperienceMaker):
    """
    Multi Step experience maker.
    """
    @torch.no_grad()
    def make_experience(self, input_ids: torch.Tensor, **generate_kwargs) -> Experience:
        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        self.reward_model.eval()
        self.ac = ActorCritic(self.actor, self.critic)
        self.buffer = []
        
        self.gamma = 1
        self.lamda = 0.25
        self.kl_coef = 1/128
        in_len = input_ids.size(1)
        
        # generate the sequence and get value at the same time
        sequences, values , attention_mask, action_mask = self.ac.generate(input_ids,
                                            **generate_kwargs)
        
        # compute action log probs
        action_logits = self.actor.model(sequences)['logits']
        base_action_logits = self.initial_model.model(sequences)['logits']
        
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        base_log_probs = F.log_softmax(base_action_logits, dim=-1)
        
        # compute kl_div
        kl_list = compute_approx_kl(action_log_probs, base_log_probs)
        # clip kl
        kl_list = torch.clamp(kl_list, max=10, min=1e-4)
        
        # add eos token to the end of sequence and compute reward
        eos_tensor = torch.tensor([self.eos_token_id], device=input_ids.device).repeat(input_ids.size(0), 1)
        sequence_with_eos = torch.cat([sequences, eos_tensor], dim=-1)
        rewards = self.reward_model(sequence_with_eos)
        
        # reward clip
        rewards = torch.clamp(rewards, max=10, min=-10)    
        
        # running mean reward
        for i in range(rewards.size(0)):
            value = rewards[i]
            self.reward_count += 1
            delta = value - self.reward_mean
            self.reward_mean += delta / self.reward_count
            delta2 = value - self.reward_mean
            self.reward_M2 += delta * delta2
        
        std = self.reward_M2 / (self.reward_count - 1)
        rewards = (rewards - self.reward_mean)/std
        
        print('rewards: ', rewards)
        rewards = rewards * (1 - self.kl_coef)
        
        # get action mask
        action_mask = action_mask[:, in_len:]
        kl_list = kl_list[:, in_len:]
        
        # compute the advantages
        advantages, returns = self.compute_gae(kl_list, rewards, values, action_mask)
          
        for i in range(in_len, sequences.size(1) - 1):
            for j in range(sequences.size(0)):
                if sequences[j, i] != self.eos_token_id:
                    _state = sequences[j, :i]
                    _action_log_prob = action_log_probs[j, i]
                    _value = values[j, i-in_len]
                    _return = returns[j, i-in_len]
                    _adv = advantages[j, i-in_len]
                    _attention_mask = attention_mask[j, :i]
                    _action_mask = action_mask[j, :i-in_len]
                    exp = Experience(_state, _action_log_prob, _value, _return, _adv, _attention_mask, _action_mask)
                    self.buffer.append(exp)
        buffer = self.buffer
        return buffer

    @torch.no_grad()
    def compute_gae(self, kl_list: torch.Tensor,
                          reward: torch.Tensor,
                          values: torch.Tensor, 
                          action_mask: torch.Tensor) -> torch.Tensor:
        kl = -kl_list * action_mask * self.kl_coef
        values = values * action_mask
        T = torch.sum(values.ne(0), dim=1)
        self.total_len = sum(T)
        max_len = max(T)
        gae_values = torch.zeros_like(values)
        delta_list = torch.zeros_like(values)
        
        # add reward to kl[:, -1]
        for i in range(len(T)):
            kl[i, T[i]-1] += reward[i]
        
        # compute delta
        for t in range(max_len - 1):
            next_v = values[:,t + 1] if t + 1 < max_len else 0
            delta_list[:, t] = kl[:, t] + self.gamma * next_v - values[:, t]

        # compute gae
        gae_values[:, max_len - 1] = delta_list[:, max_len - 1]
        for t in range(max_len - 2, -1, -1):
            gae_values[:, t] = delta_list[:, t] + self.gamma * self.lamda * gae_values[:, t + 1]
        
        # compute return
        returns = gae_values + values
        return gae_values, returns