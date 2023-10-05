import torch
import torch.nn.functional as F
from coati.models.base import Actor, Critic, RewardModel
from coati.models.generation import generate
from coati.models.utils import calc_action_log_probs, compute_reward
from transformers import PreTrainedTokenizer

from .base import Experience, ExperienceMaker


class NaiveExperienceMaker(ExperienceMaker):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        reward_model: RewardModel,
        initial_model: Actor,
        tokenizer: PreTrainedTokenizer,
        rm_model_tokenizer: PreTrainedTokenizer,
        kl_coef: float = 0.1,
    ) -> None:
        super().__init__(actor, critic, reward_model, initial_model)
        self.tokenizer = tokenizer
        self.rm_model_tokenizer = rm_model_tokenizer
        self.kl_coef = kl_coef

    @torch.no_grad()
    def make_experience(self, input_ids: torch.Tensor, **generate_kwargs) -> Experience:
        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        self.reward_model.eval()

        # generate sequences
        
        sequences = generate(self.actor, input_ids, self.tokenizer, **generate_kwargs)

        self.actor.train()
        self.critic.train()

        # calculate auxiliary tensors
        attention_mask = None
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is not None:
            attention_mask = sequences.not_equal(pad_token_id).to(dtype=torch.long, device=sequences.device)

        input_len = input_ids.size(1)
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            action_mask = torch.ones_like(sequences, dtype=torch.bool)
        else:
            # left padding may be applied, only mask action
            action_mask = (sequences[:, input_len:] == eos_token_id).cumsum(dim=-1) == 0
            action_mask = F.pad(action_mask, (1 + input_len, -1), value=True)  # include eos token and input
        action_mask[:, :input_len] = False
        action_mask = action_mask[:, 1:]
        action_mask = action_mask[:, -(sequences.size(1) - input_len) :]
        num_actions = action_mask.size(1)

        with torch.no_grad():
            actor_output = self.actor(sequences, attention_mask)["logits"]
            action_log_probs = calc_action_log_probs(actor_output, sequences, num_actions)

            self.actor.eval()
            actor_output_kl = self.actor(sequences, attention_mask)["logits"]
            action_log_probs_kl = calc_action_log_probs(actor_output_kl, sequences, num_actions)
            self.actor.train()

            base_model_output = self.initial_model(sequences, attention_mask)["logits"]
        
            base_action_log_probs = calc_action_log_probs(base_model_output, sequences, num_actions)
            value = self.critic(sequences, attention_mask)
            sequences_text = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
            sequences_rm = self.rm_model_tokenizer(
                sequences_text, return_tensors="pt", padding="max_length", truncation=True, max_length=300
            )
            r = self.reward_model(**{'input_ids':sequences_rm['input_ids'].to(dtype=torch.long, device=sequences.device), 
                                  'attention_mask':sequences_rm['attention_mask'].to(device=sequences.device)}).logits.squeeze(-1)
        # torch.set_printoptions(threshold=10_000)
        reward = compute_reward(r, self.kl_coef, action_log_probs_kl, base_action_log_probs, action_mask=action_mask)

        advantage = 0
        advantages = []
        value = value[:,-num_actions:] * action_mask
        for t in range(num_actions-1, -1, -1):
            q_next = value[:, t+1] if t!=num_actions-1 else 0.
            advantage = 1.0 * (reward[:, t]+ 1.0 * q_next - value[:, t]) + 0.95 * advantage
            advantages.append(advantage)
        advantages = torch.stack(advantages[::-1], dim=1)
        advantages = advantages.detach()
        value = value.detach()
        r = r.detach()

        return Experience(sequences, action_log_probs, value, r, advantages, attention_mask, action_mask)
