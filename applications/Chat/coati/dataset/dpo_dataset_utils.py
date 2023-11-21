from typing import Any, Dict, List, Union

import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM


def get_log_probability(logits: torch.Tensor, labels: torch.Tensor):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def get_reference_model_reward(
    data_point: Dict[str, Any],
    model: LlamaForCausalLM,
) -> Dict[str, Union[int, str, List[int]]]:
    (
        chosen_input_ids,
        chosen_attention_mask,
        chosen_loss_mask,
        rejected_input_ids,
        rejected_attention_mask,
        rejected_loss_mask,
    ) = (
        data_point["chosen_input_ids"],
        data_point["chosen_attention_mask"],
        data_point["chosen_loss_mask"],
        data_point["rejected_input_ids"],
        data_point["rejected_attention_mask"],
        data_point["rejected_loss_mask"],
    )
    with torch.no_grad():
        current_device = torch.cuda.current_device()
        chosen_logits = model(
            input_ids=torch.tensor([chosen_input_ids]).to(current_device),
            attention_mask=torch.tensor([chosen_attention_mask]).to(current_device),
        ).logits
        chosen_logits = get_log_probability(chosen_logits, torch.tensor([chosen_input_ids]).to(current_device)).cpu()
        rejected_logits = model(
            input_ids=torch.tensor([rejected_input_ids]).to(current_device),
            attention_mask=torch.tensor([rejected_attention_mask]).to(current_device),
        ).logits
        rejected_logits = get_log_probability(
            rejected_logits, torch.tensor([rejected_input_ids]).to(current_device)
        ).cpu()

        data_point["chosen_reward"] = (chosen_logits * torch.tensor(chosen_loss_mask)).sum(-1)
        data_point["rejected_reward"] = (rejected_logits * torch.tensor(rejected_loss_mask)).sum(-1)

    return data_point
