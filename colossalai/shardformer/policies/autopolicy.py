import torch.nn as nn


def build_policies():
    r"""
    Build the policies for the model

    Return:
        The dict for the policies
    """
    auto_policy_dict = {}

    from transformers import BertForMaskedLM

    from .bert import BertForMaskedLMPolicy
    auto_policy_dict[BertForMaskedLM] = BertForMaskedLMPolicy

    from transformers import BertForSequenceClassification

    from .bert import BertForSequenceClassificationPolicy
    auto_policy_dict[BertForSequenceClassification] = BertForSequenceClassificationPolicy
    from transformers.models.llama.modeling_llama import LlamaModel

    from .llama import LlamaPolicy
    auto_policy_dict[LlamaModel] = LlamaPolicy

    from transformers import LlamaForSequenceClassification

    from .llama import LlamaForSequenceClassificationPolicy
    auto_policy_dict[LlamaForSequenceClassification] = LlamaForSequenceClassificationPolicy

    from transformers import LlamaForCausalLM

    from .llama import LlamaForCausalLMPolicy
    auto_policy_dict[LlamaForCausalLM] = LlamaForCausalLMPolicy

    from transformers import GPT2Model

    from .gpt2 import GPT2Policy
    auto_policy_dict[GPT2Model] = GPT2Policy

    from transformers import GPT2LMHeadModel

    from .gpt2 import GPT2LMHeadModelPolicy
    auto_policy_dict[GPT2LMHeadModel] = GPT2LMHeadModelPolicy

    return auto_policy_dict


def get_autopolicy(model: nn.Module):
    r"""
    Return the auto policy for the model

    Args:
        model (:class:`nn.Module`): The model to get the auto policy

    Return:
        :class:`Policy`: The auto policy for the model
    """
    auto_policy_dict = build_policies()
    policy = auto_policy_dict.get(model.__class__, None)
    if policy is None:
        raise NotImplementedError(
            f"Auto policy for {model.__class__.__qualname__} is not implemented\n Supported models are {[i.__qualname__ for i in auto_policy_dict.keys()]}"
        )
    return policy


# from transformers.models.bert.modeling_bert import BertForMaskedLM, BertForPreTraining
# model = BertForPreTraining
# policy = get_autopolicy(model)
# print(policy)
