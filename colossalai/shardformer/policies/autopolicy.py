import torch.nn as nn

def build_policies():
    """
    Build the policies for the model
    
    Return:
        The dict for the policies
    """
    auto_policy_dict = {}

    from transformers.models.bert.modeling_bert import BertForMaskedLM
    from .bert import BertForMaskedLMPolicy
    auto_policy_dict[BertForMaskedLM] = BertForMaskedLMPolicy

    from transformers.models.bert.modeling_bert import BertForSequenceClassification
    from .bert import BertForSequenceClassificationPolicy
    auto_policy_dict[BertForSequenceClassification] = BertForSequenceClassificationPolicy
    
    return auto_policy_dict

def get_autopolicy(model:nn.Module):
    """
    Return the auto policy for the model

    Args:
        model: The model to be used

    Return:
        The auto policy for the model
    """
    auto_policy_dict = build_policies()
    policy = auto_policy_dict.get(model.__class__, None)
    if policy is None:   
        raise NotImplementedError(f"Auto policy for {model.__class__.__qualname__} is not implemented\n Supported models are {[i.__qualname__ for i in auto_policy_dict.keys()]}")
    return policy

# from transformers.models.bert.modeling_bert import BertForMaskedLM, BertForPreTraining
# model = BertForPreTraining
# policy = get_autopolicy(model)
# print(policy)
