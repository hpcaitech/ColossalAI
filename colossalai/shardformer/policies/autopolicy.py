import importlib
from dataclasses import dataclass

import torch.nn as nn

from .basepolicy import Policy


@dataclass
class PolicyLocation:
    """
    PolicyLocation describes the location of a policy class.

    Args:
        file_name (str): The file name of the policy under colossalai.shardformer.policies
        class_name (str): The class name of the policy class
    """
    file_name: str
    class_name: str


# we don't want to import all policies here
# as each policy file imports its own model zoo library
# we will allow the user to only import the policy file needed
_POLICY_LIST = {
    # BERT
    "transformers.models.bert.modeling_bert.BertModel":
        PolicyLocation(file_name="bert", class_name="BertPolicy"),
    "transformers.models.bert.modeling_bert.BertForPreTraining":
        PolicyLocation(file_name="bert", class_name="BertForPretrainingPolicy"),
    "transformers.models.bert.modeling_bert.BertForMaskedLM":
        PolicyLocation(file_name="bert", class_name="BertForMaskedLMPolicy"),
    "transformers.models.bert.modeling_bert.BertLMHeadModel":
        PolicyLocation(file_name="bert", class_name="BertLMHeadModelPolicy"),
    "transformers.models.bert.modeling_bert.BertForNextSentencePrediction":
        PolicyLocation(file_name="bert", class_name="BertForNextSentencePredictionPolicy"),
    "transformers.models.bert.modeling_bert.BertForSequenceClassification":
        PolicyLocation(file_name="bert", class_name="BertForSequenceClassificationPolicy"),
    "transformers.models.bert.modeling_bert.BertForMultipleChoice":
        PolicyLocation(file_name="bert", class_name="BertForMultipleChoicePolicy"),

    # LLaMA
    "transformers.models.llama.modeling_llama.LlamaModel":
        PolicyLocation(file_name="llama", class_name="LlamaPolicy"),
    "transformers.models.llama.modeling_llama.LlamaForCausalLM":
        PolicyLocation(file_name="llama", class_name="LlamaForCausalLMPolicy"),
    "transformers.models.llama.modeling_llama.LlamaForSequenceClassification":
        PolicyLocation(file_name="llama", class_name="LlamaForSequenceClassificationPolicy"),

    # T5

    # GPT2
}


def import_policy(policy_location: PolicyLocation) -> Policy:
    """
    Dynamically import a Policy class based on the policy location.
    """
    module_name = f"colossalai.shardformer.policies.{policy_location.file_name}"
    module = importlib.import_module(module_name)
    return getattr(module, policy_location.class_name)


def _fullname(obj):
    """
    Return the full name of an object, including the module name.
    """
    klass = obj.__class__
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__    # avoid outputs like 'builtins.str'
    return module + '.' + klass.__qualname__


def get_autopolicy(model: nn.Module) -> Policy:
    r"""
    Return the auto policy for the model

    Args:
        model (:class:`nn.Module`): The model to get the auto policy

    Return:
        :class:`Policy`: The auto policy for the model
    """
    full_name = _fullname(model)
    policy_location = _POLICY_LIST.get(full_name, None)

    if policy_location is None:
        raise NotImplementedError(
            f"Auto policy for {model.__class__.__qualname__} is not implemented\n. Supported models are {list(_POLICY_LIST.keys())}"
        )
    else:
        policy = import_policy(policy_location)
    return policy()
    return policy()
