import importlib
from dataclasses import dataclass

import torch.nn as nn

from .base_policy import Policy

__all__ = ["PolicyLocation", "get_autopolicy", "import_policy"]


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
    "transformers.models.bert.modeling_bert.BertModel": PolicyLocation(file_name="bert", class_name="BertModelPolicy"),
    "transformers.models.bert.modeling_bert.BertForPreTraining": PolicyLocation(
        file_name="bert", class_name="BertForPreTrainingPolicy"
    ),
    "transformers.models.bert.modeling_bert.BertLMHeadModel": PolicyLocation(
        file_name="bert", class_name="BertLMHeadModelPolicy"
    ),
    "transformers.models.bert.modeling_bert.BertForMaskedLM": PolicyLocation(
        file_name="bert", class_name="BertForMaskedLMPolicy"
    ),
    "transformers.models.bert.modeling_bert.BertForSequenceClassification": PolicyLocation(
        file_name="bert", class_name="BertForSequenceClassificationPolicy"
    ),
    "transformers.models.bert.modeling_bert.BertForTokenClassification": PolicyLocation(
        file_name="bert", class_name="BertForTokenClassificationPolicy"
    ),
    "transformers.models.bert.modeling_bert.BertForNextSentencePrediction": PolicyLocation(
        file_name="bert", class_name="BertForNextSentencePredictionPolicy"
    ),
    "transformers.models.bert.modeling_bert.BertForMultipleChoice": PolicyLocation(
        file_name="bert", class_name="BertForMultipleChoicePolicy"
    ),
    "transformers.models.bert.modeling_bert.BertForQuestionAnswering": PolicyLocation(
        file_name="bert", class_name="BertForQuestionAnsweringPolicy"
    ),
    # LLaMA
    "transformers.models.llama.modeling_llama.LlamaModel": PolicyLocation(
        file_name="llama", class_name="LlamaModelPolicy"
    ),
    "transformers.models.llama.modeling_llama.LlamaForCausalLM": PolicyLocation(
        file_name="llama", class_name="LlamaForCausalLMPolicy"
    ),
    "transformers.models.llama.modeling_llama.LlamaForSequenceClassification": PolicyLocation(
        file_name="llama", class_name="LlamaForSequenceClassificationPolicy"
    ),
    # T5
    "transformers.models.t5.modeling_t5.T5Model": PolicyLocation(file_name="t5", class_name="T5ModelPolicy"),
    "transformers.models.t5.modeling_t5.T5ForConditionalGeneration": PolicyLocation(
        file_name="t5", class_name="T5ForConditionalGenerationPolicy"
    ),
    "transformers.models.t5.modeling_t5.T5EncoderModel": PolicyLocation(file_name="t5", class_name="T5EncoderPolicy"),
    "transformers.models.t5.modeling_t5.T5ForTokenClassification": PolicyLocation(
        file_name="t5", class_name="T5ForTokenClassificationPolicy"
    ),
    # GPT2
    "transformers.models.gpt2.modeling_gpt2.GPT2Model": PolicyLocation(file_name="gpt2", class_name="GPT2ModelPolicy"),
    "transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel": PolicyLocation(
        file_name="gpt2", class_name="GPT2LMHeadModelPolicy"
    ),
    "transformers.models.gpt2.modeling_gpt2.GPT2DoubleHeadsModel": PolicyLocation(
        file_name="gpt2", class_name="GPT2DoubleHeadsModelPolicy"
    ),
    "transformers.models.gpt2.modeling_gpt2.GPT2ForQuestionAnswering": PolicyLocation(
        file_name="gpt2", class_name="GPT2ForQuestionAnsweringPolicy"
    ),
    "transformers.models.gpt2.modeling_gpt2.GPT2ForTokenClassification": PolicyLocation(
        file_name="gpt2", class_name="GPT2ForTokenClassificationPolicy"
    ),
    "transformers.models.gpt2.modeling_gpt2.GPT2ForSequenceClassification": PolicyLocation(
        file_name="gpt2", class_name="GPT2ForSequenceClassificationPolicy"
    ),
    # GPTJ
    "transformers.models.gptj.modeling_gptj.GPTJModel": PolicyLocation(file_name="gptj", class_name="GPTJModelPolicy"),
    "transformers.models.gptj.modeling_gptj.GPTJForCausalLM": PolicyLocation(
        file_name="gptj", class_name="GPTJForCausalLMPolicy"
    ),
    "transformers.models.gptj.modeling_gptj.GPTJForQuestionAnswering": PolicyLocation(
        file_name="gptj", class_name="GPTJForQuestionAnsweringPolicy"
    ),
    "transformers.models.gptj.modeling_gptj.GPTJForSequenceClassification": PolicyLocation(
        file_name="gptj", class_name="GPTJForSequenceClassificationPolicy"
    ),
    # ViT
    "transformers.models.vit.modeling_vit.ViTModel": PolicyLocation(file_name="vit", class_name="ViTModelPolicy"),
    "transformers.models.vit.modeling_vit.ViTForImageClassification": PolicyLocation(
        file_name="vit", class_name="ViTForImageClassificationPolicy"
    ),
    "transformers.models.vit.modeling_vit.ViTForMaskedImageModeling": PolicyLocation(
        file_name="vit", class_name="ViTForMaskedImageModelingPolicy"
    ),
    # OPT
    "transformers.models.opt.modeling_opt.OPTModel": PolicyLocation(file_name="opt", class_name="OPTModelPolicy"),
    "transformers.models.opt.modeling_opt.OPTForCausalLM": PolicyLocation(
        file_name="opt", class_name="OPTForCausalLMPolicy"
    ),
    "transformers.models.opt.modeling_opt.OPTForSequenceClassification": PolicyLocation(
        file_name="opt", class_name="OPTForSequenceClassificationPolicy"
    ),
    "transformers.models.opt.modeling_opt.OPTForQuestionAnswering": PolicyLocation(
        file_name="opt", class_name="OPTForQuestionAnsweringPolicy"
    ),
    # Bloom
    "transformers.models.bloom.modeling_bloom.BloomModel": PolicyLocation(
        file_name="bloom", class_name="BloomModelPolicy"
    ),
    "transformers.models.bloom.modeling_bloom.BloomForCausalLM": PolicyLocation(
        file_name="bloom", class_name="BloomForCausalLMPolicy"
    ),
    "transformers.models.bloom.modeling_bloom.BloomForSequenceClassification": PolicyLocation(
        file_name="bloom", class_name="BloomForSequenceClassificationPolicy"
    ),
    "transformers.models.bloom.modeling_bloom.BloomForTokenClassification": PolicyLocation(
        file_name="bloom", class_name="BloomForTokenClassificationPolicy"
    ),
    "transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering": PolicyLocation(
        file_name="bloom", class_name="BloomForQuestionAnsweringPolicy"
    ),
    # Whisper
    "transformers.models.whisper.modeling_whisper.WhisperModel": PolicyLocation(
        file_name="whisper", class_name="WhisperModelPolicy"
    ),
    "transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration": PolicyLocation(
        file_name="whisper", class_name="WhisperForConditionalGenerationPolicy"
    ),
    "transformers.models.whisper.modeling_whisper.WhisperForAudioClassification": PolicyLocation(
        file_name="whisper", class_name="WhisperForAudioClassificationPolicy"
    ),
    # Sam
    "transformers.models.sam.modeling_sam.SamModel": PolicyLocation(file_name="sam", class_name="SamModelPolicy"),
    # Blip2
    "transformers.models.blip_2.modeling_blip_2.Blip2Model": PolicyLocation(
        file_name="blip2", class_name="Blip2ModelPolicy"
    ),
    "transformers.models.blip_2.modeling_blip_2.Blip2ForConditionalGeneration": PolicyLocation(
        file_name="blip2", class_name="Blip2ForConditionalGenerationPolicy"
    ),
    # ChatGLM
    "transformers_modules.modeling_chatglm.ChatGLMModel": PolicyLocation(
        file_name="chatglm2", class_name="ChatGLMModelPolicy"
    ),
    "transformers_modules.modeling_chatglm.ChatGLMForConditionalGeneration": PolicyLocation(
        file_name="chatglm2", class_name="ChatGLMForConditionalGenerationPolicy"
    ),
    # Deepseek
    "transformers_modules.modeling_deepseek.DeepseekModel": PolicyLocation(
        file_name="deepseek", class_name="DeepseekModelPolicy"
    ),
    "transformers_modules.modeling_deepseek.DeepseekForCausalLM": PolicyLocation(
        file_name="deepseek", class_name="DeepseekForCausalLMPolicy"
    ),
    # DeepseekV3
    "transformers_modules.modeling_deepseek.DeepseekV3Model": PolicyLocation(
        file_name="deepseek_v3", class_name="DeepseekV3ModelPolicy"
    ),
    "transformers_modules.modeling_deepseek.DeepseekV3ForCausalLM": PolicyLocation(
        file_name="deepseek_v3", class_name="DeepseekV3ForCausalLMPolicy"
    ),
    # Falcon
    "transformers.models.falcon.modeling_falcon.FalconModel": PolicyLocation(
        file_name="falcon", class_name="FalconModelPolicy"
    ),
    "transformers.models.falcon.modeling_falcon.FalconForCausalLM": PolicyLocation(
        file_name="falcon", class_name="FalconForCausalLMPolicy"
    ),
    "transformers.models.falcon.modeling_falcon.FalconForSequenceClassification": PolicyLocation(
        file_name="falcon", class_name="FalconForSequenceClassificationPolicy"
    ),
    "transformers.models.falcon.modeling_falcon.FalconForTokenClassification": PolicyLocation(
        file_name="falcon", class_name="FalconForTokenClassificationPolicy"
    ),
    "transformers.models.falcon.modeling_falcon.FalconForQuestionAnswering": PolicyLocation(
        file_name="falcon", class_name="FalconForQuestionAnsweringPolicy"
    ),
    # mistral
    "transformers.models.mistral.modeling_mistral.MistralModel": PolicyLocation(
        file_name="mistral", class_name="MistralModelPolicy"
    ),
    "transformers.models.mistral.modeling_mistral.MistralForCausalLM": PolicyLocation(
        file_name="mistral", class_name="MistralForCausalLMPolicy"
    ),
    "transformers.models.mistral.modeling_mistral.MistralForSequenceClassification": PolicyLocation(
        file_name="mistral", class_name="MistralForSequenceClassificationPolicy"
    ),
    # mixtral
    "transformers.models.mixtral.modeling_mixtral.MixtralModel": PolicyLocation(
        file_name="mixtral", class_name="MixtralModelPolicy"
    ),
    "transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM": PolicyLocation(
        file_name="mixtral", class_name="MixtralForCausalLMPolicy"
    ),
    "transformers.models.mixtral.modeling_mixtral.MixtralForSequenceClassification": PolicyLocation(
        file_name="mixtral", class_name="MixtralForSequenceClassificationPolicy"
    ),
    # Qwen2
    "transformers.models.qwen2.modeling_qwen2.Qwen2Model": PolicyLocation(
        file_name="qwen2", class_name="Qwen2ModelPolicy"
    ),
    "transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM": PolicyLocation(
        file_name="qwen2", class_name="Qwen2ForCausalLMPolicy"
    ),
    "transformers.models.qwen2.modeling_qwen2.Qwen2ForSequenceClassification": PolicyLocation(
        file_name="qwen2", class_name="Qwen2ForSequenceClassificationPolicy"
    ),
    # command
    "transformers.models.cohere.modeling_cohere.CohereModel": PolicyLocation(
        file_name="command", class_name="CommandModelPolicy"
    ),
    "transformers.models.cohere.modeling_cohere.CohereForCausalLM": PolicyLocation(
        file_name="command", class_name="CommandForCausalLMPolicy"
    ),
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
    if module == "builtins":
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    # patch custom models which are not in transformers
    # it can be like 'transformers_modules.THUDM.chatglm3-6b.103caa40027ebfd8450289ca2f278eac4ff26405.modeling_chatglm' (from huggingface hub)
    # or like 'transformers_modules.chatglm.modeling_chatglm' (from local directory)
    if module.startswith("peft"):
        klass = obj.base_model.model.__class__
        module = klass.__module__
    if module.startswith("transformers_modules"):
        split_module = module.split(".")
        if len(split_module) >= 2:
            module = f"{split_module[0]}.{split_module[-1]}"
    return module + "." + klass.__qualname__


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
            f"Auto policy for {model.__class__.__qualname__} ({full_name}) is not implemented\n. Supported models are {list(_POLICY_LIST.keys())}"
        )
    else:
        policy = import_policy(policy_location)
    return policy()
