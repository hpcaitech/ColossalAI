from transformers import BloomForCausalLM, LlamaForCausalLM, PretrainedConfig

from colossalai.inference.config import InferenceConfig

_SUPPORTED_MODELS = {
    "BloomForCausalLM": BloomForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "LLaMAForCausalLM": LlamaForCausalLM,  # For decapoda-research/llama-*
}


def init_model(inference_config: InferenceConfig, model_config: PretrainedConfig):
    # Get the supported model
    supported_model = None
    archs = getattr(model_config, "architectures", [])
    for arch in archs:
        if arch in _SUPPORTED_MODELS:
            supported_model = _SUPPORTED_MODELS[arch]

    assert supported_model is not None, "The input model is currently not supported."

    return supported_model.from_pretrained(inference_config.model)
