import torch
import transformers

from ..registry import ModelAttribute, model_zoo

# ===============================
# Register single-sentence Whisper
# ===============================


# define data gen function
def data_gen():
    # Generated from following code snippet
    #
    # from transformers import AutoFeatureExtractor, WhisperModel
    # from datasets import load_dataset

    # model = WhisperModel.from_pretrained("openai/whisper-base")
    # feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
    # ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    # inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
    # input_features = inputs.input_features
    # decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id

    input_features = torch.rand(1, 80, 3000)
    decoder_input_ids = torch.tensor([[1, 1]]) * 50258
    return dict(input_features=input_features, decoder_input_ids=decoder_input_ids)


def data_gen_for_conditional_generation():
    # labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
    #         Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
    #         or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
    #         only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
    data = data_gen()
    data["labels"] = torch.tensor([[0, 1]], dtype=torch.int64)
    return data


def data_gen_for_audio_classification():
    # labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
    #         Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
    #         config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
    #         `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    # `WhisperForAudioClassification` does not need `decoder_input_ids`
    data = data_gen()
    data.pop("decoder_input_ids")
    data["labels"] = torch.tensor([1], dtype=torch.int64)
    return data


# define output transform function
output_transform_fn = lambda x: x

# define loss funciton
loss_fn = lambda x: torch.nn.functional.mse_loss(x["last_hidden_state"], torch.ones_like(x["last_hidden_state"]))
loss_fn_attr = lambda x: x["loss"]

config = transformers.WhisperConfig(
    classifier_proj_size=256,
    d_model=256,
    decoder_attention_heads=4,
    decoder_ffn_dim=1536,
    decoder_layers=2,
    encoder_attention_heads=4,
    encoder_ffn_dim=1536,
    encoder_layers=2,
    vocab_size=51866,
)

# register the Whisper variants
model_zoo.register(
    name="transformers_whisper",
    model_fn=lambda: transformers.WhisperModel(config),
    data_gen_fn=data_gen,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)

model_zoo.register(
    name="transformers_whisper_for_conditional_generation",
    model_fn=lambda: transformers.WhisperForConditionalGeneration(config),
    data_gen_fn=data_gen_for_conditional_generation,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn_attr,
    model_attribute=ModelAttribute(has_control_flow=True),
)

model_zoo.register(
    name="transformers_whisper_for_audio_classification",
    model_fn=lambda: transformers.WhisperForAudioClassification(config),
    data_gen_fn=data_gen_for_audio_classification,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn_attr,
    model_attribute=ModelAttribute(has_control_flow=True),
)
