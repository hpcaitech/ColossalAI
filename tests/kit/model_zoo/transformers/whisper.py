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

    input_features = torch.randn(1, 80, 3000)
    decoder_input_ids = torch.tensor([[1, 1]]) * 50258
    return dict(input_features=input_features, decoder_input_ids=decoder_input_ids)


def data_gen_for_sequence_classification():
    # sequence classification data gen
    # `labels` is the label for sequence classification, 0 or 1
    data = data_gen()
    data['labels'] = torch.tensor([1], dtype=torch.int64)
    return data


def data_gen_for_token_classification():
    # token classification data gen
    # `labels` is the type not the token id for token classification, 0 or 1
    data = data_gen()
    data['labels'] = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int64)
    return data


# define output transform function
output_transform_fn = lambda x: x

# define loss funciton
loss_fn = lambda x: x.last_hidden_state.mean()

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
model_zoo.register(name='transformers_whisper',
                   model_fn=lambda: transformers.WhisperModel(config),
                   data_gen_fn=data_gen,
                   output_transform_fn=output_transform_fn,
                   loss_fn=loss_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
