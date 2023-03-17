import torch
import transformers

from ..registry import ModelAttribute, model_zoo

# ===============================
# Register single-sentence BERT
# ===============================
BATCH_SIZE = 2
SEQ_LENGTH = 16


def data_gen_fn():
    input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
    token_type_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
    attention_mask = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
    return dict(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)


output_transform_fn = lambda x: x

config = transformers.BertConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=4, intermediate_size=256)

# register the BERT variants
model_zoo.register(name='transformers_bert',
                   model_fn=lambda: transformers.BertModel(config),
                   data_gen_fn=data_gen_fn,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_bert_for_pretraining',
                   model_fn=lambda: transformers.BertForPreTraining(config),
                   data_gen_fn=data_gen_fn,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_bert_lm_head_model',
                   model_fn=lambda: transformers.BertLMHeadModel(config),
                   data_gen_fn=data_gen_fn,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_bert_for_masked_lm',
                   model_fn=lambda: transformers.BertForMaskedLM(config),
                   data_gen_fn=data_gen_fn,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_bert_for_sequence_classification',
                   model_fn=lambda: transformers.BertForSequenceClassification(config),
                   data_gen_fn=data_gen_fn,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_bert_for_token_classification',
                   model_fn=lambda: transformers.BertForTokenClassification(config),
                   data_gen_fn=data_gen_fn,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))


# ===============================
# Register multi-sentence BERT
# ===============================
def data_gen_for_next_sentence():
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    next_sentence = "The sky is blue due to the shorter wavelength of blue light."
    encoding = tokenizer(prompt, next_sentence, return_tensors="pt")
    return encoding


def data_gen_for_mcq():
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    choice0 = "It is eaten with a fork and a knife."
    choice1 = "It is eaten while held in the hand."
    encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="pt", padding=True)
    encoding = {k: v.unsqueeze(0) for k, v in encoding.items()}
    return encoding


# register the following models
model_zoo.register(name='transformers_bert_for_next_sentence',
                   model_fn=lambda: transformers.BertForNextSentencePrediction(config),
                   data_gen_fn=data_gen_for_next_sentence,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_bert_for_mcq',
                   model_fn=lambda: transformers.BertForMultipleChoice(config),
                   data_gen_fn=data_gen_for_mcq,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
