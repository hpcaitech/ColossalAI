import torch
import transformers

from ..registry import ModelAttribute, model_zoo

# ===============================
# Register single-sentence BERT
# ===============================


# define data gen function
def data_gen():
    # Generated from following code snippet
    #
    # from transformers import BertTokenizer
    # input = 'Hello, my dog is cute'
    # tokenized_input = tokenizer(input, return_tensors='pt')
    # input_ids = tokenized_input['input_ids']
    # attention_mask = tokenized_input['attention_mask']
    # token_type_ids = tokenized_input['token_type_ids']
    input_ids = torch.tensor([[101, 7592, 1010, 2026, 3899, 2003, 10140, 102]], dtype=torch.int64)
    token_type_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int64)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int64)
    return dict(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)


def data_gen_for_lm():
    # LM data gen
    # the `labels` of LM is the token of the output, cause no padding, use `input_ids` as `labels`
    data = data_gen()
    data['labels'] = data['input_ids'].clone()
    return data


def data_gen_for_pretraining():
    # pretraining data gen
    # `next_sentence_label` is the label for next sentence prediction, 0 or 1
    data = data_gen_for_lm()
    data['next_sentence_label'] = torch.tensor([1], dtype=torch.int64)
    return data


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


def data_gen_for_mcq():
    # multiple choice question data gen
    # Generated from following code snippet
    #
    # tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    # prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    # choice0 = "It is eaten with a fork and a knife."
    # choice1 = "It is eaten while held in the hand."
    # data = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="pt", padding=True)
    # data = {k: v.unsqueeze(0) for k, v in encoding.items()}
    # data['labels'] = torch.tensor([0], dtype=torch.int64)
    input_ids = torch.tensor([[[
        101, 1999, 3304, 1010, 10733, 2366, 1999, 5337, 10906, 1010, 2107, 2004, 2012, 1037, 4825, 1010, 2003, 3591,
        4895, 14540, 6610, 2094, 1012, 102, 2009, 2003, 8828, 2007, 1037, 9292, 1998, 1037, 5442, 1012, 102
    ],
                               [
                                   101, 1999, 3304, 1010, 10733, 2366, 1999, 5337, 10906, 1010, 2107, 2004, 2012, 1037,
                                   4825, 1010, 2003, 3591, 4895, 14540, 6610, 2094, 1012, 102, 2009, 2003, 8828, 2096,
                                   2218, 1999, 1996, 2192, 1012, 102, 0
                               ]]])
    token_type_ids = torch.tensor(
        [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]])
    attention_mask = torch.tensor(
        [[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]])
    labels = torch.tensor([0], dtype=torch.int64)

    return dict(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)


# define output transform function
output_transform_fn = lambda x: x

# define loss funciton
loss_fn_for_bert_model = lambda x: x.pooler_output.mean()
loss_fn = lambda x: x.loss

config = transformers.BertConfig(hidden_size=128,
                                 num_hidden_layers=2,
                                 num_attention_heads=4,
                                 intermediate_size=256,
                                 hidden_dropout_prob=0,
                                 attention_probs_dropout_prob=0)

# register the BERT variants
model_zoo.register(name='transformers_bert',
                   model_fn=lambda: transformers.BertModel(config),
                   data_gen_fn=data_gen,
                   output_transform_fn=output_transform_fn,
                   loss_fn=loss_fn_for_bert_model,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_bert_for_pretraining',
                   model_fn=lambda: transformers.BertForPreTraining(config),
                   data_gen_fn=data_gen_for_pretraining,
                   output_transform_fn=output_transform_fn,
                   loss_fn=loss_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_bert_lm_head_model',
                   model_fn=lambda: transformers.BertLMHeadModel(config),
                   data_gen_fn=data_gen_for_lm,
                   output_transform_fn=output_transform_fn,
                   loss_fn=loss_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_bert_for_masked_lm',
                   model_fn=lambda: transformers.BertForMaskedLM(config),
                   data_gen_fn=data_gen_for_lm,
                   output_transform_fn=output_transform_fn,
                   loss_fn=loss_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_bert_for_sequence_classification',
                   model_fn=lambda: transformers.BertForSequenceClassification(config),
                   data_gen_fn=data_gen_for_sequence_classification,
                   output_transform_fn=output_transform_fn,
                   loss_fn=loss_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_bert_for_token_classification',
                   model_fn=lambda: transformers.BertForTokenClassification(config),
                   data_gen_fn=data_gen_for_token_classification,
                   output_transform_fn=output_transform_fn,
                   loss_fn=loss_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_bert_for_next_sentence',
                   model_fn=lambda: transformers.BertForNextSentencePrediction(config),
                   data_gen_fn=data_gen_for_sequence_classification,
                   output_transform_fn=output_transform_fn,
                   loss_fn=loss_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_bert_for_mcq',
                   model_fn=lambda: transformers.BertForMultipleChoice(config),
                   data_gen_fn=data_gen_for_mcq,
                   output_transform_fn=output_transform_fn,
                   loss_fn=loss_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
