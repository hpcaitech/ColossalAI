import torch
import transformers

from ..registry import ModelAttribute, model_zoo

# ===============================
# Register Bloom
# ===============================


def data_gen():
    # Generated from following code snippet
    #
    # from transformers import BloomTokenizer
    # input = 'Hello, my dog is cute'
    # tokenized_input = tokenizer(input, return_tensors='pt')
    # input_ids = tokenized_input['input_ids']
    # attention_mask = tokenized_input['attention_mask']
    input_ids = torch.tensor([[59414, 15, 2670, 35433, 632, 207595]], dtype=torch.int64)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.int64)
    return dict(input_ids=input_ids, attention_mask=attention_mask)


def data_gen_for_lm():
    # LM data gen
    # the `labels` of LM is the token of the output, cause no padding, use `input_ids` as `labels`
    data = data_gen()
    data['labels'] = data['input_ids'].clone()
    return data


def data_gen_for_token_classification():
    # token classification data gen
    # `labels` is the type not the token id for token classification, 0 or 1
    data = data_gen()
    data['labels'] = torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.int64)
    return data


def data_gen_for_sequence_classification():
    # sequence classification data gen
    data = data_gen()
    data['labels'] = torch.tensor([0], dtype=torch.int64)
    return data


def data_gen_for_question_answering():
    # obtained with the following code
    #
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
    # question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    # inputs = tokenizer(question, text, return_tensors="pt")

    input_ids = torch.tensor(
        [[57647, 1620, 23967, 620, 107373, 34, 91514, 620, 107373, 1620, 267, 35378, 48946, 18161]], dtype=torch.int64)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int64)
    return dict(input_ids=input_ids, attention_mask=attention_mask)


# define output transform function
output_transform_fn = lambda x: x

# define loss function
loss_fn_for_bloom_model = lambda x: x.last_hidden_state.mean()
loss_fn_for_causal_lm = lambda x: x.loss
loss_fn_for_classification = lambda x: x.logits.mean()
loss_fn_for_question_answering = lambda x: x.end_logits.mean()

config = transformers.BloomConfig(n_layer=1,
                                  n_head=4,
                                  vocab_size=250880,
                                  hidden_dropout=0,
                                  attention_dropout=0,
                                  hidden_size=64)

# register the following models
model_zoo.register(name='transformers_bloom',
                   model_fn=lambda: transformers.BloomModel(config),
                   data_gen_fn=data_gen,
                   output_transform_fn=output_transform_fn,
                   loss_fn=loss_fn_for_bloom_model,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_bloom_for_causal_lm',
                   model_fn=lambda: transformers.BloomForCausalLM(config),
                   data_gen_fn=data_gen_for_lm,
                   output_transform_fn=output_transform_fn,
                   loss_fn=loss_fn_for_causal_lm,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_bloom_for_sequence_classification',
                   model_fn=lambda: transformers.BloomForSequenceClassification(config),
                   data_gen_fn=data_gen_for_sequence_classification,
                   output_transform_fn=output_transform_fn,
                   loss_fn=loss_fn_for_classification,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_bloom_for_token_classification',
                   model_fn=lambda: transformers.BloomForTokenClassification(config),
                   data_gen_fn=data_gen_for_token_classification,
                   output_transform_fn=output_transform_fn,
                   loss_fn=loss_fn_for_classification,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_bloom_for_question_answering',
                   model_fn=lambda: transformers.BloomForQuestionAnswering(config),
                   data_gen_fn=data_gen_for_question_answering,
                   output_transform_fn=output_transform_fn,
                   loss_fn=loss_fn_for_question_answering,
                   model_attribute=ModelAttribute(has_control_flow=True))
