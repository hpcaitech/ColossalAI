import torch
import transformers

from ..registry import ModelAttribute, model_zoo

try:
    from transformers import Qwen2Config

    HAS_QWEN2 = True
except ImportError:
    HAS_QWEN2 = False

if HAS_QWEN2:
    # ===============================
    # Register Qwen2
    # ===============================

    def data_gen():
        # the input ids are corresponding to the sentence
        # 'Hello, my dog is cute'
        #
        # the code is give below:
        # -----------------------------------
        # from transformers import Qwen2TokenizerFast
        # tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen1.5-7B-Chat")
        # input = 'Hello, my dog is cute'
        # tokenized_input = tokenizer(input, return_tensors='pt').to('cuda')
        # -----------------------------------

        input_ids = torch.Tensor(
            [[9707, 11, 847, 5562, 374, 13, 123, 18838], [9707, 11, 847, 5562, 374, 17, 89, 18838]]
        ).long()
        attention_mask = torch.Tensor([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]]).long()
        return dict(input_ids=input_ids, attention_mask=attention_mask)

    # label is needed for casual lm
    def data_gen_for_casual_lm():
        data = data_gen()
        labels = data["input_ids"].clone()
        data["labels"] = labels
        return data

    # transform the output to a dict
    output_transform_fn = lambda x: x

    # function to get the loss
    loss_fn = lambda output: output["last_hidden_state"].mean()
    loss_fn_for_casual_lm = lambda output: output["loss"]
    loss_fn_for_seq_classification = lambda output: output["logits"].mean()

    config = Qwen2Config(
        hidden_size=128,
        intermediate_size=256,
        max_window_layers=4,
        num_attention_heads=16,
        num_hidden_layers=4,
        num_key_value_heads=16,
    )

    config.pad_token_id = 0

    # register the following models
    # transformers.Qwen2Model,
    # transformers.Qwen2ForCausalLM,
    # transformers.Qwen2ForSequenceClassification,
    model_zoo.register(
        name="transformers_qwen2",
        model_fn=lambda: transformers.Qwen2Model(config),
        data_gen_fn=data_gen,
        output_transform_fn=output_transform_fn,
        loss_fn=loss_fn,
        model_attribute=ModelAttribute(has_control_flow=True),
    )
    model_zoo.register(
        name="transformers_qwen2_for_casual_lm",
        model_fn=lambda: transformers.Qwen2ForCausalLM(config),
        data_gen_fn=data_gen_for_casual_lm,
        output_transform_fn=output_transform_fn,
        loss_fn=loss_fn_for_casual_lm,
        model_attribute=ModelAttribute(has_control_flow=True),
    )
    model_zoo.register(
        name="transformers_qwen2_for_sequence_classification",
        model_fn=lambda: transformers.Qwen2ForSequenceClassification(config),
        data_gen_fn=data_gen,
        output_transform_fn=output_transform_fn,
        loss_fn=loss_fn_for_seq_classification,
        model_attribute=ModelAttribute(has_control_flow=True),
    )
