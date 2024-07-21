import torch
import transformers

from ..registry import ModelAttribute, model_zoo

try:
    from transformers import CohereConfig

    HAS_COMMAND = True
except ImportError:
    HAS_COMMAND = False

if HAS_COMMAND:
    # ===============================
    # Register Command-R
    # ===============================

    def data_gen():
        input_ids = torch.Tensor(
            [
                [1, 15043, 29892, 590, 11203, 338, 274, 1082, 1, 15043, 29892, 590, 11203, 338, 274, 1082],
                [1, 15043, 29892, 590, 11203, 338, 274, 1082, 1, 15043, 29892, 590, 11203, 338, 274, 1082],
            ]
        ).long()

        attention_mask = torch.Tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        ).long()

        return dict(input_ids=input_ids, attention_mask=attention_mask)

    # label is needed for causal lm
    def data_gen_for_causal_lm():
        data = data_gen()
        labels = data["input_ids"].clone()
        data["labels"] = labels
        return data

    # transform the output to a dict
    output_transform_fn = lambda x: x

    # function to get the loss
    loss_fn = lambda output: output["last_hidden_state"].mean()
    loss_fn_for_causal_lm = lambda output: output["loss"]
    loss_fn_for_seq_classification = lambda output: output["logits"].mean()

    config = CohereConfig(
        num_hidden_layers=8,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        max_position_embeddings=128,
    )

    if hasattr(config, "pad_token_id"):
        config.pad_token_id = config.eos_token_id

    # register the following models
    # transformers.CohereModel,
    # transformers.CohereForCausalLM,
    model_zoo.register(
        name="transformers_command",
        model_fn=lambda: transformers.CohereModel(config),
        data_gen_fn=data_gen,
        output_transform_fn=output_transform_fn,
        loss_fn=loss_fn,
        model_attribute=ModelAttribute(has_control_flow=True),
    )
    model_zoo.register(
        name="transformers_command_for_causal_lm",
        model_fn=lambda: transformers.CohereForCausalLM(config),
        data_gen_fn=data_gen_for_causal_lm,
        output_transform_fn=output_transform_fn,
        loss_fn=loss_fn_for_causal_lm,
        model_attribute=ModelAttribute(has_control_flow=True),
    )
