import torch
import transformers

from ..registry import ModelAttribute, model_zoo

try:
    from transformers import LlamaConfig

    HAS_LLAMA = True
except ImportError:
    HAS_LLAMA = False

if HAS_LLAMA:
    # ===============================
    # Register LLaMA
    # ===============================

    def data_gen():
        # the input ids are corresponding to the sentence
        # 'Hello, my dog is cute'
        #
        # the code is give below:
        # -----------------------------------
        # from transformers import LlamaTokenizerFast
        # tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
        # input = 'Hello, my dog is cute'
        # tokenized_input = tokenizer(input, return_tensors='pt').to('cuda')
        # -----------------------------------

        input_ids = torch.Tensor(
            [
                [1, 15043, 29892, 590, 11203, 338, 274, 1082, 1, 15043, 29892, 590, 11203, 338, 274, 1082],
                [1, 15043, 29892, 590, 11203, 338, 274, 1082, 1, 15043, 29892, 590, 11203, 338, 274, 1082],
            ]
        ).long()
        attention_mask = torch.ones_like(input_ids)
        return dict(input_ids=input_ids, attention_mask=attention_mask)

    # label is needed for causal lm
    def data_gen_for_causal_lm():
        data = data_gen()

        # Test padded sequence
        padding = torch.zeros(2, data["input_ids"].shape[1] // 2, dtype=torch.long)
        data["input_ids"] = torch.cat([data["input_ids"], padding], dim=1)
        data["attention_mask"] = torch.cat([data["attention_mask"], padding], dim=1)

        ignore_idx = -100
        labels = data["input_ids"].clone()
        labels[~data["attention_mask"].bool()] = ignore_idx
        data["labels"] = labels
        return data

    # transform the output to a dict
    output_transform_fn = lambda x: x

    # function to get the loss
    loss_fn = lambda output: output["last_hidden_state"].mean()
    loss_fn_for_causal_lm = lambda output: output["loss"]
    loss_fn_for_seq_classification = lambda output: output["logits"].mean()

    config = LlamaConfig(
        num_hidden_layers=8,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        max_position_embeddings=128,
    )

    if hasattr(config, "pad_token_id"):
        config.pad_token_id = config.eos_token_id

    # register the following models
    # transformers.LlamaForCausalLM,
    # transformers.LlamaModel,
    # transformers.LlamaForSequenceClassification,
    model_zoo.register(
        name="transformers_llama_for_causal_lm",
        model_fn=lambda: transformers.LlamaForCausalLM(config),
        data_gen_fn=data_gen_for_causal_lm,
        output_transform_fn=output_transform_fn,
        loss_fn=loss_fn_for_causal_lm,
        model_attribute=ModelAttribute(has_control_flow=True),
    )
    model_zoo.register(
        name="transformers_llama",
        model_fn=lambda: transformers.LlamaModel(config),
        data_gen_fn=data_gen,
        output_transform_fn=output_transform_fn,
        loss_fn=loss_fn,
        model_attribute=ModelAttribute(has_control_flow=True),
    )
    model_zoo.register(
        name="transformers_llama_for_sequence_classification",
        model_fn=lambda: transformers.LlamaForSequenceClassification(config),
        data_gen_fn=data_gen,
        output_transform_fn=output_transform_fn,
        loss_fn=loss_fn_for_seq_classification,
        model_attribute=ModelAttribute(has_control_flow=True),
    )
