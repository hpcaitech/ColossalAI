import torch
import transformers

from tests.kit.benchmark import BenchmarkSpec

from ..registry import ModelAttribute, model_zoo

try:
    from transformers import Qwen3Config

    HAS_QWEN3 = True
except ImportError:
    HAS_QWEN3 = False

if HAS_QWEN3:
    # ===============================
    # Register Qwen3
    # ===============================

    def data_gen():
        # the input ids are corresponding to the sentence
        # 'Hello, my dog is cute'
        #
        # the code is give below:
        # -----------------------------------
        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B')
        # input = "This is a test sentence. This is a test sentence. This is a test sentence. This is a test sentence."
        # tokenized_input = tokenizer(input, return_tensors='pt').to('cuda')
        # -----------------------------------

        # NOTE: due to sp convention, need to be a multiple of 4
        input_ids = torch.tensor(
            [
                [
                    1986,
                    374,
                    264,
                    1273,
                    11652,
                    13,
                    1096,
                    374,
                    264,
                    1273,
                    11652,
                    13,
                    1096,
                    374,
                    264,
                    1273,
                    11652,
                    13,
                    1096,
                    374,
                    264,
                    1273,
                    11652,
                    13,
                ]
            ],
            dtype=torch.long,
        )
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
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

    config = Qwen3Config(
        hidden_size=128,
        intermediate_size=256,
        max_window_layers=4,
        num_attention_heads=16,
        num_hidden_layers=4,
        num_key_value_heads=16,
        attn_implementation="sdpa",  # for tests on fp32
        sliding_window=None,  # not supported by sdpa
        use_cache=False,
    )

    config.pad_token_id = 0

    # Real sizes for benchmarking, from the config.json of Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B and Qwen/Qwen3-8B
    QWEN3 = dict(
        vocab_size=151936,
        num_key_value_heads=8,
        head_dim=128,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        max_position_embeddings=40960,
        attn_implementation="sdpa",
        use_cache=False,
    )
    QWEN3_SIZES = {
        "0.6b": dict(
            hidden_size=1024,
            intermediate_size=3072,
            num_hidden_layers=28,
            num_attention_heads=16,
            tie_word_embeddings=True,
        ),
        "1.7b": dict(
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=28,
            num_attention_heads=16,
            tie_word_embeddings=True,
        ),
        "8b": dict(
            hidden_size=4096,
            intermediate_size=12288,
            num_hidden_layers=36,
            num_attention_heads=32,
            tie_word_embeddings=False,
        ),
    }

    def qwen3(size):
        return lambda: transformers.Qwen3ForCausalLM(Qwen3Config(**QWEN3, **QWEN3_SIZES[size]))

    benchmark = BenchmarkSpec(models={size: qwen3(size) for size in QWEN3_SIZES}, seq_len=4096)

    # register the following models
    # transformers.Qwen3Model,
    # transformers.Qwen3ForCausalLM,
    # transformers.Qwen3ForSequenceClassification,
    model_zoo.register(
        name="transformers_qwen3",
        model_fn=lambda: transformers.Qwen3Model(config),
        data_gen_fn=data_gen,
        output_transform_fn=output_transform_fn,
        loss_fn=loss_fn,
        model_attribute=ModelAttribute(has_control_flow=True),
    )
    model_zoo.register(
        name="transformers_qwen3_for_causal_lm",
        model_fn=lambda: transformers.Qwen3ForCausalLM(config),
        data_gen_fn=data_gen_for_causal_lm,
        output_transform_fn=output_transform_fn,
        loss_fn=loss_fn_for_causal_lm,
        model_attribute=ModelAttribute(has_control_flow=True, benchmark=benchmark),
    )
    model_zoo.register(
        name="transformers_qwen3_for_sequence_classification",
        model_fn=lambda: transformers.Qwen3ForSequenceClassification(config),
        data_gen_fn=data_gen,
        output_transform_fn=output_transform_fn,
        loss_fn=loss_fn_for_seq_classification,
        model_attribute=ModelAttribute(has_control_flow=True),
    )
