from colossalai.shardformer.policies.t5 import T5BasePolicy


def test_t5_pipeline_distribution():
    num_test_cases = 8
    test_dict = {
        "num_encoder_layers": [2, 1, 3, 2, 3, 2, 10, 5],
        "num_decoder_layers": [2, 8, 0, 2, 1, 5, 6, 22],
        "num_stages": [2, 2, 2, 4, 4, 4, 8, 8],
        "decoder_starting_stage": [1, 1, 2, 2, 3, 1, 5, 2],
    }

    for i in range(num_test_cases):
        _, decoder_starting_stage = T5BasePolicy.distribute_t5_layers(
            test_dict["num_encoder_layers"][i], test_dict["num_decoder_layers"][i], test_dict["num_stages"][i]
        )
        assert test_dict["decoder_starting_stage"][i] == decoder_starting_stage


def test_t5_pipeline_layers():
    num_test_cases = 4
    test_dict = {
        "num_encoder_layers": [2, 3, 2, 4],
        "num_decoder_layers": [2, 0, 2, 8],
        "num_stages": [2, 2, 4, 4],
        "layers_per_stage": [
            [[0, 2], [0, 2]],
            [[0, 1], [1, 3]],
            [[0, 1], [1, 2], [0, 1], [1, 2]],
            [[0, 4], [0, 3], [3, 6], [6, 8]],
        ],
    }

    for i in range(num_test_cases):
        layers_per_stage, decoder_starting_stage = T5BasePolicy.distribute_t5_layers(
            test_dict["num_encoder_layers"][i], test_dict["num_decoder_layers"][i], test_dict["num_stages"][i]
        )

        for stage in range(test_dict["num_stages"][i]):
            start_idx, end_idx = test_dict["layers_per_stage"][i][stage]
            predicted_start, predicted_end = T5BasePolicy.get_t5_stage_index(
                layers_per_stage, stage, decoder_starting_stage
            )
            assert start_idx == predicted_start
            assert end_idx == predicted_end
