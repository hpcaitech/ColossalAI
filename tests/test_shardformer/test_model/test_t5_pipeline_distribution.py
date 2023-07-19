from colossalai.shardformer.policies.t5 import distribute_t5_layers


def test_t5_pipeline_distribution():

    config = {'d_model': 128, 'num_layers': 2, 'dropout_rate': 0, 'decoder_start_token_id': 0}

    num_test_cases = 8
    test_dict = {
        'num_encoder_layers': [2, 1, 3, 2, 3, 2, 10, 5],
        'num_decoder_layers': [2, 8, 0, 2, 1, 5, 6, 22],
        'num_stages': [2, 2, 2, 4, 4, 4, 8, 8],
        'decoder_starting_stage': [1, 1, -1, 2, 3, 1, 5, 2]
    }

    for i in range(num_test_cases):
        _, ans = distribute_t5_layers(test_dict['num_encoder_layers'][i], test_dict['num_decoder_layers'][i],
                                      test_dict['num_stages'][i])
        assert test_dict['decoder_starting_stage'][i] == ans
