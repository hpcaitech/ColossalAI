import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import ElixirPlugin
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo


def run_fn(model_fn, data_gen_fn, output_transform_fn):
    os_config = dict(initial_scale=64, max_norm=1.0)
    plugin = ElixirPlugin(optimizer_config=os_config)
    booster = Booster(plugin=plugin)
    model = model_fn()
    optimizer = HybridAdam(model.parameters(), lr=1e-3)
    criterion = lambda x: x.mean()
    data = data_gen_fn()

    data = {k: v.to('cuda') if torch.is_tensor(v) or 'Tensor' in v.__class__.__name__ else v for k, v in data.items()}

    model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)

    output = model(**data)
    output = output_transform_fn(output)
    output_key = list(output.keys())[0]
    loss = criterion(output[output_key])

    booster.backward(loss, optimizer)
    optimizer.step()


def check_elixir_plugin(early_stop: bool = True):
    """check elixir plugin over model zoo

    Args:
        early_stop (bool, optional): Whether to stop when getting the first error. Defaults to True.
    """
    passed_info = {}
    failed_info = {}

    for name, (model_fn, data_gen_fn, output_transform_fn, _) in model_zoo.items():
        # have not been tested with torchrec
        if name.startswith('torchrec'):
            continue

        # dm_nfnet is not supported because of the skipinit_gain parameter in its NormFreeBlock
        # there is `out.mul_(self.skipinit_gain)`, which should be changed to `out *= self.skipinit_gain`
        if name in ['timm_dm_nfnet']:
            continue

        # Elixir stipulate that parameters with gradients should have gradients after the backward pass
        # here are some unsupported models

        # these models use layer drop
        # some randomly selected layers are not used in computations
        if name in ['torchaudio_wav2vec2_base', 'torchaudio_hubert_base']:
            continue

        # because our criterion function is too simple to generate gradients for all parameters
        # following models are not supported
        # users should provide complete input data to use all parameters
        if name in ('diffusers_auto_encoder_kl', 'diffusers_vq_model', 'diffusers_unet2d_model', 'transformers_albert',
                    'transformers_albert_for_pretraining', 'transformers_bert_for_pretraining',
                    'transformers_gpt_double_heads', 'transformers_t5', 'transformers_t5_for_conditional_generation',
                    'transformers_t5_encoder_model'):
            continue

        # currently, nn.RNN is not supported yet
        if name in ('torchaudio_deepspeech', 'torchaudio_wavernn', 'torchaudio_tacotron'):
            continue

        try:
            run_fn(model_fn, data_gen_fn, output_transform_fn)
            passed_info[name] = 'passed'
        except Exception as e:
            failed_info[name] = str(e)
            print(f"failed model name: {name}")
            if early_stop:
                raise e

        torch.cuda.empty_cache()

    if dist.get_rank() == 0:
        print(f'Passed models({len(passed_info)}): {list(passed_info.keys())}\n\n')
        print(f'Failed models({len(failed_info)}): {list(failed_info.keys())}\n\n')
    assert len(failed_info) == 0, '\n'.join([f'{k}: {v}' for k, v in failed_info.items()])


def run_dist(rank, world_size, port, early_stop: bool = True):
    # init dist env
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host='localhost')
    check_elixir_plugin(early_stop=early_stop)


@pytest.mark.skip(reason="skip this test now")
@rerun_if_address_is_in_use()
def test_elixir_plugin(early_stop: bool = True):
    spawn(run_dist, 1, early_stop=early_stop)


if __name__ == '__main__':
    test_elixir_plugin(early_stop=True)
