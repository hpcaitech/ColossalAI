from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin
from colossalai.nn.optimizer import HybridAdam
from colossalai.tensor.colo_parameter import ColoParameter
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port
from tests.kit.model_zoo import model_zoo


def check_gemini_plugin(early_stop: bool = True):
    """check gemini plugin over model zoo

    Args:
        early_stop (bool, optional): Whether to stop when getting the first error. Defaults to True.
    """
    passed_models = []
    failed_info = {}    # (model_name, error) pair

    for name, (model_fn, data_gen_fn, output_transform_fn, _) in model_zoo.items():
        # These models lead to CUDA error
        if name in ('diffusers_auto_encoder_kl', 'diffusers_vq_model', 'diffusers_unet2d_model', 'timm_resmlp',
                    'timm_gmixer_12_224', 'timm_gmlp_b16_224', 'timm_mixer_b16_224', 'timm_convnext'):
            continue
        # These models are not compatible with gemini
        if name in [
                'diffusers_clip_vision_model', 'timm_resnet', 'timm_beit', 'timm_beitv2', 'timm_eca_nfnet',
                'timm_efficientformer', 'timm_hrnet_w18_small', 'timm_nf_ecaresnet101', 'timm_nf_regnet_b0',
                'timm_skresnet18', 'timm_wide_resnet50_2', 'timm_convit', 'timm_dm_nfnet', 'timm_swin_transformer',
                'torchaudio_conformer', 'torchaudio_deepspeech', 'torchaudio_wavernn', 'torchaudio_tacotron',
                'deepfm_interactionarch', 'deepfm_simpledeepfmnn', 'dlrm', 'dlrm_interactionarch',
                'torchvision_googlenet', 'torchvision_inception_v3', 'torchvision_mobilenet_v3_small',
                'torchvision_resnet18', 'torchvision_resnext50_32x4d', 'torchvision_wide_resnet50_2',
                'torchvision_vit_b_16', 'torchvision_convnext_base', 'torchvision_swin_s', 'transformers_albert',
                'transformers_albert_for_pretraining', 'transformers_bert', 'transformers_bert_for_pretraining',
                'transformers_gpt_double_heads', 'torchaudio_hubert_base', 'torchaudio_wav2vec2_base',
                'transformers_t5_for_conditional_generation', 'transformers_t5', 'transformers_t5_encoder_model'
        ]:
            continue

        try:
            plugin = GeminiPlugin(placement_policy='cuda', strict_ddp_mode=True, max_norm=1.0, initial_scale=2**5)
            booster = Booster(plugin=plugin)
            model = model_fn()
            optimizer = HybridAdam(model.parameters(), lr=1e-3)
            criterion = lambda x: x.mean()
            data = data_gen_fn()

            data = {
                k: v.to('cuda') if torch.is_tensor(v) or 'Tensor' in v.__class__.__name__ else v
                for k, v in data.items()
            }

            model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)

            for n, p in model.named_parameters():
                assert isinstance(p, ColoParameter), f'{n} is not a ColoParameter'

            output = model(**data)
            output = output_transform_fn(output)
            output_key = list(output.keys())[0]
            loss = criterion(output[output_key])

            booster.backward(loss, optimizer)
            optimizer.step()
            passed_models.append(name)

            del booster, plugin, model, optimizer, criterion, data, output, loss
        except Exception as e:
            failed_info[name] = e
            if early_stop:
                raise e

        torch.cuda.empty_cache()

    if dist.get_rank() == 0:
        print(f'Passed models({len(passed_models)}): {passed_models}\n\n')
        print(f'Failed models({len(failed_info)}): {list(failed_info.keys())}\n\n')
    assert len(failed_info) == 0, '\n'.join([f'{k}: {v}' for k, v in failed_info.items()])


def check_dataloader_sharding():
    plugin = GeminiPlugin()

    # create a custom dasetset with 0 to 10
    dataset = torch.utils.data.TensorDataset(torch.arange(0, 10))
    train_dataloader = plugin.prepare_train_dataloader(dataset, batch_size=2)

    # get the first batch of data
    batch = next(iter(train_dataloader))[0].cuda()
    is_rank_0 = dist.get_rank() == 0

    if is_rank_0:
        batch_to_compare = batch.clone()
    else:
        batch_to_compare = batch
    # pass to the rank 1 value to rank 0
    dist.broadcast(batch_to_compare, src=1)

    # compare on rank 0
    if is_rank_0:
        assert not torch.equal(batch,
                               batch_to_compare), 'Same number was found across ranks but expected it to be different'


def run_dist(rank, world_size, port, early_stop: bool = True):
    # init dist env
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host='localhost')
    check_dataloader_sharding()
    check_gemini_plugin(early_stop=early_stop)


@rerun_if_address_is_in_use()
def test_gemini_plugin(early_stop: bool = True):
    world_size = 2
    run_func = partial(run_dist, world_size=world_size, port=free_port(), early_stop=early_stop)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_gemini_plugin(early_stop=False)
