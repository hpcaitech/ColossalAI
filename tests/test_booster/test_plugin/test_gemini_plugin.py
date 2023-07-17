from contextlib import nullcontext
from typing import Optional

import torch
import torch.distributed as dist

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin
from colossalai.fx import is_compatible_with_meta
from colossalai.lazy.lazy_init import LazyInitContext
from colossalai.nn.optimizer import HybridAdam
from colossalai.tensor.colo_parameter import ColoParameter
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.zero import ColoInitContext
from tests.kit.model_zoo import model_zoo


def run_fn(init_method, model_fn, data_gen_fn, output_transform_fn) -> Optional[str]:
    try:
        if init_method == 'colo':
            ctx = ColoInitContext()
        elif init_method == 'lazy':
            ctx = LazyInitContext()
        else:
            ctx = nullcontext()
        plugin = GeminiPlugin(placement_policy='cuda', strict_ddp_mode=True, max_norm=1.0, initial_scale=2**5)
        booster = Booster(plugin=plugin)
        with ctx:
            model = model_fn()
        optimizer = HybridAdam(model.parameters(), lr=1e-3)
        criterion = lambda x: x.mean()
        data = data_gen_fn()

        data = {
            k: v.to('cuda') if torch.is_tensor(v) or 'Tensor' in v.__class__.__name__ else v for k, v in data.items()
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

    except Exception as e:
        return repr(e)


# TODO(ver217): CI does not support lazy now
# @parameterize('init_method', ['lazy', 'none', 'colo'])


@parameterize('init_method', ['none'])
def check_gemini_plugin(init_method: str = 'none', early_stop: bool = True):
    """check gemini plugin over model zoo

    Args:
        early_stop (bool, optional): Whether to stop when getting the first error. Defaults to True.
    """
    is_support_meta = is_compatible_with_meta()
    if not is_support_meta and init_method == 'lazy':
        return

    passed_models = []
    failed_info = {}    # (model_name, error) pair

    for name, (model_fn, data_gen_fn, output_transform_fn, _, _) in model_zoo.items():
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

        if init_method == 'lazy' and name in [
                'timm_convmixer', 'timm_vision_transformer', 'timm_deit', 'timm_deit3', 'timm_inception_v3',
                'timm_tnt_b_patch16_224', 'timm_rexnet', 'torchvision_densenet121', 'torchvision_efficientnet_b0',
                'torchvision_mobilenet_v2', 'torchvision_mnasnet0_5', 'torchvision_regnet_x_16gf',
                'torchvision_shufflenet_v2_x0_5', 'torchvision_efficientnet_v2_s'
        ]:
            continue

        err = run_fn(init_method, model_fn, data_gen_fn, output_transform_fn)
        torch.cuda.empty_cache()

        if err is None:
            passed_models.append(name)
        else:
            failed_info[name] = err
            if early_stop:
                break

    if dist.get_rank() == 0:
        print(f'Init method: {init_method}')
        print(f'Passed models({len(passed_models)}): {passed_models}\n\n')
        print(f'Failed models({len(failed_info)}): {list(failed_info.keys())}\n\n')
    assert len(failed_info) == 0, '\n'.join([f'{k}: {v}' for k, v in failed_info.items()])


def run_dist(rank, world_size, port, early_stop: bool = True):
    # init dist env
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host='localhost')
    check_gemini_plugin(early_stop=early_stop)


@rerun_if_address_is_in_use()
def test_gemini_plugin(early_stop: bool = True):
    spawn(run_dist, 4, early_stop=early_stop)


if __name__ == '__main__':
    test_gemini_plugin(early_stop=False)
