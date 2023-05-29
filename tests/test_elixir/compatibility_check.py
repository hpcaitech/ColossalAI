import torch
import torch.distributed as dist

import colossalai
from colossalai.elixir import ElixirModule, ElixirOptimizer
from colossalai.elixir.search import minimum_waste_search
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo


def check_elixir_compatibility(early_stop: bool = True):
    """check gemini plugin over model zoo

    Args:
        early_stop (bool, optional): Whether to stop when getting the first error. Defaults to True.
    """
    passed_models = []
    failed_info = {}    # (model_name, error) pair

    for name, (model_fn, data_gen_fn, output_transform_fn, _) in model_zoo.items():
        # These models lead to CUDA error
        if name in ('diffusers_auto_encoder_kl', 'diffusers_vq_model', 'diffusers_unet2d_model', 'timm_resmlp',
                    'timm_gmixer_12_224', 'timm_gmlp_b16_224', 'timm_mixer_b16_224', 'timm_convnext',
                    'torchaudio_wav2vec2_base', 'torchaudio_hubert_base', 'torchvision_convnext_base'):
            continue

        try:
            print(name)
            global_size = dist.get_world_size()
            global_group = dist.GroupMember.WORLD

            model = model_fn()
            optimizer = HybridAdam(model.parameters(), lr=1e-3)
            criterion = lambda x: x.mean()
            data = data_gen_fn()

            data = {
                k: v.to('cuda') if torch.is_tensor(v) or 'Tensor' in v.__class__.__name__ else v
                for k, v in data.items()
            }

            sr = minimum_waste_search(
            # pre-commit: do not rearrange
                m=model,
                group_size=global_size,
                unified_dtype=torch.float16,
                prefetch=False,
                verbose=True)

            model = ElixirModule(model, sr, global_group, prefetch=False, dtype=torch.float16)
            optimizer = ElixirOptimizer(model, optimizer, initial_scale=32)

            output = model(**data)
            output = output_transform_fn(output)
            output_key = list(output.keys())[0]
            loss = criterion(output[output_key])

            optimizer.backward(loss)
            optimizer.step()
            passed_models.append(name)

            del model, optimizer, criterion, data, output, loss
        except Exception as e:
            failed_info[name] = e
            if early_stop:
                raise e

        torch.cuda.empty_cache()

    if dist.get_rank() == 0:
        print(f'Passed models({len(passed_models)}): {passed_models}\n\n')
        print(f'Failed models({len(failed_info)}): {list(failed_info.keys())}\n\n')
    assert len(failed_info) == 0, '\n'.join([f'{k}: {v}' for k, v in failed_info.items()])


def run_dist(rank, world_size, port, early_stop: bool = True):
    # init dist env
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host='localhost')
    check_elixir_compatibility(early_stop=early_stop)


@rerun_if_address_is_in_use()
def exam_compatibility(early_stop: bool = True):
    spawn(run_dist, 2, early_stop=early_stop)


if __name__ == '__main__':
    exam_compatibility(early_stop=False)
