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
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import COMMON_MODELS, IS_FAST_TEST, model_zoo


@clear_cache_before_run()
def run_fn(init_method, model_fn, data_gen_fn, output_transform_fn, zero_size, tp_size) -> Optional[str]:
    try:
        if init_method == "lazy":
            ctx = LazyInitContext()
        else:
            ctx = nullcontext()
        extra_dp_size = dist.get_world_size() // (zero_size * tp_size)
        enable_all_optimization = True if tp_size > 1 else False
        plugin = GeminiPlugin(
            max_norm=1.0,
            initial_scale=2**5,
            tp_size=tp_size,
            extra_dp_size=extra_dp_size,
            enable_all_optimization=enable_all_optimization,
        )
        booster = Booster(plugin=plugin)
        with ctx:
            model = model_fn()
        optimizer = HybridAdam(model.parameters(), lr=1e-3)
        criterion = lambda x: x.mean()
        data = data_gen_fn()

        data = {
            k: v.to("cuda") if torch.is_tensor(v) or "Tensor" in v.__class__.__name__ else v for k, v in data.items()
        }

        model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)

        for n, p in model.named_parameters():
            assert isinstance(p, ColoParameter), f"{n} is not a ColoParameter"

        output = model(**data)
        output = output_transform_fn(output)
        output_key = list(output.keys())[0]
        loss = criterion(output[output_key])

        booster.backward(loss, optimizer)
        optimizer.step()

    except NotImplementedError:
        print(f"Tensor Parallelism policy for {model.__class__} is not implemented yet\n.")
    except Exception as e:
        # raise e
        return repr(e)


# TODO(ver217): CI does not support lazy now
# @parameterize('init_method', ['lazy', 'none', 'colo'])


@parameterize("subset", [COMMON_MODELS] if IS_FAST_TEST else ["torchvision", "transformers", "diffusers"])
@parameterize("init_method", ["none"])
@parameterize("zero_size", [2])
@parameterize("tp_size", [2])
def check_gemini_plugin(
    subset: str, init_method: str = "none", early_stop: bool = True, zero_size: int = 1, tp_size: int = 1
):
    """check gemini plugin over model zoo

    Args:
        early_stop (bool, optional): Whether to stop when getting the first error. Defaults to True.
    """
    is_support_meta = is_compatible_with_meta()
    if not is_support_meta and init_method == "lazy":
        return

    passed_models = []
    failed_info = {}  # (model_name, error) pair

    for name, (model_fn, data_gen_fn, output_transform_fn, _, _) in model_zoo.get_sub_registry(subset).items():
        # These models lead to CUDA error
        if name in (
            "diffusers_auto_encoder_kl",
            "diffusers_vq_model",
            "diffusers_unet2d_model",
            "timm_resmlp",
            "timm_gmixer_12_224",
            "timm_gmlp_b16_224",
            "timm_mixer_b16_224",
            "timm_convnext",
            "torchvision_convnext_base",
        ):
            continue
        # These models are not compatible with gemini
        if name in [
            "timm_convit",
            "timm_dm_nfnet",
            "torchvision_vit_b_16",
            "transformers_t5",
            "transformers_t5_for_conditional_generation",
            "transformers_t5_encoder_model",  # does not support apex rmsnorm
            "transformers_chatglm",
            "transformers_sam",
            "transformers_vit",
            "transformers_gpt_double_heads",  # TODO check why does the model fail to run using Gemini
            "transformers_falcon",  # TODO check why falcon fails to run Gemini
            "transformers_falcon_for_causal_lm",
            "transformers_falcon_for_sequence_classification",
            "transformers_falcon_for_token_classification",
            "transformers_falcon_for_question_answering",
            "transformers_gptj_lm",  # lead to OOM when running in ci
            "transformers_gptj_for_question_answering",
            "transformers_gptj_for_sequence_classification",
        ]:
            continue

        if init_method == "lazy" and name in [
            "timm_convmixer",
            "timm_vision_transformer",
            "timm_deit",
            "timm_deit3",
            "timm_inception_v3",
            "timm_tnt_b_patch16_224",
            "timm_rexnet",
            "torchvision_densenet121",
            "torchvision_efficientnet_b0",
            "torchvision_mobilenet_v2",
            "torchvision_mnasnet0_5",
            "torchvision_regnet_x_16gf",
            "torchvision_shufflenet_v2_x0_5",
            "torchvision_efficientnet_v2_s",
        ]:
            continue

        # TODO debug blip2 when using tp, something wrong with shift_logits's shape
        if "transformers_blip2" in name:
            tp_size = 1

        err = run_fn(init_method, model_fn, data_gen_fn, output_transform_fn, zero_size, tp_size)
        if err is None:
            passed_models.append(name)
        else:
            failed_info[name] = err
            if early_stop:
                break

    if dist.get_rank() == 0:
        print(f"Init method: {init_method}")
        print(f"Passed models({len(passed_models)}): {passed_models}\n\n")
        print(f"Failed models({len(failed_info)}): {list(failed_info.keys())}\n\n")
    assert len(failed_info) == 0, "\n".join([f"{k}: {v}" for k, v in failed_info.items()])


def run_dist(rank, world_size, port, early_stop: bool = True):
    # init dist env
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")
    check_gemini_plugin(early_stop=early_stop)


@rerun_if_address_is_in_use()
def test_gemini_plugin(early_stop: bool = True):
    spawn(run_dist, 4, early_stop=early_stop)


if __name__ == "__main__":
    test_gemini_plugin(early_stop=False)
