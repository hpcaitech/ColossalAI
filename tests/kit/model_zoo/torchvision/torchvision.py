import torch
import torchvision
import torchvision.models as tm
from packaging import version

from ..registry import ModelAttribute, model_zoo

data_gen_fn = lambda: dict(x=torch.rand(4, 3, 224, 224))
output_transform_fn = lambda x: dict(output=x)

# special data gen fn
inception_v3_data_gen_fn = lambda: dict(x=torch.rand(4, 3, 299, 299))


# special model fn
def swin_s():
    from torchvision.models.swin_transformer import Swin_T_Weights, _swin_transformer

    # adapted from torchvision.models.swin_transformer.swin_small
    weights = None
    weights = Swin_T_Weights.verify(weights)
    progress = True

    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0,  # it is originally 0.2, but we set it to 0 to make it deterministic
        weights=weights,
        progress=progress,
    )


# special output transform fn
google_net_output_transform_fn = lambda x: (
    dict(output=sum(x)) if isinstance(x, torchvision.models.GoogLeNetOutputs) else dict(output=x)
)
swin_s_output_output_transform_fn = lambda x: (
    {f"output{idx}": val for idx, val in enumerate(x)} if isinstance(x, tuple) else dict(output=x)
)
inception_v3_output_transform_fn = lambda x: (
    dict(output=sum(x)) if isinstance(x, torchvision.models.InceptionOutputs) else dict(output=x)
)

model_zoo.register(
    name="torchvision_alexnet", model_fn=tm.alexnet, data_gen_fn=data_gen_fn, output_transform_fn=output_transform_fn
)
model_zoo.register(
    name="torchvision_densenet121",
    model_fn=tm.densenet121,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)
model_zoo.register(
    name="torchvision_efficientnet_b0",
    model_fn=tm.efficientnet_b0,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
    model_attribute=ModelAttribute(has_stochastic_depth_prob=True),
)
model_zoo.register(
    name="torchvision_googlenet",
    model_fn=tm.googlenet,
    data_gen_fn=data_gen_fn,
    output_transform_fn=google_net_output_transform_fn,
)
model_zoo.register(
    name="torchvision_inception_v3",
    model_fn=tm.inception_v3,
    data_gen_fn=inception_v3_data_gen_fn,
    output_transform_fn=inception_v3_output_transform_fn,
)
model_zoo.register(
    name="torchvision_mobilenet_v2",
    model_fn=tm.mobilenet_v2,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)
model_zoo.register(
    name="torchvision_mobilenet_v3_small",
    model_fn=tm.mobilenet_v3_small,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)
model_zoo.register(
    name="torchvision_mnasnet0_5",
    model_fn=tm.mnasnet0_5,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)
model_zoo.register(
    name="torchvision_resnet18", model_fn=tm.resnet18, data_gen_fn=data_gen_fn, output_transform_fn=output_transform_fn
)
model_zoo.register(
    name="torchvision_regnet_x_16gf",
    model_fn=tm.regnet_x_16gf,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)
model_zoo.register(
    name="torchvision_resnext50_32x4d",
    model_fn=tm.resnext50_32x4d,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)
model_zoo.register(
    name="torchvision_shufflenet_v2_x0_5",
    model_fn=tm.shufflenet_v2_x0_5,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)
model_zoo.register(
    name="torchvision_squeezenet1_0",
    model_fn=tm.squeezenet1_0,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)

model_zoo.register(
    name="torchvision_vgg11", model_fn=tm.vgg11, data_gen_fn=data_gen_fn, output_transform_fn=output_transform_fn
)
model_zoo.register(
    name="torchvision_wide_resnet50_2",
    model_fn=tm.wide_resnet50_2,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)

if version.parse(torchvision.__version__) >= version.parse("0.12.0"):
    model_zoo.register(
        name="torchvision_vit_b_16",
        model_fn=tm.vit_b_16,
        data_gen_fn=data_gen_fn,
        output_transform_fn=output_transform_fn,
    )
    model_zoo.register(
        name="torchvision_convnext_base",
        model_fn=tm.convnext_base,
        data_gen_fn=data_gen_fn,
        output_transform_fn=output_transform_fn,
        model_attribute=ModelAttribute(has_stochastic_depth_prob=True),
    )

if version.parse(torchvision.__version__) >= version.parse("0.13.0"):
    model_zoo.register(
        name="torchvision_swin_s",
        model_fn=swin_s,
        data_gen_fn=data_gen_fn,
        output_transform_fn=swin_s_output_output_transform_fn,
    )
    model_zoo.register(
        name="torchvision_efficientnet_v2_s",
        model_fn=tm.efficientnet_v2_s,
        data_gen_fn=data_gen_fn,
        output_transform_fn=output_transform_fn,
        model_attribute=ModelAttribute(has_stochastic_depth_prob=True),
    )
