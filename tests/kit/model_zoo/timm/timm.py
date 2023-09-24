import timm.models as tm
import torch

from ..registry import ModelAttribute, model_zoo

## ==============
# Register models without control flow
## ==============
data_gen_fn = lambda: dict(x=torch.rand(2, 3, 224, 224))
output_transform_fn = lambda x: dict(output=x)

model_zoo.register(
    name="timm_resnet", model_fn=tm.resnest.resnest50d, data_gen_fn=data_gen_fn, output_transform_fn=output_transform_fn
)
model_zoo.register(
    name="timm_beit",
    model_fn=tm.beit.beit_base_patch16_224,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)
model_zoo.register(
    name="timm_cait", model_fn=tm.cait.cait_s24_224, data_gen_fn=data_gen_fn, output_transform_fn=output_transform_fn
)
model_zoo.register(
    name="timm_convmixer",
    model_fn=tm.convmixer.convmixer_768_32,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)
model_zoo.register(
    name="timm_efficientnetv2",
    model_fn=tm.efficientnet.efficientnetv2_m,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)
model_zoo.register(
    name="timm_resmlp", model_fn=tm.resmlp_12_224, data_gen_fn=data_gen_fn, output_transform_fn=output_transform_fn
)
model_zoo.register(
    name="timm_vision_transformer",
    model_fn=tm.vision_transformer.vit_base_patch16_224,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)
model_zoo.register(
    name="timm_deit",
    model_fn=tm.deit_base_distilled_patch16_224,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)
model_zoo.register(
    name="timm_beitv2",
    model_fn=tm.beitv2_base_patch16_224,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)
model_zoo.register(
    name="timm_coat", model_fn=tm.coat.coat_lite_mini, data_gen_fn=data_gen_fn, output_transform_fn=output_transform_fn
)

model_zoo.register(
    name="timm_deit3",
    model_fn=tm.deit3_base_patch16_224,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)

model_zoo.register(
    name="timm_eca_nfnet", model_fn=tm.eca_nfnet_l0, data_gen_fn=data_gen_fn, output_transform_fn=output_transform_fn
)
model_zoo.register(
    name="timm_efficientformer",
    model_fn=tm.efficientformer_l1,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)
model_zoo.register(
    name="timm_ese_vovnet19b_dw",
    model_fn=tm.ese_vovnet19b_dw,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)
model_zoo.register(
    name="timm_gmixer_12_224",
    model_fn=tm.gmixer_12_224,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)
model_zoo.register(
    name="timm_gmlp_b16_224", model_fn=tm.gmlp_b16_224, data_gen_fn=data_gen_fn, output_transform_fn=output_transform_fn
)
model_zoo.register(
    name="timm_hardcorenas_a",
    model_fn=tm.hardcorenas_a,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)
model_zoo.register(
    name="timm_hrnet_w18_small",
    model_fn=tm.hrnet_w18_small,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)
model_zoo.register(
    name="timm_inception_v3", model_fn=tm.inception_v3, data_gen_fn=data_gen_fn, output_transform_fn=output_transform_fn
)
model_zoo.register(
    name="timm_mixer_b16_224",
    model_fn=tm.mixer_b16_224,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)
model_zoo.register(
    name="timm_nf_ecaresnet101",
    model_fn=tm.nf_ecaresnet101,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)
model_zoo.register(
    name="timm_nf_regnet_b0", model_fn=tm.nf_regnet_b0, data_gen_fn=data_gen_fn, output_transform_fn=output_transform_fn
)
model_zoo.register(
    name="timm_regnetv_040", model_fn=tm.regnetv_040, data_gen_fn=data_gen_fn, output_transform_fn=output_transform_fn
)
model_zoo.register(
    name="timm_skresnet18", model_fn=tm.skresnet18, data_gen_fn=data_gen_fn, output_transform_fn=output_transform_fn
)
model_zoo.register(
    name="timm_tnt_b_patch16_224",
    model_fn=tm.tnt_b_patch16_224,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)
model_zoo.register(
    name="timm_wide_resnet50_2",
    model_fn=tm.wide_resnet50_2,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
)
model_zoo.register(
    name="timm_convit", model_fn=tm.convit_base, data_gen_fn=data_gen_fn, output_transform_fn=output_transform_fn
)
model_zoo.register(
    name="timm_dm_nfnet", model_fn=tm.dm_nfnet_f0, data_gen_fn=data_gen_fn, output_transform_fn=output_transform_fn
)

# ==============
# Register models with control flow
# ==============
model_zoo.register(
    name="timm_convnext",
    model_fn=tm.convnext.convnext_base,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)
model_zoo.register(
    name="timm_vgg",
    model_fn=tm.vgg.vgg11,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)
model_zoo.register(
    name="timm_dpn",
    model_fn=tm.dpn.dpn68,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)
model_zoo.register(
    name="timm_densenet",
    model_fn=tm.densenet.densenet121,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)
model_zoo.register(
    name="timm_rexnet",
    model_fn=tm.rexnet.rexnet_100,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)
model_zoo.register(
    name="timm_swin_transformer",
    model_fn=tm.swin_transformer.swin_base_patch4_window7_224,
    data_gen_fn=data_gen_fn,
    output_transform_fn=output_transform_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)
