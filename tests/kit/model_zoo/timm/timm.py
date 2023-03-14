import timm.models as tm
import torch

from ..registry import ModelAttribute, model_zoo

## ==============
# Register models without control flow
## ==============

data_gen_fn = lambda: dict(x=torch.rand(2, 3, 224, 224))
output_transform_fn = lambda x: dict(output=x)

model_zoo.register(name='timm_resnet',
                   model_fn=tm.resnest.resnest50d,
                   data_gen_fn=data_gen_fn,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=False))
model_zoo.register(name='timm_beit',
                   model_fn=tm.beit.beit_base_patch16_224,
                   data_gen_fn=data_gen_fn,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=False))
model_zoo.register(name='timm_cait',
                   model_fn=tm.cait.cait_s24_224,
                   data_gen_fn=data_gen_fn,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=False))
model_zoo.register(name='timm_convmixer',
                   model_fn=tm.convmixer.convmixer_768_32,
                   data_gen_fn=data_gen_fn,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=False))
model_zoo.register(name='timm_efficientnetv2',
                   model_fn=tm.efficientnet.efficientnetv2_m,
                   data_gen_fn=data_gen_fn,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=False))
model_zoo.register(name='timm_resmlp',
                   model_fn=tm.resmlp_12_224,
                   data_gen_fn=data_gen_fn,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=False))
model_zoo.register(name='timm_vision_transformer',
                   model_fn=tm.vision_transformer.vit_base_patch16_224,
                   data_gen_fn=data_gen_fn,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=False))
model_zoo.register(name='timm_deit',
                   model_fn=tm.deit_base_distilled_patch16_224,
                   data_gen_fn=data_gen_fn,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=False))

# ==============
# Register models with control flow
# ==============
model_zoo.register(name='timm_convnext',
                   model_fn=tm.convnext.convnext_base,
                   data_gen_fn=data_gen_fn,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='timm_vgg',
                   model_fn=tm.vgg.vgg11,
                   data_gen_fn=data_gen_fn,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='timm_dpn',
                   model_fn=tm.dpn.dpn68,
                   data_gen_fn=data_gen_fn,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='timm_densenet',
                   model_fn=tm.densenet.densenet121,
                   data_gen_fn=data_gen_fn,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='timm_rexnet',
                   model_fn=tm.rexnet.rexnet_100,
                   data_gen_fn=data_gen_fn,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='timm_swin_transformer',
                   model_fn=tm.swin_transformer.swin_base_patch4_window7_224,
                   data_gen_fn=data_gen_fn,
                   output_transform_fn=output_transform_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
