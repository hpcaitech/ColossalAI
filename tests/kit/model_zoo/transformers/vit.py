import torch
import transformers

from ..registry import ModelAttribute, model_zoo

# ===============================
# Register single-sentence VIT
# ===============================


# define data gen function
def data_gen():
    
    pixel_values = torch.randn(1, 3, 224, 224)
    return dict(pixel_values = pixel_values)

# define output transform function
output_transform_fn = lambda x: x

# function to get the loss
loss_fn_for_vit_model = lambda x : x.pooler_output.mean()
loss_fn = lambda x : x.loss

config = transformers.ViTConfig(num_hidden_layers=4,
                                hidden_size=128,
                                intermediate_size=256,
                                num_attention_heads=4)

# register the following models
# transformers.ViTModel,
# transformers.ViTForMaskedImageModeling,
# transformers.ViTForImageClassification,
model_zoo.register(name = 'transformers_vit',
                    model_fn = lambda : transformers.ViTModel(config),
                    data_gen_fn = data_gen,
                    output_transform_fn = output_transform_fn,
                    loss_fn = loss_fn_for_vit_model,
                    model_attribute = ModelAttribute(has_control_flow=True))

model_zoo.register(name = 'transformers_vit_for_masked_image_modeling',
                    model_fn = lambda : transformers.ViTForMaskedImageModeling(config),
                    data_gen_fn = data_gen,
                    output_transform_fn = output_transform_fn,
                    loss_fn = loss_fn,
                    model_attribute = ModelAttribute(has_control_flow=True))

model_zoo.register(name = 'transformers_vit_for_image_classification',
                    model_fn = lambda : transfomers.ViTForImageClassification(config),
                    data_gen_fn = data_gen,
                    output_transform_fn = output_transform_fn,
                    loss_fn = loss_fn,
                    model_attribute = ModelAttribute(has_control_flow=True))


