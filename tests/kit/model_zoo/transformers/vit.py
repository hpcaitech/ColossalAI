import torch
import transformers

from ..registry import ModelAttribute, model_zoo

# ===============================
# Register single-sentence VIT
# ===============================

config = transformers.ViTConfig(num_hidden_layers=4, hidden_size=128, intermediate_size=256, num_attention_heads=4)


# define data gen function
def data_gen():
    pixel_values = torch.randn(1, 3, 224, 224)
    return dict(pixel_values=pixel_values)


def data_gen_for_image_classification():
    data = data_gen()
    data["labels"] = torch.tensor([0])
    return data


def data_gen_for_masked_image_modeling():
    data = data_gen()
    num_patches = (config.image_size // config.patch_size) ** 2
    bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()
    data["bool_masked_pos"] = bool_masked_pos
    return data


# define output transform function
output_transform_fn = lambda x: x

# function to get the loss
loss_fn_for_vit_model = lambda x: x["pooler_output"].mean()
loss_fn_for_image_classification = lambda x: x["logits"].mean()
loss_fn_for_masked_image_modeling = lambda x: x["loss"]

# register the following models
# transformers.ViTModel,
# transformers.ViTForMaskedImageModeling,
# transformers.ViTForImageClassification,
model_zoo.register(
    name="transformers_vit",
    model_fn=lambda: transformers.ViTModel(config),
    data_gen_fn=data_gen,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn_for_vit_model,
    model_attribute=ModelAttribute(has_control_flow=True),
)

model_zoo.register(
    name="transformers_vit_for_masked_image_modeling",
    model_fn=lambda: transformers.ViTForMaskedImageModeling(config),
    data_gen_fn=data_gen_for_masked_image_modeling,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn_for_masked_image_modeling,
    model_attribute=ModelAttribute(has_control_flow=True),
)

model_zoo.register(
    name="transformers_vit_for_image_classification",
    model_fn=lambda: transformers.ViTForImageClassification(config),
    data_gen_fn=data_gen_for_image_classification,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn_for_image_classification,
    model_attribute=ModelAttribute(has_control_flow=True),
)
