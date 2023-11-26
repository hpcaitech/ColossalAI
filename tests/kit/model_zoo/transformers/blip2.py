import torch
import transformers

from ..registry import ModelAttribute, model_zoo

# ===============================
# Register single-image SAM
# ===============================


# define data gen function
def data_gen():
    # Generated from following code snippet
    #
    # from PIL import Image
    # import requests
    # from transformers import Blip2Processor, Blip2Model
    # import torch

    # processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)

    # prompt = "Question: how many cats are there? Answer:"
    # inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

    pixel_values = torch.rand(1, 3, 224, 224, dtype=torch.float32)
    input_ids = torch.tensor([[2, 45641, 35, 141, 171, 10017, 32, 89, 116, 31652, 35]], dtype=torch.int64)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int64)
    labels = torch.tensor([[34, 56]], dtype=torch.int64)
    return dict(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, labels=labels)


# define output transform function
output_transform_fn = lambda x: x

# define loss funciton
loss_fn_blip2_model = lambda x: x["loss"]

config = transformers.Blip2Config()
config.vision_config.patch_size = 14
config.text_config.num_hidden_layers = 1
config.qformer_config.num_hidden_layers = 1
config.vision_config.num_hidden_layers = 1
config.qformer_config.attention_probs_dropout_prob = 0
config.qformer_config.hidden_dropout_prob = 0
config.text_config.dropout = 0

# register the blip2 variants
model_zoo.register(
    name="transformers_blip2",
    model_fn=lambda: transformers.Blip2Model(config),
    data_gen_fn=data_gen,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn_blip2_model,
    model_attribute=ModelAttribute(has_control_flow=True),
)

model_zoo.register(
    name="transformers_blip2_conditional_gerneration",
    model_fn=lambda: transformers.Blip2ForConditionalGeneration(config),
    data_gen_fn=data_gen,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn_blip2_model,
    model_attribute=ModelAttribute(has_control_flow=True),
)
