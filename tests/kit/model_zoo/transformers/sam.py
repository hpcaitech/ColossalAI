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
    # from transformers import SamModel, SamProcessor
    #
    # model = SamModel.from_pretrained("facebook/sam-vit-base")
    # processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    #
    # img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    # raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    # input_points = [[[450, 600]]] # 2D localization of a window
    # inputs = processor(raw_image, input_points=input_points, return_tensors="pt")

    pixel_values = torch.rand(1, 3, 1024, 1024, dtype=torch.float32)
    original_sizes = torch.tensor([[1764, 2646]], dtype=torch.int64)
    reshaped_input_sizes = torch.tensor([[683, 1024]], dtype=torch.int64)
    input_points = torch.tensor([[[[174.1497, 232.3129]]]], dtype=torch.float64)
    return dict(
        pixel_values=pixel_values,
        original_sizes=original_sizes,
        reshaped_input_sizes=reshaped_input_sizes,
        input_points=input_points,
    )


# define output transform function
output_transform_fn = lambda x: x

# define loss funciton
loss_fn = lambda x: x["iou_scores"].mean()

config = transformers.SamConfig()
config.vision_config.num_hidden_layers = 2

# register the BERT variants
model_zoo.register(
    name="transformers_sam",
    model_fn=lambda: transformers.SamModel(config),
    data_gen_fn=data_gen,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)
