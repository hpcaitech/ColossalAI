import sys,os
import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch.fx import GraphModule
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from torch.fx import symbolic_trace
from colossalai.fx.tracer import ColoTracer

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model = model.to('meta')
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def trace_huggingface_clip(bs = 1):
  input_ids = torch.randint(10000, (bs,7), dtype=torch.int64).to('meta')
  attention_mask = torch.randint(2, (bs,7), dtype=torch.int64).to('meta')
  pixel_values = torch.rand(bs, 3, 224, 224, dtype=torch.float32).to('meta')
  meta_args = {"input_ids": input_ids.to('meta'), \
              "pixel_values": pixel_values.to('meta'), \
              "attention_mask": attention_mask.to('meta')}
  tracer = ColoTracer()
  graph = tracer.trace(root=model, meta_args=meta_args, concrete_args = {})
  print(graph)
  
if __name__ == "__main__":
  bs = 2
  print(f'------- batch size = {bs}, will trace clip model success. -------')
  trace_huggingface_clip(bs = bs)
  print(f'------- batch size = 1 -------')
  trace_huggingface_clip()
  
