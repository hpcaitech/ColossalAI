from coati.models.gpt import GPTActor
from coati.models.generation import generate
from transformers import GPT2Tokenizer
from transformers import AutoConfig
import torch

config = AutoConfig.from_pretrained('/home/lcyab/data/Anthropic_rlhf/actor/pretrain')
config.dropout = 0.0
actor = GPTActor(config=config, lora_rank=0)
actor.model.load_state_dict(torch.load('/home/lcyab/data/Anthropic_rlhf/actor/pretrain'+ '/pytorch_model.bin', map_location="cpu"), strict=True)
input_ids = torch.tensor([[50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 16594,   257, 21247,   546,   262,  6817,   286, 14442, 27186,   290, 26005, 48960,    13]]).cuda()
print(actor)
actor.eval()
actor = actor.cuda()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
out = generate(actor, input_ids, tokenizer, max_length=50)
print(out)
print(tokenizer.batch_decode(out))
