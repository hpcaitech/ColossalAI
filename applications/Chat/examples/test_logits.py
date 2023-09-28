import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Step 1: Load the GPT-2 tokenizer and model
model_name = "/home/lcyab/data/Anthropic_rlhf/actor/pretrain"  # You can use other GPT-2 variants like "gpt2-medium", "gpt2-large", etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Step 2: Encode the input text
input_text = "Your input text here."
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Step 3: Generate logits
model.eval()
logits = model(input_ids).logits
print(logits.size())
input_token_logits = logits.gather(dim=-1, index=input_ids.unsqueeze(-1))
print(input_token_logits)


logits = model(input_ids).logits
print(logits.size())
input_token_logits = logits.gather(dim=-1, index=input_ids.unsqueeze(-1))
print(input_token_logits)


# # Logits shape: (batch_size, sequence_length, vocab_size)
# # In this case, batch_size=1, sequence_length=the length of the input text, and vocab_size=50257 (for GPT-2)

# # If you want to get the logits for a specific token in the sequence, you can index the logits tensor:
# # For example, to get the logits for the first token in the sequence:
# first_token_logits = logits[0, 0, :]

# # You can also convert the logits to probabilities using softmax:
# probabilities = torch.softmax(first_token_logits, dim=-1)
