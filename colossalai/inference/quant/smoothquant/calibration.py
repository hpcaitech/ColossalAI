import functools

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm


def get_act_scales(model, tokenizer, dataset_path, num_samples=512, seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(functools.partial(stat_input_hook, name=name)))
    # print("data path", dataset_path)
    # dataset = load_dataset("csv", data_files=dataset_path, split="train")
    # dataset = dataset.shuffle(seed=42)

    # dataset = load_dataset("/home/lcxk/data3/datasets/cc_news.py", data_files=dataset_path)
    dataset = load_dataset("json", data_files=dataset_path)

    print("text", dataset["train"]["rows"][0][1]["row"]["text"])
    # for test in dataset["train"]["rows"]:
    #     print(test)
    dataset = dataset.shuffle(seed=42)
    # print("text", dataset["rows"][0])

    for i in tqdm(range(num_samples)):
        # print("text", dataset[i])
        input_ids = tokenizer(
            dataset["train"]["rows"][0][i]["row"]["text"],
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
        ).input_ids.to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    return act_scales
