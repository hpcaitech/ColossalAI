import os
from multiprocessing import Pool

import torch

# download pytorch model ckpt in https://huggingface.co/facebook/opt-66b/tree/main
# you can use whether wget or git lfs

path = "/path/to/your/ckpt"
new_path = "/path/to/the/processed/ckpt/"

assert os.path.isdir(path)
files = []
for filename in os.listdir(path):
    filepath = os.path.join(path, filename)
    if os.path.isfile(filepath):
        files.append(filepath)

with Pool(14) as pool:
    ckpts = pool.map(torch.load, files)

restored = {}
for ckpt in ckpts:
    for k, v in ckpt.items():
        if k[0] == "m":
            k = k[6:]
        if k == "lm_head.weight":
            k = "head.dense.weight"
        if k == "decoder.final_layer_norm.weight":
            k = "decoder.layer_norm.weight"
        if k == "decoder.final_layer_norm.bias":
            k = "decoder.layer_norm.bias"
        restored[k] = v
restored["decoder.version"] = "0.0"


split_num = len(restored.keys()) // 60
count = 0
file_count = 1
tmp = {}
for k, v in restored.items():
    print(k)
    tmp[k] = v
    count = count + 1
    if count == split_num:
        filename = str(file_count) + "-restored.pt"
        torch.save(tmp, os.path.join(new_path, filename))
        file_count = file_count + 1
        count = 0
        tmp = {}

filename = str(file_count) + "-restored.pt"
torch.save(tmp, os.path.join(new_path, filename))
