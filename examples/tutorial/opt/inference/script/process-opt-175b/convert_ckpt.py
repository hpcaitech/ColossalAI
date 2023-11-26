import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def parse_shape_info(flat_dir: str):
    data = load_json(os.path.join(flat_dir, "shape.json"))
    flat_info = defaultdict(lambda: defaultdict(list))
    for k, shape in data.items():
        matched = re.match(r"decoder.layers.\d+", k)
        if matched is None:
            flat_key = "flat_param_0"
        else:
            flat_key = f"{matched[0]}.flat_param_0"
        flat_info[flat_key]["names"].append(k)
        flat_info[flat_key]["shapes"].append(shape)
        flat_info[flat_key]["numels"].append(int(np.prod(shape)))
    return flat_info


def convert(flat_dir: str, output_dir: str, part: int):
    flat_path = os.path.join(flat_dir, f"reshard-model_part-{part}-shard0.pt")
    output_path = os.path.join(output_dir, f"reshard-model_part-{part}.pt")
    flat_meta = load_json(os.path.join(flat_dir, "flat-meta.json"))
    flat_sd = torch.load(flat_path)
    print(f"Loaded flat state dict from {flat_path}")
    output_sd = {}
    for flat_key, param_meta in flat_meta.items():
        flat_param = flat_sd["model"][flat_key]
        assert (
            sum(param_meta["numels"]) == flat_param.numel()
        ), f'flat {flat_key} {flat_param.numel()} vs {sum(param_meta["numels"])}'
        for name, shape, param in zip(
            param_meta["names"], param_meta["shapes"], flat_param.split(param_meta["numels"])
        ):
            output_sd[name] = param.view(shape)

    torch.save(output_sd, output_path)
    print(f"Saved unflat state dict to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("flat_dir")
    parser.add_argument("output_dir")
    parser.add_argument("part", type=int)
    args = parser.parse_args()
    convert(args.flat_dir, args.output_dir, args.part)
