import argparse
import dataclasses
import os
import parser
from typing import List

import tqdm
from huggingface_hub import hf_hub_download, snapshot_download


@dataclasses.dataclass
class HFRepoFiles:
    repo_id: str
    files: List[str]

    def download(self, dir_path: str):
        for file in self.files:
            file_path = hf_hub_download(self.repo_id, file, local_dir=dir_path)

    def download_all(self):
        snapshot_download(self.repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="test_models")
    parser.add_argument("--config-only", default=False, action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.model_dir):
        print(f"[INFO]: {args.model_dir} already exists")
        exit(0)

    repo_list = {
        "gpt2": HFRepoFiles(repo_id="gpt2", files=["config.json", "tokenizer.json", "vocab.json", "merges.txt"]),
        "bloom": HFRepoFiles(
            repo_id="bigscience/bloom-560m", files=["config.json", "tokenizer.json", "tokenizer_config.json"]
        ),
        "opt": HFRepoFiles(
            repo_id="facebook/opt-350m", files=["config.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
        ),
    }

    os.mkdir(args.model_dir)
    for model_name in tqdm.tqdm(repo_list):
        dir_path = os.path.join(args.model_dir, model_name)
        if args.config_only:
            os.mkdir(dir_path)
            repo_list[model_name].download(dir_path)
        else:
            repo_list[model_name].download_all()
