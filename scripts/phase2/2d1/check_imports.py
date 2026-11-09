"""Check cached model configuration and Python imports, without loading weights."""

import json
import traceback
from pathlib import Path

import huggingface_hub
import transformers
from check_remote_tiny import MODELS as REMOTE_MODELS
from huggingface_hub import snapshot_download
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

REVISIONS = dict(REMOTE_MODELS.values())
REVISIONS["GSAI-ML/LLaDA-8B-Base"] = "0f2787f2d87eac5eed8a087d5ecd24277e6255b2"

MODELS = [
    "GSAI-ML/LLaDA-8B-Base",
    "inclusionAI/LLaDA2.1-mini",
    "inclusionAI/LLaDA2.2-mini",
    "ByteDance/Ouro-1.4B",
    "ByteDance/Ouro-2.6B",
]


def main():
    print("Transformers:", transformers.__version__)
    print("huggingface_hub:", huggingface_hub.__version__)
    failed = []
    for repo_id in MODELS:
        print(f"\n========== {repo_id} ==========")
        stage = "查找本地文件"
        try:
            path = snapshot_download(repo_id, revision=REVISIONS[repo_id], local_files_only=True)
            raw = json.loads((Path(path) / "config.json").read_text())
            print("快照版本:", Path(path).name)
            print("模型代码入口:", raw["auto_map"]["AutoModelForCausalLM"])

            stage = "加载配置"
            config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
            print("CONFIG_OK:", type(config).__name__)
            print("实际 vocab_size:", getattr(config, "vocab_size", None))
            print("实际 pad_token_id:", getattr(config, "pad_token_id", None))

            stage = "导入模型类"
            model_class = get_class_from_dynamic_module(
                raw["auto_map"]["AutoModelForCausalLM"],
                path,
                local_files_only=True,
            )
            print("IMPORT_OK:", model_class.__name__)
        except Exception:
            failed.append(repo_id)
            print("FAILED，失败阶段:", stage)
            traceback.print_exc()

    print(f"\n导入检查结果: {len(MODELS) - len(failed)}/{len(MODELS)} 通过")
    if failed:
        print("失败模型:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
