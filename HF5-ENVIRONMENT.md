# HF5 环境留档与服务器安装

本次固定的是 2026-09-22 Mac 上 `colossalai-hf5` 的实际已安装版本：Python 3.11.16、PyTorch 2.11.0、TorchVision 0.26.0、TorchAudio 2.11.0、Transformers 5.17.0、pytest 8.4.2。原有 requirements 文件未修改。

这是当前开发环境的版本快照，**不是已经验证通过的完整 Linux GPU 环境**。Mac 快照没有 Linux PyTorch 所需的 NVIDIA/Triton 等平台依赖，也没有锁定系统驱动、编译器、CUDA Toolkit 和包下载哈希；这些需要在服务器安装并验证后再补充留档。

## 文件分别有什么用

| 文件 | 用途 |
| --- | --- |
| `requirements-hf5.lock.txt` | 当前已安装 Python 包的精确版本，已移除本机路径及 ColossalAI Git 安装地址。服务器安装 CUDA PyTorch 后再安装此文件。 |
| `requirements-hf5-cu128.txt` | 仅安装 PyTorch 三件套，单独使用官方 CUDA 12.8 wheel 索引。 |
| `environment-hf5-server.yml` | 创建 Python 3.11.16 Conda 环境的起点；它本身不包含下面的 pip 安装步骤。 |
| `logs/2026-09-22-hf5-environment-lock/` | 原始 pip/Conda 导出、实际依赖元数据、检查结果和校验和。原始 Conda 导出用于 Mac 留档，不用于 Linux 安装。 |

## 1. 上传当前源码

把当前修改后的仓库目录上传到服务器，包含新增的根目录环境文件、`tests/test_shardformer/test_hf5_compat/` 和 `logs/2026-09-22-qwen-gpu-validation/`。不要上传 `anaconda3/`、`__pycache__/`、`.pytest_cache/`；Mac 环境目录不能直接搬到 Linux 使用。

只克隆远端仓库、只上传 Git 已跟踪文件，或者直接使用原始 `pip freeze` 中的 `-e git+...`，都会遗漏尚未提交的修改或新增文件。开发安装必须指向这份实际上传的源码。

## 2. 在 Linux NVIDIA 服务器安装

以下命令在**服务器上传后的仓库根目录**依次执行，假设已安装 Conda。每一步成功后再执行下一步。

```bash
conda env create -f environment-hf5-server.yml
conda activate colossalai-hf5-gpu

python -m pip install pip==26.2.1 setuptools==81.0.0 wheel==0.48.0
python -m pip install -r requirements-hf5-cu128.txt
python -m pip install -r requirements-hf5.lock.txt

BUILD_EXT=0 python -m pip install -e . --no-deps --no-build-isolation
```

CUDA wheel 版本组合来自 [PyTorch 官方版本页](https://pytorch.org/get-started/previous-versions/)。必须在新环境中先安装 CUDA 文件，再安装普通依赖文件，避免使用其他来源的 Torch 构建。普通清单中的 `torch==2.11.0` 也允许 `2.11.0+cu128`，后续用 `torch.version.cuda` 确认 CUDA 构建。

Python 的精确补丁版本在 Linux Conda 源上的可用性尚未实测；如果解析失败，保留错误日志，不要把失败当作环境配置完成。其他包在 Linux 上也可能需要不同平台的 wheel 或编译工具。安装器会补齐平台专属的传递依赖，其版本尚未被本次 Mac 快照锁定。

`--no-deps` 只用于最后的 ColossalAI 源码安装，避免仓库旧声明把 Torch/Transformers 降级。它不会补齐缺失依赖。`BUILD_EXT=0` 表示安装时不预编译扩展；运行某些 GPU kernel 时仍可能需要 CUDA Toolkit 和编译工具。此轮不要直接安装原有 `requirements/requirements.txt` 或 `requirements/requirements-test.txt` 来补包，它们仍包含旧版约束。

## 3. 验证安装结果

```bash
nvidia-smi
python - <<'PY'
from pathlib import Path
import torch
import torch.distributed as dist
import transformers
import colossalai
from packaging.version import Version

print('torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('transformers:', transformers.__version__)
print('colossalai:', colossalai.__file__)
print('GPUs:', torch.cuda.device_count())
assert Version(torch.__version__).base_version == '2.11.0'
assert torch.version.cuda == '12.8'
assert transformers.__version__ == '5.17.0'
assert Path(colossalai.__file__).resolve().parent == Path.cwd() / 'colossalai'
assert torch.cuda.is_available()
assert dist.is_nccl_available()
x = torch.arange(6, dtype=torch.float32, device='cuda').reshape(2, 3)
torch.testing.assert_close((x @ x.T).cpu(), torch.tensor([[5., 14.], [14., 50.]]))
print('Source import and basic CUDA computation: PASS')
PY
```

这一步只验证环境和一个 CUDA 计算，不代表模型数值对齐或多卡通信通过。接着按照 [Qwen GPU 验证指南](logs/2026-09-22-qwen-gpu-validation/GPU-validation-guide.md) 执行单卡数值检查，再收集和运行 TP/PP/SP 测试。

## 4. 已知缺口：当前 pip check 不通过

留档时实际发现：

- 仓库声明 `torch>=2.2.0,<=2.5.1`、`transformers==4.51.3`、`peft>=0.7.1,<=0.13.2`，与当前安装的 2.11.0、5.17.0、0.19.1 不一致。
- 仓库声明但本地未安装：`diffusers`、`fabric`、`fastapi`、`google`、`pre-commit`、`protobuf`、`ray`、`rpyc`、`sentencepiece`、`uvicorn`。

这些缺口被如实保留，没有为了让快照看起来完整而新增未经验证的包。已完成的 Qwen 本地测试不覆盖所有仓库功能；原有 GPU 测试收集会经过 model_zoo 的聚合导入，之前已经遇到缺少 `diffusers`。完整 GPU 测试仍需按实际导入错误补齐依赖、确认 HF5 兼容版本并重新留档。不要把所有 `pip check` 错误都视为可以忽略：上面是已知情况，新增冲突需要单独排查。

## 5. 服务器验证后补充精确快照

在同一服务器环境、仓库根目录执行：

```bash
hf5_run_dir="logs/hf5-server-env-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$hf5_run_dir"
python -m pip freeze --all > "$hf5_run_dir/pip-freeze-raw.txt"
python -m pip check > "$hf5_run_dir/pip-check.txt" 2>&1
python -m pip inspect > "$hf5_run_dir/pip-inspect.json"
conda env export > "$hf5_run_dir/conda-server.yml"
nvidia-smi > "$hf5_run_dir/nvidia-smi.txt"
git rev-parse HEAD > "$hf5_run_dir/git-commit.txt"
git diff --binary > "$hf5_run_dir/working-tree.patch"
git ls-files --others --exclude-standard > "$hf5_run_dir/untracked-files.txt"
python - <<'PY' > "$hf5_run_dir/requirements-server-resolved.txt"
import importlib.metadata as md
import re
pins = []
for d in md.distributions():
    name = re.sub(r'[-_.]+', '-', d.metadata['Name']).lower()
    if name != 'colossalai':
        pins.append(f'{name}=={d.version}')
print('# Installed server package versions; install transferred ColossalAI source separately.')
print('# CUDA wheels require the matching PyTorch index; this file has no artifact hashes.')
print('\n'.join(sorted(pins)))
PY
```

`pip check` 当前可能返回非零，需阅读其输出；如果 shell 开启了 `set -e`，在保存后续快照前先处理这一退出。记录中包含 Linux 新增的 NVIDIA/Triton 等版本。Git diff 不包含未跟踪文件的内容，仍需保存实际新增源码及测试文件。只有完成相应 GPU 验证后，才能把服务器快照标记为已验证基线。
