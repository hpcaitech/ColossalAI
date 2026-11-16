# 两套训练环境：ColossalAI 容器 / 官方 Cosmos recipe 环境

> 已按“不要跑了”停掉所有运行：h20-5 上无训练进程，torch 2.13 环境的重建也已终止。

## 6. ColossalAI 容器

### 6.1 node1（NPU，nzf-cai-dev）

a. 镜像：`ascendai/torch-npu:2.13.0rc1-cann9.1.0-910b-openeuler24.03-py3.12`（12.9 GB，机器上已有，不用拉）。
   容器必须以 root 运行（`--user 0:0`），否则容器内 HCCL 初始化失败；保留 `HwHiAiUser`(1000)、`share`(2017) 附加组，只读挂载 `/etc/hccn.conf` 和 `/mnt/disk0/jiangzs`（后者是 `jiangzhuosong:share 2770`，缺 share 组会读不到）。
b. 参考 `/mnt/disk0/jiangzs/scripts/dev-container.sh`。我们自己的副本：`/home/niuzifan/scripts/dev-container.sh`，只改 `NAME`（`nzf-cai-dev`）和 `ROOT`（`/home/niuzifan`）两个变量。不带参数进交互 shell，带参数就在容器里执行那条命令；改 flags 要先 `docker rm -f nzf-cai-dev`。命令写进脚本文件再 `bash ~/scripts/x.sh`，不要把引号嵌进 `dev-container.sh bash -c "..."`（会丢 `cd`）。
c. venv 各建一个，互不干扰：`/home/niuzifan/envs/cai-npu`（镜像自带 py3.12；torch 2.13.0+cpu、torch_npu 2.13.0.rc1、transformers 5.17.0、colossalai 0.5.0、diffusers 0.40.0）。venv 里没有 `torchrun`，用 `python -m torch.distributed.run`。
d. 代码：`/home/niuzifan/src/ColossalAI`（分支 `npu/dev`）；注意力开发在 worktree `/home/niuzifan/ColossalAI-npu-attention-entry`（`feat/npu-attention-entry`）。
e. 注意：`export PYTHONPATH=<repo>:$PYTHONPATH` 必须追加，覆盖会丢 CANN 的 `tbe`/`impl`（eager 正常，`torch.compile(backend="npu")` 会报 Failed to initialize GE）；`npu-smi` 的 AICore% 不可信，判断占用看进程表和 HBM 基线；同卡上有别的进程会抬高步时，性能数字要在设备空闲时测。

### 6.2 h20-5（GPU，nzf-cai-dev-h20）

a. 镜像：`hews-colossalai-dev:h20-ffmpeg-20261109`（14.4 GB，基于 `torch2.13-cu130-tf5.17.0-fmt-20261103`，内置 FFmpeg 4.4.2）。
b. 长驻容器会掉 GPU（容器内 `nvidia-smi` 报 NVML Unknown Error 或 torch 报 No CUDA GPUs，宿主正常，`docker restart` 只顶一两分钟）。GPU 活都走一次性容器：`/home/niuzifan/scripts/oneshot.sh <cmd...>`（flags 同 dev-container.sh，另加 `--rm`）。FFmpeg 装在容器层，所以必须用上面这个已 commit 的镜像，否则重建容器后 torchcodec 缺 `libavutil.so.56`。
c. venv：`/home/niuzifan/envs/cai-h20`（py3.10.12、torch 2.13.0+cu130、transformers 5.17.0）；宿主直跑用 `~/opt-venv`（镜像 venv 的拷贝）+ `user-venv.pth` 桥到该 venv，FFmpeg 库在 `~/opt-libs`，运行时 `LD_LIBRARY_PATH=~/opt-libs`。
d. 这套容器/venv 只跑 ColossalAI 训练，官方 Cosmos 的依赖不装进来。

## 7. 官方 Cosmos recipe 环境（h20-5 宿主 venv）

a. 代码：`/home/niuzifan/cosmos/cf-official` —— `NVIDIA/cosmos-framework` @ `bbd4a67` 的干净副本（本地 `git clone --shared`，LFS 跳过）。相对官方只有：新增 Edge 配方文件 `action_policy_droid_edge.py`、`configs/base/config.py` 里一行注册（官方只发 Nano 配方），以及 3 个宿主健壮性补丁（`device_monitor` 的 NVML 回退、`distributed` 的 pynvml 容错、`action_sft_dataset` 的 `filter_dict_source_prefix` 参数）。
b. 解释器：`/home/niuzifan/opt/pbs/py313`（CPython 3.13.18，python-build-standalone，走 `mirror.nju.edu.cn`；本机 GitHub 直连不通）。uv 用 0.11.29（仓库 `required-version` 钉死），在 `/home/niuzifan/envs/uv-0.11.29/bin/uv`。
c. venv：`/home/niuzifan/cosmos/cf-official/.venv`。安装命令（官方 cp313 轮子都要求 py3.13）：
   - 文档推荐栈：`uv sync --extra=train --group=cu130-train` → torch 2.10.0+cu130、torchvision 0.25.0、torchcodec 0.10.0、flash-attn-3-nv 1.0.3、flash-attn 2.7.4、natten 0.21.6.dev6、transformer-engine 2.12、torchao 0.16.0、triton 3.6.0。
   - torch 2.13 栈：`--group=cu130-torch213-train` → torch 2.13.0+cu130、natten 0.21.6+cu130.torch213 等；该组里 flash-attn-3-nv/flash-attn 被注释掉（无 torch213 轮子）。
d. 官方轮子与网络：natten / flash-attn-3-nv / flash-attn / transformer-engine 的 wheel 都托管在 GitHub Release（官方索引页直接指向 GitHub）。GitHub 不可达，所以用 `gh-proxy.com` 下载到 `~/cosmos/wheels-official-t210`（torch210）和 `~/cosmos/wheels-official`（torch213），sha256 与官方索引逐一对过；`uv sync` 时用 `--no-install-package` 跳过这些包，再本地安装。`lerobot==0.4.4` 从 PyPI 装（`--no-deps`，DROID 数据模块 import 它）；`megatron-core`、`evdev` 不装（git 依赖不可达，且训练路径不用）。
e. 运行：`bash /home/niuzifan/scripts/run_policy_edge_official.sh`（GPUs 4–7，TOML `/home/niuzifan/cosmos/bench/action_policy_droid_edge.toml`）。环境必须带 `LD_LIBRARY_PATH=~/opt-libs`（torchcodec）、`HF_HOME=~/.hf_home`、`HF_HUB_OFFLINE=1`，并把 `.venv/bin` 放进 `PATH`（框架会 subprocess 调 `uv` 拉 HF checkpoint，否则 `FileNotFoundError: 'uv'`）。
f. 现状：torch 2.10 的完整 venv 保留在 `/home/niuzifan/cosmos/venv-official-t210`（12 GB，可导入，TE FusedAdam 可用）；torch 2.13 的重建已停止，`cf-official/.venv` 目前是 72 KB 的空壳。
g. 已知限制：torch 2.10 + triton 3.6 在这台 H20 上编译语言区反向图失败（Inductor 的 `triton_per_fused_...` 融合核需要 254016 B 共享内存，硬件上限 232448 B；`persistent_reductions=0`、`max_tiles=1` 都不改变）。torch 2.13 能编译，但 VAE AOT 的数值门会拒绝 16f/24f 块、`encode` 回退 eager（此前实测：误差 ~0.3，latent 均值 0.73；框架 docstring 记录过 2.12/2.9 的同类问题）。
