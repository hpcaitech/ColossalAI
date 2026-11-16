# 六机 48 卡训练环境（node1/5/6/7/8/9）

本篇记录 2026-11-12 完成的六机环境统一结果，供后续 kernel 优化与长跑复现。
环境指纹由 `/home/niuzifan/scripts/env_fingerprint.sh` 采集，六台输出逐字段相同。

## 一、机器

| 别名 | hostname | 控制面 IP | RoCE 段（/etc/hccn.conf） | 本地盘 | 备注 |
| --- | --- | --- | --- | --- | --- |
| node1 | npu-910b2-46 | 192.168.0.46 | 10.52.66.122–129 | 2.4 T 可用 | 同时跑其他任务，长跑不排它 |
| node5 | npu-910b2-6 | 192.168.0.6 | 10.52.66.90–97 | 2.9 T 可用 | 作为 6 机主节点（MASTER_ADDR） |
| node6 | npu-910b2-98 | 192.168.0.98 | 见 hccn.conf | 3.8 T 可用 | 本次新接入，从 node5 引导 |
| node7 | npu-910b2-138 | 192.168.0.138 | 10.52.66.114–121 | 3.9 T 可用 | |
| node8 | npu-910b2-114 | 192.168.0.114 | 见 hccn.conf | 3.3 T 可用 | |
| node9 | npu-910b2-88 | 192.168.0.88 | 10.52.66.2–9 | 2.9 T 可用 | 替代原 node4 位 |

全部 8×910B2（64 GB/卡），驱动 25.2.1（V100R001C21SPC009B220），`/home` 是同一份 NFS（物理盘在 node4）。

## 二、统一环境

| 项 | 值 |
| --- | --- |
| 镜像 | `ascendai/torch-npu:2.13.0rc1-cann9.1.0-910b-openeuler24.03-py3.12`（ID `23112526d56c`） |
| 容器 | `nzf-cai-dev`，`--privileged --network host --shm-size 512g --user 0:0`，工作目录 `/home/niuzifan/src/ColossalAI` |
| venv | 本地盘 `/mnt/disk0/jiangzs/nzf/cai-npu`（容器内路径相同），`include-system-site-packages=true` |
| 仓库 | 本地盘 `/mnt/disk0/jiangzs/nzf/ColossalAI`，训练时用 `PYTHONPATH` 覆盖 NFS 上的 editable 安装 |
| 启动器 | `/home/niuzifan/scripts/dev-container-node.sh`（统一容器脚本）、`train_scale_inner.sh`（任意机数 × 8 卡） |

版本矩阵（六台一致）：

```
python 3.12.13 | torch 2.13.0+cpu | torch_npu 2.13.0rc1 | torchvision 0.28.0+cpu
diffusers 0.40.0 | transformers 5.17.0 | tilelang 0.1.4+ubuntu.24.3.cann910
NPU 8/机 | repo train_sft.py md5 fbc5dd44cf76c1bf8f801907f9d4b623
```

## 三、为什么 venv 必须在本地盘

NFS 上的 venv 单进程 `import transformers` 要 100 s 以上，`spawn` 的 dataloader worker 会成片卡在 NFS RPC 上。
把 venv 与仓库放到本地盘后，四段 import（torch/transformers/diffusers/colossalai）合计 14–15 s。
node4 上没有 `jiangzs` 目录（它的 `/mnt/disk0` 就是 NFS 导出根），所以早期那台用的是 `/var/tmp/nzf`。

## 四、TileLang 接入

### 4.1 安装

- 来源：`tile-ai/tilelang-ascend` 的 release `TileLang-ascend-v0.1.2.000-release`，资产
  `tilelang-0.1.4+ubuntu.24.3.cann910-cp312-cp312-linux_aarch64.whl`（53 MB，放在共享 `/home/niuzifan/`）。
- 集群到 GitHub 不通（`git ls-remote` 25 s 超时），但 PyPI 通，所以轮子先在本地下载再送进来。
- 安装：`pip install --no-deps <wheel>`，再补运行时依赖
  `cloudpickle ml-dtypes psutil pybind11 Cython tqdm typing-extensions absl-py attrs decorator PyYAML tornado`。
  不能带 `--deps`：依赖里写着 `torch` / `torch-npu`，pip 会去 PyPI 拉一份新的把昇腾版覆盖掉。
- 没有装的可选依赖：`pandas / matplotlib / scipy`（只被上游示例与测试用到，内核开发用不到）。

### 4.2 运行时环境变量（已写进 `dev-container-node.sh` 的 `PRE`）

| 变量 | 值 | 为什么 |
| --- | --- | --- |
| `ASCEND_HOME_PATH` | `/usr/local/Ascend/ascend-toolkit/latest` | wheel 会自己探测这个路径，但显式给更稳（官方 README 也要求） |
| `TILELANG_CACHE_DIR` | `/mnt/disk0/jiangzs/nzf/tilelang_cache` | 默认是 `~/.tilelang/cache`，在共享 NFS 上——与 ColossalAI 的 JIT 扩展缓存同一类问题 |
| `OMP_NUM_THREADS` | `8`（由训练启动器设） | 8 rank × N worker 共用 192 核，不设上限会互相拖死 |

### 4.3 一个必须知道的坑：不要给内核模块加 `from __future__ import annotations`

`@T.prim_func` 会把 kernel 的参数注解当作对象读取；一旦用了 postponed annotations，注解变成字符串，
解析器在 `def main(` 处直接报：

```
InternalError: Check failed: type_code_ == kTVMObjectHandle (11 vs. 8) : expected Object but got str
```

这个报错指向 `def main(` 那一行，看起来像语法问题，实际是注解被字符串化。**之前三版 conv3d spike 全部失败
都是这个原因**，去掉 future import 后同样的结构立刻能编译。

### 4.4 验证（六台全部通过）

`check_tilelang_env.py` 跑两个 kernel：向量加（Vector 单元）与 1024³ GEMM（Cube、L1/L0C、`T.gemm_v0`），
都与 torch 结果 `assert_close` 对拍。

| 节点 | vec_add 编译 / 运行 | GEMM 编译 / 运行 | 缓存目录 |
| --- | --- | --- | --- |
| node1 | 10.1 s / 0.98 ms | 10.2 s / 0.82 ms | 本地盘 |
| node5 | 0.2 s / 0.76 ms（缓存已热） | 10.1 s / 0.65 ms | 本地盘 |
| node6 | 9.4 s / 0.77 ms | 8.8 s / 0.64 ms | 本地盘 |
| node7 | 9.1 s / 0.76 ms | 9.0 s / 0.79 ms | 本地盘 |
| node8 | 9.4 s / 0.75 ms | 8.8 s / 0.67 ms | 本地盘 |
| node9 | 8.7 s / 0.81 ms | 8.8 s / 0.62 ms | 本地盘 |

冷编译约 9–10 s/核，热缓存后 vec_add 只有 0.2 s；两个 kernel 的运行时间都在 1 ms 内，与 torch 对拍全部 `OK`。
`ASCEND_HOME` 六台都自动探测为 `/usr/local/Ascend/ascend-toolkit/latest`，`cache enabled = True`。

### 4.5 单条命令自检

```bash
dev-container-node.sh "ASCEND_RT_VISIBLE_DEVICES=0 python /home/niuzifan/scripts/check_tilelang_env.py"
```

## 五、运维注意

- 单条 ssh 会偶发 `banner exchange` 超时：多台同时 pip 安装/编译时尤其明显，重试即可，不是节点故障。
- pip 缓存目录不可写（`container-home` 属主问题），每次安装都要重新下载，批量铺节点时按 2–3 分钟/台预留。
- 容器 512 GB `/dev/shm`：在途量 ≈ `rank × worker × prefetch 2 × 单卡一批字节`，32 样本/卡 + 4 worker 实测水位 190 G；
  256 G 会在第 4–5 步打满并报 `unable to allocate shared memory(shm)`。容器重启才会清空残留的 shm 段。
- 每节点 `OMP_NUM_THREADS=8`（`train_scale_inner.sh` 里设）：不设时限会让 8 rank × N worker 各按满核起线程，互相拖死。
