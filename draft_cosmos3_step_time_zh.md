# Cosmos 3 Edge × DROID 策略后训练：4×H20 单步耗时实测与全量训练估算

- 日期：2026-11-09
- 机器：h20-5（8× NVIDIA H20 96 GB，本次只用后四张，即物理卡 4/5/6/7）
- 代码：官方 `NVIDIA/cosmos-framework` @ `bbd4a67`（浅克隆）
- 配方：`action_policy_droid_edge`（Cosmos3-Edge 基座 + DROID 动作策略后训练）
- 数据：`Cosmos3-DROID` success split，480 分辨率、33 帧窗口、`concat_view`

## 一、结论

| 指标 | 实测值 |
| --- | --- |
| 单卡每步样本数 | **32**（`max_samples_per_batch=32`，不做 token packing 截断；每步 473k token/4 卡 ≈ 118k/卡 ≈ 3.7k/样本，与 32 样本吻合） |
| 稳态单步耗时 | **29.5 s/步**（12 步里第 2–12 步 29.27–30.74 s，中位 29.49 s） |
| 单步构成（框架自带计时） | **VAE 编码 18.1 s（61.5%）** + 模型前反向/优化器/通信 11.4 s |
| 单样本耗时 | 0.92 s/样本/卡（其中 VAE 0.57 s） |
| 4 卡吞吐 | 128 样本/步 ÷ 29.5 s = **4.34 样本/s**；16.1k token/s |
| 显存占用 | 约 **54 GB / 96 GB** 每卡（activation checkpointing=full，bf16） |

官方训练计划是 **10000 步 × 全局 batch 8192 = 81.92M 样本**（Edge 配方注释：64 卡 × 128 样本/卡）。
按实测 0.922 s/样本/卡线性折算：

| 规模 | 全量（81.92M 样本） | 本地 DROID 一遍（16.34M 窗口） |
| --- | --- | --- |
| 4 卡（本次） | 640,000 步 ≈ **218 天** | ≈ **43.6 天** |
| 8 卡（整机） | ≈ 109 天 | ≈ 21.8 天 |
| 64 卡（配方自身布局） | ≈ 13.7 天 | ≈ 2.7 天 |
| 256 卡 | ≈ 82 h ≈ 3.4 天 | ≈ 16 h |

> 折算假设：每卡每步仍是 32 样本，单步耗时随卡数基本不变（通信增量未计），
> 且不含验证（每 100 步）与 checkpoint 保存（每 100 步，实测一次约 47–72 s）。

## 二、实测明细（12 步）

日志：`~/cosmos/outputs/policy_edge_bench/logs/policy_edge_bench_1109_171254.log`

| 迭代 | 耗时 (s) | 说明 |
| --- | --- | --- |
| 1 | 753.28 | `torch.compile` 首次编译（`compiled_region=language`） |
| 2 | 30.39 | |
| 3 | 30.74 | |
| 4 | 226.65 | VAE 480 桶的 AOT 编译（`compile_tokenizer` 回调，`compile_after_iterations=3`） |
| 5 | 29.73 | |
| 6 | 29.63 | |
| 7 | 29.45 | |
| 8 | 29.49 | |
| 9 | 29.49 | |
| 10 | 29.69 | |
| 11 | 29.49 | |
| 12 | 29.27 | |

一次性开销（不计入单步耗时）：

| 阶段 | 耗时 |
| --- | --- |
| 数据预热带（64 个 spawn worker 冷启 DROID 数据集） | 602 s |
| 首次编译（第 1 步） | 753 s |
| VAE AOT 编译（第 4 步） | 227 s |
| 收尾 checkpoint 保存（1.42B 参数） | 47 s |

训练结束后 `exit 0`，第 12 步正常落盘，本次测量有效。

## 三、单步耗时构成（框架 IterSpeed 实测）

用 `++trainer.callbacks.iter_speed.hit_thres=2` 重跑 6 步拿到框架自带的拆解
（日志 `~/cosmos/outputs/policy_edge_breakdown/logs/policy_edge_bench_1109_180534.log`）：

| 迭代 | iter_speed | token/步（4 卡） | VAE `encode` | `prepare_data`（tokenize+VAE） |
| --- | --- | --- | --- | --- |
| 5 | 29.42 s | 472,959（16.1k tok/s） | 18.096 s（61.5%） | 18.114 s（61.6%） |
| 6 | 29.49 s | 474,908（16.1k tok/s） | 18.093 s（61.3%） | 18.111 s（61.4%） |

- **VAE 编码占 61%，模型前反向 + 优化器 + 通信约 11.4 s。**
- 单样本 VAE 摊 0.57 s（批量 32 一起编码，比单样本串行的 0.658 s 快 ~15%）。
- 每步 473k token（全 4 卡），约 3.7k token/样本，与 32 样本/卡吻合。
- 结论：这一步是 **VAE 主导**，不是模型算力主导。想压低单步时间优先看 VAE
  （AOT 已开、收益仅 ~2%；可考虑 latent 缓存/更大 VAE 批/更快的 VAE 实现），
  其次是放宽 activation checkpointing（涨显存换速度）。

## 四、为什么此前一直「卡住」——根因与修复

1. **TE FusedAdam 缺失**：配方默认 `FusedAdam`（transformer_engine），镜像里没有该包。
   本地 Edge 实验改成 `optimizer_type="AdamW"`（torch 自带 fused AdamW，同样的更新式）。
2. **注意力后端（真正的拦路虎）**：训练走 varlen causal attention，
   - cuDNN 后端不支持 varlen（`Varlen for cuDNN Attention is not integrated yet`）；
   - Hopper 上官方推荐的 flash-attn-3-nv、以及 NATTEN、flash-attn v2 都没装；
   - 官方 wheel 源（`nvidia-cosmos.github.io/cosmos-dependencies/v1.5.0`）只有 **cp313** 轮子，
     而本环境是 Python 3.10.12 + torch 2.13.0+cu130，装不上。
   修复：按 `nvidia-cosmos/cosmos-dependencies` 的构建方法，从源码编译
   **NATTEN v0.21.6**（`sm_90a`，只编 H20 一个架构，64 并发，约 6 分钟），
   CUTLASS 用 NATTEN 子模块钉住的 v4.3 头文件（blobless + sparse checkout 只取 `include/`）。
   装进 `~/envs/cai-h20`（容器与宿主共用）。验证：后端探测 natten dense/varlen 均为 True，
   训练实际选中 `hopper-fmha`。
3. **宿主 `uv` 包装器自递归**：`python -m uv` → PyPI shim 找到 `bin/uv`（就是它自己）→ 无限 exec 循环，
   CPU 100% 卡死。修复：包装器直接 exec `~/envs/cai-h20/bin/uv`（37 MB 真二进制）。
4. 其它健壮性修复：`device_monitor` NVML 不可用时回退 torch；`run_policy_edge_bench.sh`
   正确传出退出码；重试脚本按「是否已过数据预热」决定是否重试（避免 GPU 注入抖动导致的盲目重试）。

## 五、口径、假设与偏差

- 基座 checkpoint 为空（随机初始化）：权重不影响耗时；但**没有**做真实加载与 EMA 预热。
- `precision=bfloat16`，`data_parallel_shard_degree=4`，`activation_checkpointing=full`（TOML 默认值，
  与官方一致），`torch.compile` 开（官方一致）。
- 33 帧窗口来自配方 `chunk_length=32`；VAE `encode_exact_durations=[33]`。
- 注意力后端是 **NATTEN 0.21.6 hopper-fmha**，不是官方 Hopper 首选的 flash-attn-3-nv
  （无 py3.10 轮子）。flash3 下模型段可能更快，单步耗时或低于 29.5 s。
- 不含验证（`validation_iter=100`）与定期 checkpoint（`save_iter=100`，每次约 47 s → 摊到每步 +0.5 s）。
- 4 卡的步时外推到 64/256 卡时未计 FSDP 通信增量与共享存储/数据带宽变化。

## 六、复现

```bash
# 宿主直跑（不需要容器，GPU 4-7）
ssh h20-5 'ITERS=12 bash /home/niuzifan/scripts/run_policy_edge_host.sh'
```

- 运行脚本：`~/scripts/run_policy_edge_host.sh`（宿主）→ `~/scripts/run_policy_edge_bench.sh`
- TOML：`~/cosmos/bench/action_policy_droid_edge.toml`
- NATTEN 轮子：`~/cosmos/wheels/natten-0.21.6-cp310-cp310-linux_x86_64.whl`
- NATTEN 构建脚本：`~/scripts/build_natten.sh`

## 七、待办 / 可选

- 若要贴近官方口径，可再用 `max_samples_per_batch=128` 跑一次（每卡 128 样本），
  单步约 4 倍耗时，但每样本摊销会略好（VAE 占比会更高）。
