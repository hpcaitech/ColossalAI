# Cosmos 3 在线路径优化日志（node6，8×910B2）

本文件是**在线训练路径**的优化工作记录：每一个功能点写清楚改了什么、怎么测、效果多少、结论是什么。
口径统一为单机 8 卡、`--samples-per-rank 32`、`--grad-accum 1`、`--hsdp-shard-size 8`、4 worker、
`--image-augmentation torchvision`、VAE bf16 + fast encoder + encode-batch 8、480p / 33 帧。

测试环境：node6（npu-910b2-98），容器 `nzf-cai-dev`，本地盘仓库 `/mnt/disk0/jiangzs/nzf/ColossalAI`，
本地盘 venv `/mnt/disk0/jiangzs/nzf/cai-npu`，`COLOSSALAI_EXTENSION_CACHE` 指本地盘。
标准入口：`scripts/bench_online_step.sh`；指标汇总：`scripts/summarize_online_metrics.py`；调度 A/B：
`scripts/ab_online_scheduling.sh`。

> **当前状态（2026-11-15 00:15 UTC）**：VAE 侧 kernel 优化收尾（融合 norm + 两趟 + AvgDown 零宽 pad，
> 全部逐位一致），单机稳态 **24.9–27.5 s/步**、VAE 段 **10.0 s/步**、显存守卫验证通过（平在 55.1 GiB）；
> profile 的两个新结论见 F14（四类搬运全部到顶）与 F15（DiT norm 1.33 s/步，逐位不可复刻，厂商算子
> 0.0001% → `--dit-norm-kernel` opt-in）。**host 侧已被排除**（F15.1 的逐调用点归因：全是同步/背压，
> 不是派发成本）；**数据侧的周期性停摆已根治**（OMP4 + 编译版 ColorJitter，F16），只剩 2–3 s 常规 wait；
> **跨步 VAE/bwd 重叠已被实测否定**（两条流被设备串行化，只藏住 2%，见文末）。
> 本文所有数字都来自 node6/node1 真机。

## 两套可部署配置（2026-11-15 收尾，直接照着选一套）

以下数字全部是 node1+node6 **16 rank、6 步、`--max-episodes 2048`** 的稳态中位数；单机 8 卡一并列出。

| | **配置 A：严格逐位** | **配置 B：A + `--dit-norm-kernel`** |
| --- | --- | --- |
| 16 rank 步时 | 27.37 s | **26.48 s（−3.3%）** |
| 16 rank 吞吐（16×32=512 样本/步） | 18.7 样本/s | **19.3 样本/s** |
| 16 rank MFU | 26.9% | **27.8%** |
| 单机 8 卡步时 / 吞吐 / MFU | 27.0 s / 9.5 样本/s / 27.2% | **25.82 s（20 步）/ 9.91 / 28.3%** |
| device used | 56.2 GiB | **51.1 GiB** |
| VAE / fwd / bwd / wait | 10.04 / 2.84 / 11.54 / 2.49 s | 10.07 / **2.35** / 11.69 / 2.72 s |
| 与上游数值关系 | **全部逐位一致** | 前向每 100 万元素 1.2 个偏 1 ulp；**反向与优化器逐位一致** |

两套共用：`--vae-norm-kernel`（两趟）+ `--fast-vae-encoder` + `--vae-content-pad 16` +
`OMP_NUM_THREADS=4` + `--worker-threads 4` + `COSMOS3_FAST_JITTER=1` + `--dit-sum-mean-norm`
+ `TORCHINDUCTOR_CACHE_DIR=<本地盘>/ext_cache/inductor`，4 worker，编码 batch 8。**B 只多一个开关**，
`--no-dit-norm-kernel` 即回到 A。

选 A 还是 B 是唯一的策略问题（见 F15 的建议）；其余每一项都已在 16 rank 上验过，且**没有多机衰减**。

**跨机器可复现**（部署前必查项）：配置 A 在 **node1 单机**独立跑 5 步（4 个稳态）是
**26.79 s 中位数、VAE 9.90、fwd 2.83、bwd 10.91、wait 2.81、device used 55.89 GiB、9.55 样本/s**，
与 node6 的 27.08 s / 9.48 样本/s 在 1% 以内一致——tilelang 内核按节点各自编译、inductor 缓存各在本地盘，
这套配置换机器不需要改任何东西。

## 结论摘要（滚动更新）

- **基线**（node6，8 卡，32 样本/rank）：32.78 s/步、7.81 样本/s；VAE 15.19 s（46.3%）、fwd 2.76、
  bwd 11.99、optim 0.29、data wait 2.2–4.0 s；峰值 40.3 GiB allocated / 47.3 GiB device；
  MFU 22.4%（transformer）/ 31.5%（含 VAE）。
- **最大瓶颈不是 VAE，是数据供给**：PyTorch DataLoader worker 被强制单线程，32 个 worker 只供给
  ~3–4 样本/s，而训练需要 7.8 样本/s；首步 wait 325 s、step 5 停摆 127–130 s 都是这个原因。
- **VAE 配置空间是平的**（0.449–0.472 s/样本），不值得再扫；VAE 侧唯一可观的杠杆是**裁掉画布 padding**：
  实测 pad 16（宽 656）0.377 vs 完整 736 的 0.461 s/样本（**−18.2%**），relL2 1.48e-2（与 bf16 VAE 计算噪声
  ~1.9e-2 同量级）；保守的 pad 64（704）0.431（−6.5%），relL2 2.48e-3（**低于 bf16 存盘噪声**）。
  训练循环里实测（8 样本尺度）：step 8.01→**7.34 s（−8.4%）**、VAE 3.88→**3.17（−18.3%）**、
  MFU 23.0%→**25.0%**、显存 24.4→22.9 GiB。
- **已实现**：action memmap 缓存（逐位一致、去掉 parquet ~7% 的 worker CPU，但**不修停摆**）、
  `--worker-threads`（在 `worker_init_fn` 里覆盖 PyTorch 的 1 线程）、`--vae-content-pad`（画布裁剪）、
  `--video-uint8`（数据直通，node1 A/B **−8.8% 步时**、wait 3.12→0.50 s）、`--vae-norm-kernel`
  （融合通道优先 RMSNorm，**latent 逐位一致**，编码 **−16.1%**，整步 **−6.8%**，见 F8）、
  `--profile-step`（稳态步 profile，见 F10）。
- **当前最好整步**（2026-11-15 收尾，node6，`fastjit` 臂 8 步）：**26.8 s 稳态均值**
  （`--vae-norm-kernel` 两趟 + AvgDown 零宽 pad + `OMP_NUM_THREADS=4` + `COSMOS3_FAST_JITTER=1`，
  4 worker × 4 线程），**无 50 s 级尖峰**（基线 28.9 s 均值、最大 38.7 s）。分解：VAE 10.0–10.1、fwd 2.8、
  bwd 10.3–11.6、optim 0.4–0.5、wait 1.8–3.0。相对本文件开头的 32.78 s 基线 **−18%**。
- **多机最终基准（F18，2026-11-15）**：node1+node6 16 rank、6 步、`--max-episodes 2048`：
  **27.24 s/步、18.8 样本/s、MFU 27.0%、VAE 9.92 s（36.4%）、device_used 56.2/61 GiB、
  16 个 rank 全部 27.22–27.24 s**。单机 8 卡 26.8 s / 9.55 样本/s → **同一步时下吞吐翻倍（96% 扩展）**，
  VAE 优化在多机上是线性叠加的。
- **多机验证（F13）**：2 机 16 rank（node1+node6）norm_off 31.5 s → norm_on **26.8 s（−15%）**，
  VAE 15.05→12.91 s——单机收益在多机成立；但 16 rank 下显存 47.7→**58–60/61 GiB**（逐步增长），
  已加自适应 `empty_cache` 保护（free < 3 GiB 时触发）。
- **profile 的另一面（F14）**：VAE 编码 **99.9% busy、零 idle、零 launch-bound、elementwise 连跑只有
  0.72 ms**——VAE 侧只剩四类"搬运"，且实测都已到顶：`cat` 1261 GB/s、Transpose ≈1.0 TB/s（带宽 bound），
  pad 345 GB/s（只有 59 ms，已跳过零宽 pad），conv TransData 无法用预转 weight 格式消掉。
- **稳态步归属修正（F15.1）**：`step_trace_time` 显示通信 11.36 s 里 **9.51 s 是重叠的**，只有 1.85 s
  暴露；此前"通信零重叠"的结论作废（当时只和 matmul/conv 比对）。真正的大头是 computing 21.7 s。
- **DiT 的 norm 不是融合算子（F15）**：`hidden_act="relu2"` 选中的 `Cosmos3NemotronRMSNorm` 是 5 个 fp32
  kernel 的分解链，profile 里 **394 条链/步 = 1.33 s（5.3%）**。行归约 24 种候选全部无法与 `mean(-1)`
  逐位一致（最好差 35% 的行 1 ulp），**所以自研融合 kernel 在逐位约束下关闭**；厂商 `npu_rms_norm` 只差
  0.0001% 的元素（每 90 万 1 个 1 ulp），反向可做到逐位一致 → 已实现 `--dit-norm-kernel`（**默认关**，
  开则约 −0.9~1.3 s/步）。
- **两趟 RMSNorm（F13.2，默认启用）**：大 shape 358→756 GB/s（5.11→3.62 ms），**逐位一致**，
  阈值 S ≥ 65536；预计整步 −1.7%。
- **已排除的方向**（都有实测）：conv 自研（74% MFU）、channels_last_3d（torch_npu 不支持）、
  梯度桶大小（无变化）、异步 reduce-scatter（无变化，串行化在设备侧）、fused norm+silu（无 swish 原语，
  无法逐位一致）、AvgDown3D 的 stride 归约（不逐位一致且慢 89%）。
- **明确不做**：继续扫 VAE 配置、`--overlap-allgather`（上限 0.3 s）、grad-accum 提速（~0.3%）、
  transformer elementwise 融合（<1%）。

## 测试规范（所有实验共用）

- **路径**：代码/venv/扩展缓存/日志/指标全部在本地盘；NFS 只读 checkpoint 与数据集。日志
  `/mnt/disk0/jiangzs/nzf/bench_online/logs/<tag>.log`，指标 `.../runs/<tag>/metrics/rank-*.jsonl`，
  A/B 结果 `.../runs/<tag>/`，中间产物（VAE sweep / canvas）也在 `bench_online/` 下。
- **标准命令**（容器内）：
  `bash .../scripts/bench_online_step.sh <tag> 32 <iters> [额外参数]`；默认 8 卡、32 样本/rank、
  grad-accum 1、HSDP shard 8、4 worker、torchvision 增强、VAE bf16 + fast encoder + eb8。
- **指标**：每步 `perf/step_ms`、`data/vae_ms`、`perf/forward_ms`、`perf/backward_ms`、`perf/optim_ms`、
  `data/wait_ms`、`mem/peak_gib`、`mem/device_used_gib`；汇总脚本按"最慢 rank"给 step、按 rank 求和给
  samples、按 `samples/step_s` 给吞吐、按 20.34 TFLOP/样本 ×4 与 353.9 TFLOPS/NPU 给 MFU（可加 VAE 33 TFLOP）。
- **A/B 纪律**：固定 seed、固定 `--max-episodes 8`（避免数据文件边界停摆）、4 步、丢掉 step 1（冷启动）、
  至少比较 step 2–4 的中位数；任何改动先过 loss/数值门禁再谈性能。
- **卡住时先 py-spy**：`py-spy dump --pid <rank>` 与 `<worker>` 一次即可区分 rank 在 `_get_data`、
  worker 在 `adjust_hue_image`/parquet/mp4；比猜"HCCL 卡死"快得多。本轮首步与 step 5 的根因就是这样定的。
- **已知非稳态**：首步冷启动（worker spawn + 首 batch 解码/增强 + VAE/模型 kernel 加载）可达数分钟；
  64-episode 子集在 step 5 会出现 130–160 s 的 data_starvation（根因是单线程 worker 供给不足，不是 parquet）。
- **日志格式**：`bench_online_step.sh` 不经过 `train_scale_inner.sh` 的行前缀过滤器，`done` 是裸行；
  A/B 脚本的等待模式用 `(^| )done$`，不要用 `] done$`（这是 cache64 第一次被误判为"没跑完"的原因）。
- **端口与 teardown**：`done` 在 torch_npu 的 teardown 之前打印，进程还会占着 rendezvous 端口和 NPU 显存；
  A/B 的每个 case 必须用独立 `MASTER_PORT`（脚本用 `29531 + RANDOM%500`），并在 `done` 后
  `pkill -f train_sft.py` / `torch.distributed.run`，否则下一个 case 会 `EADDRINUSE` 或 OOM。
- **旧日志**：每个 case 启动前必须 `rm -f` 它的旧日志；否则 `wait_log` 会被上一次的 Traceback 立即命中，
  把还没开始的新 case 误判为失败（`ab_targeted.sh` 已按此实现）。
- **代码一致性**（2026-11-15 踩到）：node6 上的 `examples/worldmodel/cosmos3_harness/` 曾缺
  `checkpoint_ledger.py` / `run_context.py` / `training_heartbeat.py`，而工作树同时被别的会话改动过，
  导致"训练能跑但 import 的不是当前代码"。**每次 A/B 前先整体 rsync `examples/worldmodel/cosmos3` 与
  `cosmos3_harness`**，并用 `shasum -a 256` 记录被测文件（**2026-11-15 又踩一次**：只同步了
  `cosmos3/`，而 `train_sft.py` 新引入的 `apply_checkpoint` 只在本地 harness 里有，两台机器直接
  `ImportError`——同步必须两个目录一起；2026-11-15 06:40 UTC 最终被测树（sha256 前 8 位）：train_sft 682794b9 / droid_dataset 22a36b0f /
  packing 74e66084 / vae_norm_kernel 88e4d28f / vae_fast_encoder c2f28cb3 / dit_norm_kernel a1fe279b /
  fast_color_jitter f9ad1b51 / video_pad_norm 8079d8d7 / harness checkpoint 27de9aa3 / train ff76211c /
  training_metrics 20f40791）。
  测试：`pytest examples/worldmodel/cosmos3/tests` **238 passed, 3 skipped**（含并行会话新增的用例，19.8 s）。

---

## F0 基线（2026-11-14，8 步，tag `online_base`）

### 配置

`train_scale_inner.sh online_base ... --samples-per-rank 32 --grad-accum 1 --hsdp-shard-size 8 --max-iter 8
--num-workers 4 --image-augmentation torchvision --vae-encode-batch 8 --fast-vae-encoder`

### 稳态（step 2–4，8 rank 全量）

| 指标 | 值 |
| --- | --- |
| step 中位数 | **32.78 s**（31.12 / 33.03 / 32.77） |
| 吞吐 | **7.81 samples/s**（256 样本/步） |
| 每样本墙钟 | 0.128 s |
| VAE | **15.19 s（46.3%）**，0.474 s/样本/卡 |
| forward | 2.76 s（0.086 s/样本） |
| backward | 11.99 s（0.375 s/样本） |
| optimizer | 0.29 s |
| dataloader wait | 2.22–4.02 s（**7–12%**） |
| 峰值显存 | allocated 40.30 GiB / device 47.33 GiB（61.0 GiB） |
| MFU（transformer only，fwd 20.34 TFLOP/样本 ×4 重算） | **22.4%** |
| MFU（含 VAE 33 TFLOP/样本） | **约 31.5%** |

### 两个非稳态事件（必须记住）

1. **首步 355.78 s**，其中 `wait 324.84 s`（91%）。冷启动的数据侧成本，包含 64 episode 的 parquet 索引、
   action 表首次整列读取、mp4 冷解码与增强。只出现一次，但做 4 步短测时会污染第一行，必须丢掉 step 1。
2. **step 5 155.97 s**，其中 `wait 127.25 s`（82%），`data_starvation` 事件。rank 栈显示部分 rank 在
   `pack_batch`、部分已进入 backward，是**数据侧文件/parquet 边界的周期性停摆**，不是 VAE、不是 HCCL。
   触发点与 `droid_dataset.py::_action_table` 的注释一致（"the first sample past a file boundary paid a cold
   full-column read while the other ranks waited"）。长跑里这会周期性出现，是数据阶段的头号修复项。

### 结论

- 单机 8 卡的在线步时与六机 48 卡的 35.1 s 同量级（单机少了跨节点通信 ~4.8 s），基线可信。
- VAE 占 46%，backward 37%，数据 wait 7–12%。优化顺序应为：**数据停摆/等待 > VAE 配置与重叠 >
  all-gather/梯度累积 > 算子融合**。
- 训练侧 elementwise 融合（RMSNorm/ReLU² 等）按 token 量估算只有 0.1–0.3 s/步（<1%），不是当前主项；
  这条负结果先记下，避免后续重复投入。

---

## F1 VAE 配置扫描（完成：负结果）

`sweep`：`scripts/bench_vae_sweep.py`，单卡、真实 DROID 窗口（8 样本同 canvas，544×736）、
`dtype ∈ {bf16, fp16}` × `chunk ∈ {native, 4, 8, 12}` × `encode-batch ∈ {4, 8}`。
判据：每样本墙钟、峰值 HBM、latent 与 native 的 relL2。

### bf16 结果（8 样本同 canvas，单卡）

| chunk | encode-batch | 8 样本墙钟 | s/样本 | 峰值 HBM |
| --- | --- | --- | --- | --- |
| native (4) | 4 | 3.709 s | 0.464 | 8.43 GiB |
| native (4) | 8 | 3.642 s | 0.455 | 14.89 GiB |
| 4 | 4 / 8 | 3.709 / 3.642 | 0.464 / 0.455 | 8.43 / 14.88 |
| 8 | 4 | 3.737 | 0.467 | 12.85 |
| 8 | 8 | 3.777 | 0.472 | 23.72 |
| 12 | 4 | **3.603** | **0.450** | 17.26 |
| 12 | 8 | 3.662 | 0.458 | 32.56 |

### fp16 结果

| chunk | encode-batch | 8 样本墙钟 | s/样本 | 峰值 HBM |
| --- | --- | --- | --- | --- |
| native (4) | 4 | 3.698 s | 0.462 | 9.76 GiB |
| native (4) | 8 | 3.637 | 0.455 | 16.22 |
| 4 | 4 / 8 | 3.699 / 3.637 | 0.462 / 0.455 | 9.76 / 16.21 |
| 8 | 4 / 8 | 3.729 / 3.765 | 0.466 / 0.471 | 14.18 / 25.05 |
| 12 | 4 / 8 | **3.589 / 3.649** | **0.449 / 0.456** | 18.59 / 33.89 |

结论：**配置空间基本是平的**。native/4 与 4 帧同一路径、数字逐位一致；8 帧更慢且显存翻倍；
全表最快是 fp16 + chunk 12 + eb4（0.449 s/样本），只比当前训练配置（bf16 native eb8，0.455）快 1.3%，
显存却从 14.9 涨到 18.6 GiB；训练循环里的 VAE 是 0.474 s/样本（比孤立测量多 4% 的主循环开销）。
**所以切换 chunk 12 / fp16 最多值 ~0.1–0.2 s/步（0.3–0.6%），不值得引入新的编译与显存风险。**

> 这条是负结果，但很重要：VAE 的配置旋钮已经拧到头了，剩下的只有"少做"（重叠/缓存/裁掉无效画布）
> 或"换实现"（kernel 融合/Conv3D），不要再在 encode-batch / chunk / dtype 上反复试。

---

## F2 调度 A/B（进行中）

`scripts/ab_online_scheduling.sh`，case 列表：base / wt4 / canvas<pad> / ag / w8 / w8o2 / ga4。
每个 4 步、`--max-episodes 8`。**注意首步成本随 batch（256 样本）而不是数据集大小增长**，所以即使 8 episode，
每个 case 的首步仍要 ~5 min；要快速 A/B 用 `SAMPLES_PER_RANK=8`（脚本支持，case ~2 min，相对效应不变）。
loader 本身已经是 `persistent_workers=True`（`train_sft.py` 的 `prepare_dataloader`），所以 8-episode
子集不会每步重启 worker；这一点先确认过，否则 wait 会被 spawn 开销污染。

| case | 改动 | 预期 |
| --- | --- | --- |
| base | 默认 | 对照 |
| ag | `--overlap-allgather` | 参数 all-gather 与下一步 VAE/取数重叠；按 optim 段 0.29 s 估计上限只有 ~0.3 s |
| w8 | `--num-workers 8` | 降低 wait（2–4 s） |
| w8o2 | 8 worker + `OMP_NUM_THREADS=2` | 降 CPU 线程争用 |
| ga4 | `--grad-accum 4` | 摊薄 optimizer/all-gather，按每样本比较 |

（结果待填）

### 快速尺度的 base（`SAMPLES_PER_RANK=8`，用于相对比较）

8 样本/rank、64 样本/步、4 步：首步 134.55 s（wait 124.61，冷启动）；稳态 **8.01 s**（7.97/8.17/8.15），
VAE 3.88（48.4%）、fwd 0.71、bwd 2.54、optim 0.29、wait 0.66；峰值 24.40/31.67 GiB；MFU 23.0%；
8 个 rank 全部 8.00 s（完全均衡）。8 样本的 batch 小，worker 预取能跟上，**wait 只剩 0.66 s**，
所以数据效应的相对幅度会被压缩；`wt4` 的方向仍可判，绝对收益以 32 样本的集成运行为准。

### wt4（`--worker-threads 4`）结果

| 指标 | base | wt4 | 差 |
| --- | --- | --- | --- |
| 首步 | 134.55 s（wait 124.61） | **102.45 s（wait 90.13）** | **−28% wait** |
| 稳态 step | 7.97 / 8.17 / 8.15 | 8.82 / 8.59 / 8.84 | **+8%** |
| VAE | 3.90–3.98 | 3.92–3.94 | 持平 |
| forward | 0.71–0.72 | 0.71–0.73 | 持平 |
| backward | 2.47–2.54 | 2.60–3.05 | +0.1–0.5 |
| optimizer | 0.29–0.33 | 0.45–0.59 | +0.2–0.3 |

解释：8 样本/步时主进程的设备工作少，host dispatch 与 HybridAdam 的逐参数 launch 在关键路径上；
32 个 worker × 4 线程 = **128 线程**与主进程抢 CPU，backward/optimizer 段被拖慢。**冷启动明显受益**
（wait −28%），但 8 样本的稳态是净亏。32 样本生产尺度下主进程设备工作占比更大、wait 2.9 s 的收益也更大，
净效应由集成运行决定；若仍为负，可退到 `--worker-threads 2` 或只给提取/冷启动用。

### canvas16（`--vae-content-pad 16`）结果（默认 1 线程 worker）

| step | VAE | step |
| --- | --- | --- |
| 2 | 3.14 | 7.59 |
| 3 | 3.23 | 7.34 |
| 4 | 3.15 | 7.13 |
| base | 3.90–3.98 | 7.97 / 8.17 / 8.15 |

**汇总（8 样本尺度，8 rank 全量）**：

| 指标 | base | canvas16 | 差 |
| --- | --- | --- | --- |
| step 中位数 | 8.01 s | **7.34 s** | **−8.4%** |
| 吞吐 | 7.99 样本/s | **8.71 样本/s** | +9.0% |
| VAE | 3.88 s（48.4%） | **3.17 s（43.2%）** | **−18.3%** |
| fwd / bwd / optim | 0.71 / 2.54 / 0.29 | 0.71 / 2.55 / 0.30 | 持平 |
| 峰值 allocated / device | 24.40 / 31.67 GiB | **22.92 / 29.38 GiB** | −1.5 GiB |
| MFU | 23.0% | **25.0%** | +2.0 点 |
| 首步 | 134.55 s（VAE 4.35，wait 124.6） | 134.63 s（VAE 3.50，wait 120.5） | — |

backward 在 run 间有 ±0.3 s 的抖动，吃掉了一部分；32 样本下 VAE 占比 46%，预期 step 的绝对收益约 2.5–2.8 s。

## F3 VAE 画布裁剪门禁（扫描完成，待 loss A/B）

`scripts/bench_vae_canvas.py`：把 544×736 画布改成"内容宽 640 + 右侧 pad p ∈ {0,16,32,64,96}"，
测量编码时间与裁剪后 latent（33×40）相对完整画布的 relL2。理论上 p=0 省 13% 的 VAE 工作（~2 s/步），
但 mid-block 的全局 attention 会看到 padding，必须过数值门禁。计划：先跑扫描，再用同种子 20 步
loss A/B 复核最优 p。

**训练/推理一致性**：RoboLab serving 走的是同一套 packing 时，若训练用 crop、推理用完整画布，
latent 分布会错位。门禁分两级：relL2 落在 bf16 噪声内 → serving 可以不改；否则 serving 也必须带同一个
crop 参数（`pack_batch` 是共享的，只需把参数透传）。集成时把这一点写进验收。

**扫描结果（relL2 待末尾统一计算）**：

| pad | 宽 | s/样本 | 峰值 | relL2 vs 736 | 判定 |
| --- | --- | --- | --- | --- | --- |
| 0 | 640 | 0.399 | 13.12 GiB | 3.65e-2 | 超 bf16 VAE 计算噪声（~1.9e-2） |
| **16** | **656** | **0.377** | 13.43 GiB | 1.48e-2 | 在 bf16 VAE 噪声内；激进，省 18.2% |
| 32 | 672 | 0.417 | 13.74 GiB | 6.56e-3 | 中间，省 9.5% |
| **64** | 704 | 0.431 | 14.35 GiB | **2.48e-3** | **低于 bf16 存盘噪声 4.1e-3；保守可部署，省 6.5%** |
| 96（完整） | 736 | 0.461 | 14.96 GiB | 0 | 配方原样 |

三条结论：① 去掉右侧 padding 确实省 ~6–18% 的 VAE 时间；② **宽度不是单调的**——656（41 个 latent 列）
比 640（40）更快，672 又回退，说明这与 conv 的 tiling/向量对齐有关，必须实测选点，不能按"越窄越快"外推；
③ relL2 随 pad 增大快速下降：**pad 64 已低于 bf16 存盘的舍入噪声（4.1e-3）**，是保守部署点；
pad 16 的 1.48e-2 与 bf16 VAE 自身的计算误差（约 1.9e-2）同量级，是激进点，需要更长的 loss A/B 定夺。

**实现**：`resize_and_reflect_pad(..., pad_columns=pad)` 直接 pad 到 `content_w + pad`，不再先 pad 到
736 再切片；单测（540×640 输入，pad 16/32/64）与"全画布 pad 后切前 content_w+pad 列"**逐位相同**。
省掉每样本约 25 MB 的无效 float32 搬运（32 样本/卡约 0.8 GB/rank/step）。

## F4 action parquet → memmap 缓存

- 新增 `scripts/build_action_cache.py`：每个 data parquet 写成 `[rows, 17]` float32 的 `.npy`
  （joint action 7 + gripper 1 + joint state 7 + gripper state 1 + episode index 1）。
- `DroidActionPolicyDataset` 新增 `action_cache` 参数：`_read_actions` 优先走 memmap 切片，
  缺文件时回退 parquet；`train_sft.py` 新增 `--action-cache`。

### 构建结果（node6，容器内）

| split | 文件数 | 行数 | 构建时间 | 吞吐 |
| --- | --- | --- | --- | --- |
| success | 62 | 18,691,281 | 81.2 s | ~230k 行/s |
| failure | 12 | 3,721,431 | 16.0 s | ~235k 行/s |
| 合计 | 74 | 22.41M | **97 s** | 缓存体积约 1.5 GB |

### 正确性门禁（已过）

`DroidActionPolicyDataset` 同参数实例化两份（有/无 `action_cache`），对 274 个窗口逐窗口比较
`_read_actions` 的输出：`max_action_diff = 0.0`、`max_state_diff = 0.0`，**逐位一致**。

### 训练验收（完成：无损，但无稳态收益）

64 episode、6 步、`--action-cache`，与 `online_base` 同 seed/同数据顺序对拍：
- **step 1 loss 20.215628，与原基线逐位一致**——端到端证明 action 缓存无损。
- step 2 loss 20.474→14.750849 vs 基线 14.739662（差 0.08%）：DataLoader worker 的 RNG/预取顺序在两次
  进程间不是完全确定的，不是缓存问题；缓存本身已在 274 个窗口上逐位比对（max diff = 0）。
- 首步 wait **318.73 s vs 基线 324.84 s（只 −6 s）**；py-spy 显示 rank 在 `_get_data`、worker 在
  `adjust_hue_image`——首步由 32 个 spawn worker 的 ColorJitter 争用支配，action 缓存不解决这一段。
- 稳态 step 2–4：32.62 / 31.04 / 31.33 s，VAE 15.09–15.11 s，wait 1.85–3.69 s——与基线同量级，
  说明**稳态 wait 不是 parquet**，是 worker 供给不足（见 B0 的单线程发现）。
- **6 步汇总**：median 32.63 s、7.85 样本/s、VAE 15.11 s（46.3%）、fwd 2.77、bwd 11.89、optim 0.31、
  wait 3.69 s、峰值 40.30/47.33 GiB、MFU 22.5%；8 个 rank 全部 32.62 s、VAE 14.93–15.08 s，
  **完全均衡**。与原基线 32.78 s 同量级——缓存的稳态收益为零。
- **step 5 带缓存仍为 158.62 s（wait 130.28 s）**，原基线 155.97 s（wait 127.25 s）——缓存没有消除停摆。
  py-spy 仍是 rank 在 `_get_data`、worker 在 `adjust_hue_image`：**停摆是单线程增强的吞吐不足，
  不是 parquet/mp4 边界**。action 缓存去掉的是约 0.77 s/样本的 parquet CPU（~7% 供给），
  对停摆没有可见效果。

> **这是本轮最重要的方向修正**：数据侧的瓶颈是 **PyTorch worker 强制单线程 + ColorJitter 太重**，
> 不是磁盘/parquet。`--worker-threads` 的 `wt4` A/B 是核心；action 缓存保留（无损、便宜、去掉 parquet
> 成本），但不要再指望它修停摆。

> 这条修正很重要：**不要把 action 缓存当成首步冷启动的解药**。首步的 worker spawn + 首 batch 增强
> 是结构性的；真正的数据阶段主线是 **worker 线程/数量调优**（OMP 环境变量到不了 worker）与
> uint8/融合增强。

---

## B0 数据侧成本的结构（诊断）

- 每样本 `__getitem__` 约 1.4 s（无增强）：3 视角 × 33 帧解码（~0.5 s）+ 拼画布 + 增强
  （ColorJitter 226–487 ms，其中 hue 的 RGB→HSV→RGB 最贵）+ canvas pad + 归一化 + parquet。
- 内存流量（粗略）：三视角 float32 273 MB → 增强的若干 273 MB 临时量 → composite 137 MB →
  uint8 34 MB → 再 float 137 MB → pad 到 544×736 的 158 MB → normalize 158 MB → 共享内存 collate 158 MB。
  单样本 **~1.5 GB** 的 CPU 内存搬运；256 样本/步（8 卡）≈ 380 GB，按 100 GB/s 量级约 4 s——
  这与稳态 `wait 2.9 s` 同量级。**数据侧不是线程不够，而是 float32 往返太多。**
- 因此数据阶段的优先级是：① action memmap（去掉 parquet 冷读与边界停摆，已实施）；
  ② OMP/worker 调优（先测，看 wait 是否受线程争用）；③ **uint8 直通 + 融合 ColorJitter/归一化**
  （把 1.5 GB/样本的搬运砍到 ~0.4 GB）；④ 解码容器复用（seek 0.04 s/相机，占比小，最后做）。

### 关键修正：worker 是单线程的

PyTorch 的 `_worker_loop` 在调用 `worker_init_fn` **之前**强制 `torch.set_num_threads(1)`
（`torch/utils/data/_utils/worker.py:274`），所以 `OMP_NUM_THREADS=8` 对 DataLoader worker **无效**。
实测每个 worker 的 `%CPU ≈ 114%`（约 1 核），32 个 worker 的样本供给约 **3–4 样本/s**，而训练需要
8 rank × 0.97 ≈ **7.8 样本/s**——数据侧结构性地供给不足，这就是首步 wait 5+ 分钟、稳态 wait 2.9 s 的根因。

修复有两条，已实现第二条：

1. **加 worker 数**：4→8/16 per rank（`--num-workers`）；供给线性增长，代价是 shm 与 CPU 线程数。
2. **在 `worker_init_fn` 里恢复线程**：给 `_seed_data_loader_worker` 加 `threads` 参数，在种子设置后
   `torch.set_num_threads(threads)`；`train_sft.py` 新增 `--worker-threads`。4 worker × 4 线程 = 16
   线程/rank、128 线程/节点，预计单样本 CPU 降 2–4×。

A/B：`wt4`（4 worker × 4 线程，关键）、`w8`（8 worker × 1 线程）、`w8o2`（OMP 环境变量，负对照——
预期无变化，因为 worker 覆盖为 1）。

### uint8 直通设计（数据阶段）

- 增强后 composite 立刻量化到 uint8（上游语义），**pad 也在 uint8 上做**，再一次性转 bf16/float32；
  当前代码在 pad 前已经 `comp.float()`，多了一次 137→158 MB 的 float 往返。
- `video` 以 bf16（或 uint8）进共享内存/H2D：bf16 省 2×，uint8 省 4×；归一化 `x/127.5-1` 放到设备侧
  （或与 VAE 的 conv_in 融合，见 ops 草稿 P1）。
- 数值门禁：uint8→bf16 归一化与当前 float32 归一化→bf16 的差应 ≤ bf16 舍入；用同一 batch 的 latent
  relL2 对拍。
- 预期：单样本 CPU 搬运 1.5 GB → ~0.4 GB；稳态 wait 2.9 s → <1 s；首步增强争用也下降。

## B1 数据侧停摆的根因与修复方向（已实施，见 F4）

- `_read_actions` 每个 worker 在文件边界做一次 `pq.read_table` 整列读取；313,318 行/文件的
  action/state 列，32 个 worker 同时命中同一边界时形成抖动。
- 修复：把 action/state 列预处理成按文件切的 `.npy`/memmap（每文件约 20 MB），`_read_actions` 变成
  按 `row_from + start` 的 O(1) 切片；页缓存跨 worker 共享，消除 parquet 解码与重复读取。
- 验收：见 F4 的训练对拍（loss 逐位一致 + 首步/step 5 的 wait）。

---

## 代码改动清单

| 文件 | 改动 | 目的 |
| --- | --- | --- |
| `examples/worldmodel/cosmos3/train_sft.py` | `--overlap-allgather`、`--action-cache`、`--vae-content-pad`、`--worker-threads` 四个开关，分别透传给 plugin/dataset | 调度与数据路径的 A/B |
| `colossalai/booster/plugin/hybrid_parallel_plugin.py` | `_seed_data_loader_worker(..., threads=0)` 在种子设置后 `torch.set_num_threads(threads)`；`prepare_dataloader(..., worker_threads=0)` | 恢复 DataLoader worker 的并行度（PyTorch 默认强制 1 线程） |
| `examples/worldmodel/cosmos3/droid_dataset.py` | `action_cache`（memmap 切片 + parquet 回退）、`video_content_pad`（内容宽+pad）、numpy 导入 | 去掉 parquet 冷读；裁掉无效画布 |
| `scripts/bench_online_step.sh` | 单机 8 卡标准入口，全部本地盘，支持 `OMP_THREADS` | 统一测试路径 |
| `scripts/summarize_online_metrics.py` | 从 rank JSONL 汇总 step/phase/吞吐/显存/MFU/per-rank | 统一指标 |
| `scripts/ab_online_scheduling.sh` | 调度 A/B case 列表 | 量化 all-gather/worker/OMP/grad-accum |
| `scripts/ab_targeted.sh` | 定向 wt4 / canvas16；每 case fresh log + 独立端口 + 结束 pkill | 避开旧日志与 rendezvous 端口竞态 |
| `scripts/run_integrated.sh` | 32 样本、64 episode、6 步的集成运行（cache + threads + pad） | 生产尺度的最终验证 |
| `scripts/bench_vae_sweep.py` | VAE dtype×chunk×encode-batch 扫描 | 确认配置空间 |
| `scripts/bench_vae_canvas.py` | 画布右侧 pad 扫描 + latent relL2 | 画布裁剪门禁 |
| `scripts/build_action_cache.py` | parquet action/state → 每文件 `.npy` | 数据停摆修复 |
| `scripts/bench_norm_variants.py` | `npu_rms_norm`/`npu_add_rms_norm` 微基准 | 备用；按 token 量估算 <1%，暂不推进 |
| `tests/test_action_cache.py` | 合成 parquet → npy 的逐位/行序测试 | action 缓存的 CI 门禁（已通过） |

## 阶段小结（截至 2026-11-14 13:45 UTC）

| 功能点 | 状态 | 实测效果 |
| --- | --- | --- |
| F0 在线基线（node6 8 卡） | 完成 | 32.78 s/步，7.81 样本/s，VAE 46.3%，MFU 22.4% / 31.5%（含 VAE） |
| F1 VAE 配置扫描 | 完成 | 配置空间平；最快 fp16 chunk12 eb4 0.449 s/样本，仅 −1.3%，显存 +3.7 GiB |
| F2 调度 A/B | 进行中（快速尺度 8 样本/rank） | base 8.01 s（VAE 3.88、wait 0.66、MFU 23.0%）；wt4 首步 wait −28%、稳态 +8%（CPU 争用）；canvas16 进行中 |
| F3 画布裁剪门禁 | 扫描完成 | pad 16 0.377 s/样本（−18.2%），relL2 1.48e-2；pad 64 0.431（−6.5%），relL2 **2.48e-3** |
| F4 action memmap 缓存 | 完成 | 22.41M 行 / 97 s；274 窗口逐位一致；step 1 loss 逐位一致；首步 wait 318.7 vs 324.8 s；**step 5 停摆未消失** |
| 数据停摆 | 根因修正为单线程增强 | 带缓存仍 158.62 s（wait 130.28 s）；worker 在 `adjust_hue_image` |
| worker 线程覆盖 | 机制已验证（单测 0→1、4→4），A/B 进行中 | `--worker-threads` / 更多 worker |

---

## F5 集成配置（完成）

最终配置：`--action-cache /mnt/disk0/jiangzs/nzf/action_cache --worker-threads 4 --vae-content-pad 16`
（保守部署把 16 换成 64）。64 episode、6 步，与 `cache64`（同数据、同 seed、无 worker-threads/画布裁剪）
对拍。

### 实测（steps 2–6 为稳态）

| 指标 | cache64（对照） | integrated | 差 |
| --- | --- | --- | --- |
| step 中位数 | 32.63 s | **29.83 s** | **−8.6%** |
| 吞吐 | 7.85 样本/s | **8.58 样本/s** | +9.3% |
| VAE | 15.09 s（46.3%） | **12.49 s（41.9%）** | **−17.2%** |
| fwd / bwd / optim | 2.77 / 11.89 / 0.31 | 2.81 / 11.05 / 0.45 | — |
| data wait | 3.69 s | 2.66 s | −1.0 |
| 峰值 allocated / device | 40.30 / 47.33 GiB | 40.30 / 47.70 GiB | 持平 |
| MFU | 22.5% | **24.7%** | +2.2 点 |
| 首步 | 350.28 s（wait 318.7） | **216.89 s（wait 187.7）** | **−38%** |
| step 5（边界） | 158.62 s（wait 130.3） | **44.23 s（wait 17.2）** | **−72%** |

逐 rank：8 个 rank 全部 **29.83 s**、VAE 12.29–12.47 s。逐 step：28.91 / 29.83 / 29.96 / **44.23** / 28.88。
48 卡外推：35.1 → **约 32.2 s（−8.3%）**，81.92M → **约 19.8 天**（相对 21.7 天）。

**净收益**：稳态 −8.6%（VAE −17.2% 是主项），冷启动 −38%，边界停摆 −72%，MFU +2.2 点。worker 线程在
32 样本下的稳态代价（optim 0.31→0.45，wait 收益被部分抵消）是净收益小于 VAE 单项的原因。

## F6 uint8 数据直通与 TileLang 算子尝试（2026-11-14 下午）

目标：uint8 留在 uint8 进 H2D（261 MiB/8 样本 vs float32 1078 MiB），把归一化/画布 pad 从 host CPU 拿掉。

### TileLang 内核（负结果）

`video_pad_norm.py` 实现 uint8→fp32 归一化（TileLang vector kernel）。踩到四条硬限制：

1. **grid 上限 65,535 block**：65,536 个 block 直接 aicore timeout（507014）；65,535 正常。
2. **aicore 禁止 unsigned→float 的 C++ static_cast**；须走 CANN `T.tile.cast`。
3. **static_cast fp32→bf16 不支持**（"not support bf16 type cast"）；`T.tile.cast` 只有 **CAST_RINT /
   CAST_ROUND** 与 torch 逐位一致，**CAST_NONE 输出是垃圾**（2026-11-15 在 `probe_tile_ops.py` 中逐模式
   验证：fp32→bf16 的 CAST_RINT 与 `torch.to(bfloat16)` maxabs 0.0；CAST_NONE maxabs 6.2）。
4. **性能不可用**：uint8→fp16 Cast + `T.Parallel` 归一化的 lowering 只有 **~5 GB/s**（273M 元素
   264 ms）；block 从 4,096 提到 21,600 无改善，瓶颈在 Cast/标量 lowering，不是块调度。

### torch_npu 设备算子路径（采用）

`pad_normalize_torch`：uint8 H2D → aclnn `.to(fp32)` → `/127.5-1` → `.to(bf16)` → cat/flip 反射 pad
（5D bf16 的 `F.pad` 在 torch_npu 上不支持，cat+flip 等价）。

| 路径（8 样本） | 时间 | H2D |
| --- | --- | --- |
| TileLang 内核 | 264.2 ms | 261 MiB |
| **torch 设备算子** | **26.7 ms** | 261 MiB |
| 当前 host float + H2D | 35.7–45.8 ms | 1078 MiB |

数值：kernel/torch 路径的 fp32 与 host 差 **1 ulp**（1.19e-7，NPU 除法舍入），bf16 有少量翻转；
相对 VAE 自身的 bf16 计算噪声（~2e-2）可忽略。**收益 ~9 ms/8 样本/rank（~1% 步时）+ host CPU 减少**，
不是大项；数据侧的主线仍是 worker 线程（已落地）。生产路径用 torch 设备算子，TileLang 内核保留为
可复现的负结果。

## F7 uint8 数据直通 A/B（node1，完成）

同 seeds、同 64 episodes、同 action cache、同 `--worker-threads 4`、同 `--vae-content-pad 16`，唯一变量
`--video-uint8`（4 步，稳态取 3 步）：

| arm | step 中位 | 吞吐 | VAE | wait | MFU | loss 均值 |
| --- | --- | --- | --- | --- | --- | --- |
| float host 归一化 | 30.29 s | 8.45 样本/s | 12.62 s (41.6%) | 3.12 s | 24.3% | 14.339381 |
| **uint8 直通** | **27.63 s** | **9.27** | 13.05 s (47.2%) | **0.50 s** | **26.6%** | 14.302470 |

- **−8.8% 步时**，wait 3.12→0.50 s（−84%），loss 均值差 0.037（8 步，噪声量级）。
- 机理：uint8 把 H2D 从 1078 MiB/8 样本降到 261 MiB，并且 host 的 fp32 归一化+reflect pad 换成设备算子，
  **数据供给变快**（wait 直接掉），而 VAE 设备时间不变（13.05 vs 12.62 的差是数据形状噪声）。
- 这是数据侧收益，不是 VAE 收益；它和 VAE 侧的优化可以叠加（F8.4 验证）。

## F8 融合通道优先 RMSNorm 内核（2026-11-15，完成；latent 逐位一致）

### F8.1 目标：不是带宽，是 pass 数

生产配置（batch 8、33×544×656、fast encoder）的 kernel profile：编码器设备时间 **2.953 s**，busy 100%
（无 >50 ms 空洞），其中 norm 链 **549 ms（18.6%）**：

| kernel | 调用 | 时间 | 占编码器 |
| --- | --- | --- | --- |
| `aclnnLinalgVectorNorm_LpNormV2` | 198 | 78.0 ms | 2.6% |
| `aclnnDiv_BroadcastToAiCore_BroadcastTo` | 198 | 66.8 ms | 2.3% |
| `aclnnDiv_RealDivAiCore_RealDiv` | 198 | 245.6 ms | 8.3% |
| `aclnnMul_MulAiCore_Mul` | 198 | 83.2 ms | 2.8% |
| `aclnnMuls_Muls_Muls` | 199 | 72.4 ms | 2.5% |
| `aclnnClipMin` + Cast 进出 | 396 | ~226 ms | 7.7% |

diffusers 0.40 的 `WanRMS_norm` 在 bf16 下是：`F.normalize(x.float(), dim=1).to(bf16) * scale * gamma
+ bias`，即 **cast 进 fp32 → LpNorm → BroadcastTo（materialize 全尺寸 fp32 范数）→ RealDiv → cast 回 bf16
→ 两次 bf16 全张量乘 → 加 bias**。按元素算约 **14 个 bf16-张量 pass**；链本身已经跑在 ~1.17 TB/s 的等效
DRAM 带宽上——**没有带宽浪费，只有 pass 浪费**。

### F8.2 实现（`vae_norm_kernel.py`，TileLang + ascend_tile 向量算子）

- 布局 `[B, C, S]`（S=T·H·W），每个 block 持有一个 `[C, chunk]` UB tile，**读一次、写一次**：
  2-D strided `T.copy` 一次 DMA 取整个 tile（C 行、行宽 chunk）；`acc += x²` 逐行 fp32 归约；sqrt；
  逐行除→舍入 bf16→乘 scale→舍入→乘 gamma→舍入；一次 2-D 写回。
- 三种 shape 的 chunk：C=160→512、C=320→256、C=640→128（UB 192 KB，tile ≤160 KB）。
- **舍入序列与参考逐位一致**（`probe_norm_kernel.py` 在所有 11 个生产 shape 上 relL2 = 0.0、maxabs = 0.0）。
- 不整除的 S 用 **overlap-tail**：最后一个 block 回退到 `S-chunk` 起，与前一 block 重叠；因为输出写的是
  独立 buffer、两个 block 从同一输入重算，重叠列的写入值完全相同。

### F8.3 本后端上写 kernel 必须知道的四件事（都踩过）

1. **`T.Parallel` 在本后端会被标量化**：生成代码是每元素 `SetValue` + `PipeBarrier<PIPE_ALL>`，
   实测 ~5 GB/s（这正是 F6 里 uint8 kernel 慢的真正原因）。所有逐元素运算必须用 `T.tile.*`
   （ascend_tile：`cast/mul/add/div/max/fill/sqrt`），它们才会落成 `AscendC::Mul/Sqrt/...` 向量指令。
2. **fp32→bf16 只能用 `CAST_RINT`/`CAST_ROUND`**（与 torch 逐位一致）；`CAST_NONE` 输出垃圾。
   bf16→fp32 用 `CAST_NONE`（精确加宽）。
3. **`T.tile.reciprocal` 是近似指令**：用它做 `1/‖x‖` 再乘，relL2 2.7e-3；换成真正的 `T.tile.div`
   立刻逐位一致。sqrt+div 的代价可忽略（每元素一次）。
4. **`binary_op` 的标量分支有陷阱**：`src1` 若是裸 `BufferLoad`，会被解释成"以该 offset 起始的向量"；
   `T.cast(g_ub[c], "float32")` 在 fp32 上会被简化掉、退化成裸 BufferLoad（读出越界 → 垃圾）。
   解决：`T.tile.fill(g_row, g_ub[c])` 先铺成行向量，再做向量乘。
5. **每次 launch 的固定开销 ~0.37 ms**（实测：同一个 kernel 按样本循环 8 次，小 shape 也要 3.0 ms）。
   所以 kernel 必须**一次 launch 覆盖整个 batch**，不能按样本循环；这也是 B=1 方案被否掉的原因。
6. **tilelang 的 kernel 缓存只有进程内锁**：8 个 rank 同时编译同一个 kernel 会 dlopen 到写了一半的
   `kernel_lib.so`——实测 **4 个 rank SIGSEGV**。而且训练里 `pack_batch` 的 micro-batch 尾批是 4 而不是 8，
   batch 静态的 kernel 会在**训练中途**触发新编译（正是崩溃现场）。修复：`_compile_locked` 用
   `fcntl.flock` 按 (B,C,S,chunk,scale) 跨进程串行化编译。代价是**每个新 (B,C,S) 首次 ~12–16 s 编译**，
   之后走磁盘缓存；首次训练启动一次性多花 ~2–3 分钟。**锁文件放在 `~/.tilelang` 而不是 `/tmp`**：
   多机训练共享 NFS home 上的同一个缓存目录，锁必须和缓存同处一地。

### F8.4 结果

逐 shape（batch 8，单卡，`probe_norm_kernel.py`）：

| shape | 参考链 | kernel | 加速 |
| --- | --- | --- | --- |
| (8,160,4,272,328) ×32 | 16.26 ms | **5.13 ms** | 3.2× |
| (8,320,4,136,164) ×24 | 8.15 | **4.25** | 1.9× |
| (8,160,4,136,164) ×8 | 4.13 | **1.35** | 3.1× |
| (8,320,2,68,82) ×8 | 0.88 | **0.67** | 1.3× |
| (8,640,2,68,82) ×24 | 2.12 | 2.17 | 0.98×（大 C 小 S，见下） |
| (8,640,1,34,41) ×90 | 0.44 | 0.48 | 0.92×（launch-bound） |
| 合计/次编码 | 873 ms | **391 ms** | **−55%** |

端到端（真实 VAE，batch 8，33×544×656）：编码 wall **2953 → 2479 ms（−16.1%）**，
**latent relL2 = 0.0、maxabs = 0.0（逐位一致）**——对训练零影响。

整步三臂 A/B（node6，8 卡，32 样本/rank，64 episodes，`--vae-content-pad 16 --worker-threads 4`，5 步取稳态）：

| arm | step 中位 | 吞吐 | VAE | wait | MFU | peak alloc / device |
| --- | --- | --- | --- | --- | --- | --- |
| norm_off | 29.89 s | 8.56 样本/s | 12.40 s (41.5%) | — | 24.8% | 40.30 / 47.70 GiB |
| **norm_on** | **27.86 s** | **9.19** | **10.67 s (38.3%)** | 2.73 s | **26.4%** | 45.80 / 54.27 GiB |
| norm_on + uint8 | 27.99 s | 9.15 | 12.63 s (45.1%) | 0.47 s | 26.3% | 45.79 / 54.13 GiB |

- **−6.8% 步时、−14.0% VAE**，与单卡 16% 的编码收益一致（步内 VAE 只占一部分）。
- loss 轨迹逐 step 对齐（如 step 3 的 v-loss 2.1888 / 2.1879 / 2.1879），确认无训练影响。
- **代价：显存 +6.6 GiB**（47.7 → 54.3 GiB device）。融合 norm 每次调用要一块全尺寸输出（最大 914 MB），
  10 种 shape × 微批 8/4 的尺寸不同，caching allocator 会为每种尺寸各留一块。64 GB 卡仍有 ~7 GiB 余量，
  但这是这个优化的真实成本，6 机部署时要算进去。
- **uint8 在 node6 上与 norm kernel 叠加接近中性**：收尾 A/B（两臂都开 norm kernel）
  `final_float` 27.17 s（VAE 10.50 / wait 2.24）vs `final_uint8k` **26.97 s**（VAE 11.91 / wait 0.46），
  **净 −0.2 s（−0.7%）**。node1 上（无 norm kernel）uint8 是 −8.8%，因为那台机器数据侧更紧。
  **结论：uint8 是"数据供给紧"时的解药，不是普遍提速**；它的设备侧成本经 F11 定位主要在 reflect pad。
- norm_on 后 **wait 2.73 s 重新成为第二大项** → 下一步数据侧仍值得做（更多 worker / 更快增强），
  与 VAE 侧优化不冲突。

融合后的 stage profile（batch 8，单卡，`profile_vae_deep.py --norm-kernel`，编码 2.482 s）：

| stage | 融合前 | 融合后 | 变化 |
| --- | --- | --- | --- |
| down_blocks.0 | 1.514 s (51.1%) | **1.148 s (46.0%)** | −24% |
| down_blocks.1 | 0.918 (31.0%) | **0.800 (32.1%)** | −13% |
| down_blocks.2 | 0.331 (11.2%) | 0.331 (13.3%) | 0 |
| mid_block | 0.087 | 0.094 | 噪声 |
| down_blocks.3 | 0.070 | 0.076 | 噪声 |
| conv_in | 0.034 | 0.034 | 0 |
| **合计** | 2.953 | **2.482** | −16% |

单卡 profile 峰值显存 13.41 → **17.74 GiB**（+4.3 GiB，8 卡训练时表现为 +6.6 GiB）。

### F8.5 已知边界

- **大 C 小 S** 的两个 shape（C=640、S=1394/11152）比参考链慢 3–8%：tile 只有 128 列宽，每 block 640 行
  逐行 8 个向量指令，且 88 个 block 在 48 个 vector core 上只有 1.8 波（负载不均）。量级 ~9 ms/编码
  （≈0.04 s/步），暂不处理；若要把它们也吃掉，方向是按 C 切块做两遍（2 读 1 写）换更大的列宽。
- kernel 依赖 **contiguous NCDHW**；`channels_last_3d` 或非连续输入回退到参考实现（安全兜底）。
- **首次启动要编译 9 个 shape 的 kernel**（每个 ~10–15 s，8 个 rank 共享磁盘缓存，只在第一次训练启动时付一次）；
  之后走 tilelang 磁盘缓存。这是启动成本，不在稳态步时里。

## F9 下一步方向判断：encoder 已经是 conv-bound（2026-11-15，profile 后）

融合 norm 之后 encoder 剩 2.48 s，profile 里剩下的结构（融合前 2.95 s 的口径）是：

| 类别 | 时间 | 占比 | 结论 |
| --- | --- | --- | --- |
| Conv3D 本体 + conv 内部 TransData | 1.246 + 0.275 s | **61%** | 见下 |
| 残差/Silu 等 elementwise | ~0.32 s | 13% | 单点 <3%，融合收益小 |
| feat_cache 的 Cat/Slice | ~0.16 s | 6.5% | 可用常驻缓存缓冲省 ~2/3 的搬运（≈3%） |
| Transpose | 0.109 s | 4.4% | mid-block attention 相关 |
| PadV3 | 0.058 s | 2.3% | 下采样前的 reflect pad |

**Conv3D 不手写——先量了再说**：`probe_conv_shapes.py` 记录全部 conv 的形态并算 FLOPs：
一次编码（batch 8）conv 共 **326 TFLOP**（含 batch），conv 本体 1.246 s ⇒ **≈262 TFLOPS ≈ 74% MFU**
（计入 conv 内部 TransData 0.275 s 后 ≈60%）。**conv 已经接近峰值，自研 kernel 没有空间。**
与 npu-guide §4.2.2b 的容量预算一致：

1. Conv3D 与等价 GEMM 的算术强度差 ≈25×（FLOPs×27、GM 字节几乎不变），它慢是**算法与片上容量**，
   不是带宽；`baseM×baseN=128×128` 时 MTE1 还有 2× 余量，L1 只够 2 级流水才是真约束。
2. 官方 ops-nn 的 `conv3d_v2` 已经是「模式 A（按输出帧切块）+ 靠 L2」的实现，guide 的结论就是
   **接受这个模式**；手写要同时处理 im2col/深度滚动/C_in 切分，收益上限只是那 0.275 s 的 TransData。
3. 真要动 conv，唯一低风险杠杆是**布局**：输入已是 NCDHW，CANN 每次 conv 前要做
   NCDHW→NC1HWC0（801 次 TransData，0.275 s）。用 `channels_last_3d` 喂给 conv 是否能省掉它，
   需要一次 profile 验证（`profile_vae_deep.py --channels-last`），且要连带确认 Cat/Pad/Silu 在
   NDHWC 下不变慢、以及把融合 norm 改成「按最后一维归约」的 NDHWC 变体（这个变体本身更容易写）。

**融合后的真实构成**（`profile_vae_deep.py --norm-kernel`，3165 个 kernel，2.482 s，busy 100%）：

| kernel | 调用 | 时间 | 占比 |
| --- | --- | --- | --- |
| Conv3DV2 | 233 | 1.246 s | 50.2% |
| **融合 norm（main_kernel）** | 198 | **0.372 s** | 15.0% |
| conv 内部 TransData | 801 | 0.275 | 11.1% |
| Cat（feat_cache） | 272 | 0.133 | 5.4% |
| Transpose | 61 | 0.109 | 4.4% |
| Silu | 189 | 0.077 | 3.1% |
| Add（残差） | 135 | 0.074 | 3.0% |
| PadV3 | 73 | 0.059 | 2.4% |
| Conv2D | 45 | 0.045 | 1.8% |

**`channels_last_3d` 这条路关闭（负结果）**：torch_npu 的 `Module.to(memory_format=torch.channels_last_3d)`
直接报 `RuntimeError: Only contiguous_format or preserve_format is supported`，NPU 后端不支持 NDHWC
张量，所以 conv 内部的 0.275 s TransData 无法用布局消掉。

**判断**：encoder 侧剩下能动的只有 feat_cache 的 Cat/Slice（0.167 s，6.7%，滚动缓冲可省 ~2/3）和
Transpose 0.109 s（需先定位来源），合计上限 ~0.25 s/编码（≈1 s/步，3%）。conv 本体已 74% MFU，不动。
**训练循环侧（bwd 11.2 s vs fwd 2.8 s = 步时的 40%）余量更大**，下一步用 `--profile-step` 抓稳态步归属。

## F10 训练循环：一步里通信与计算的归属（2026-11-15，进行中）

`train_sft.py --profile-step 2`（新增开关）对稳态步做 torch_npu profile，`analyze_step_kernels.py` 汇总：

| 类别 | 设备时间 | 占 busy | 说明 |
| --- | --- | --- | --- |
| communication（hcom） | **10.51 s** | 32.3% | 501 次调用；56 次 reduceScatter 各 ~0.17 s |
| elementwise | 8.52 s | 26.1% | Add 1.99 / Mul 1.67 / Cast 0.62 / Pow 0.43 |
| conv（VAE） | 6.28 s | 19.2% | 4 次编码 |
| matmul | 3.53 s | 10.8% | |
| FlashAttention fwd+bwd | 2.43 s | 7.5% | |
| 融合 norm（main_kernel） | 1.49 s | 4.6% | 792 次 |

**关键观察：通信与计算零重叠**。区间并集：wall 23.19 s ≈ comm 10.51 + matmul/FA 5.96 + conv 6.28，
三者几乎完全串行。但 profile 会把一步从 27.9 s 放大到 61 s，所以这 10.5 s 是**上界**——
按未 profile 的 bwd 11.19 s − 重算 fwd 2.8 s − 梯度 matmul ~5.5 s 推算，真实通信 ≈ 2–3 s/步。

**链路本身不是瓶颈**（`bench_hccl.py`，8 rank，node6，HCCL_SOCKET_IFNAME=bond1）：

| 集合通信 | 64 MiB | 256 MiB | 512 MiB |
| --- | --- | --- | --- |
| all_reduce | 55 GiB/s | 61 | 64 |
| reduce_scatter | 107 | 110 | **119 GiB/s（512 MiB 仅 4.19 ms）** |

所以训练里每次 reduceScatter 0.17 s 不是带宽问题。**bucket 大小 A/B 已排除"同步点太多"**：

| arm | step 中位 | bwd | MFU | 显存 |
| --- | --- | --- | --- | --- |
| bucket 12（默认，≈119 桶） | 27.35 s | **10.41 s** | 26.9% | 57.03 GiB |
| bucket 48（≈30 桶） | 27.08 s | **10.41 s** | 27.2% | 57.09 GiB |

bwd 一模一样 → 成本不在同步点数量。真正的原因是 `low_level_optim.py:409` 的
`dist.reduce_scatter_tensor` 是**同步调用**：虽然放在 comm stream 上，但主机线程阻塞到集合通信完成，
所以与计算零重叠（对比：参数 all-gather 走 `ZeroOpHook` 的异步 handle，是重叠的）。

**异步 reduce-scatter 试过，无效（负结果）**：把 `dist.reduce_scatter_tensor` 改成
`async_op=True` + post-processing 仍排在同一 comm stream + 消费点 `wait_stream` 设备侧排序、
bucket 复用前再排序一次（全程无主机阻塞）：

| arm | step | bwd | MFU |
| --- | --- | --- | --- |
| 同步（默认） | 27.36 s | 10.67 s | 26.9% |
| **异步**（`COSMOS3_ASYNC_GRAD_REDUCE=1`） | 27.46 s | **10.69 s** | 26.8% |

bwd 不变 → **串行化在设备侧，不是主机阻塞**（HCCL 的集合通信任务与计算在 AI core 上不能并行）。
补丁已回退，结论保留：想省这 ~2 s 只能减通信量（fp8 通信，数值风险）或换并行策略，不在调度层。

## F11 uint8 归一化 kernel 重写（2026-11-15，完成；逐位一致，4.6×）

F6 里那个 5 GB/s 的 TileLang uint8 内核是**被 `T.Parallel` 的标量化坑了**（见 F8.3 第 1 条）。
用 `T.tile.*` 向量算子重写后（`video_pad_norm.py`）：一次 launch、147 KB UB、每元素
uint8→fp16→fp32 除法(127.5)→减 1→`CAST_RINT` 转 bf16。

> **测量口径修正**：第一次的 "5.92 vs 27.23 ms" 作废——`BLOCK=16384` 不整除
> 8×3×33×540×640 = 2^12·3^4·5^2·11，`normalize_uint8` 静默回退到 torch 路径，那个 5.92 ms 既不是内核、
> 也不含 pad。已改 `BLOCK=4096`（最大可整除的 2 的幂），8 样本调用拆成 2 次 launch（66,824 blocks 超过
> 65,535 的 grid 上限），下面的数字是重测结果。

| 路径（8 样本 540×640×33） | 时间 | H2D |
| --- | --- | --- |
| **重写后的 TileLang kernel（normalize only）** | **3.99 ms** | 261 MiB |
| torch 设备算子（normalize only，回退口径测得） | ~6 ms | 261 MiB |
| **reflect pad（`flip`+`cat` on 5-D bf16）** | **≈22 ms** | — |
| 内核 + pad（新路径整体） | 25.87 ms | 261 MiB |
| torch 设备算子 + pad（原采用） | 27.65 ms | 261 MiB |
| host float + H2D（基线） | 36.82 ms | 1078 MiB |

结论：
1. **内核本身逐位一致（`bf16_equal=True`）且比 torch 快 ~1.5×**，F6 的 5 GB/s 负结果被推翻——那是
   `T.Parallel` 标量化造成的，不是 uint8/TileLang 的问题。
2. **uint8 路径现在的瓶颈是 reflect pad 而不是归一化**：`flip`+`cat` 在 5-D bf16 上要做两次全张量
   cat（~1.7 GB 流量）却跑了 22 ms（比带宽下限慢 ~12×，torch_npu 的 flip 在 5-D bf16 上低效）。
   若要继续压：把 pad 折进内核的 store（按行写进带 stride 的 padded 缓冲），或先 pad uint8（1 B/element）
   再归一化；再往下就是 2–4 ms 量级。
3. 整步口径（同名同参三组运行）：
   - `u8_torch`（torch 归一化）：27.55 s，VAE 12.47，wait 0.45，MFU 26.7%
   - **`u8_kernel`（融合内核）：26.95 s，VAE 11.89，wait 0.42，MFU 27.3%**
   - 对照 `final_float`（float 数据路径）：27.17 s，VAE 10.50，wait 2.24，MFU 27.1%
   即 **uint8+内核对 float 净 −0.2~0.5 s（−1%）**，但 uint8 三次运行的 VAE 在 11.89–12.47 之间波动
   （±0.5 s），**内核的步级效应低于运行间噪声**；它的价值在算子级（3.99 ms、逐位一致）和把设备侧成本
   压到可忽略。node1 那种数据紧的机器才是 uint8 的主场（−8.8%）。
接入方式：`COSMOS3_VIDEO_NORM_KERNEL=1`（packing.py 的 uint8 分支自动走它）。
**注意量级**：这只省 ~21 ms/编码调用 ≈ **85 ms/步（0.3%）**，低于运行间噪声——它的意义是把 F6 的负结果
翻正、并把 uint8 路径的设备开销压到可忽略，而不是步时大项。

## F12 小项清扫（2026-11-15，均为可测/不可测的边界情况）

| 项目 | 改动 | 结果 |
| --- | --- | --- |
| loss 的 `pow(2)` | `compute_loss` 里 `.pow(2)` → `.square()`（每步 770 个 Pow kernel） | **无步级效果**（27.20 vs 27.17 s，fwd 2.82/2.83、bwd 10.42/10.47 均相同）；保留改动因为 square 数值上更精确且单指令，但 profiling 里 Pow 的 0.43 s 主要是放大 |
| reflect pad | 识别为 uint8 路径的真正瓶颈（22 ms/调用） | 未实现；上限 ~50 ms/步（0.2%） |
| encoder 的 61 次 `Transpose`（0.109 s） | `profile_vae_deep.py --with-stack` 尝试定位 | **两次尝试均未产出**（带栈 profile 在本栈上进程静默退出）；上限 0.44 s/步（1.6%） |

## F13 多机验证与两趟 RMSNorm（2026-11-15）

### F13.1 两机（node1 + node6，16 rank）验证融合 norm

`bench_online_step_2node.sh`（node1=rank0/master 192.168.0.46，node6=rank1；容器 host 网络、bond1 互通，
ICMP 被过滤但 TCP 可达；node 内 shard 8、跨机 replica 2），3 步 × 2 臂：

| arm | step | VAE | fwd | bwd | 显存 |
| --- | --- | --- | --- | --- | --- |
| norm_off（16 rank） | 31.52 / 31.54 s | 15.05 s | 2.78 | 11.33 | 47.7 GiB |
| **norm_on（16 rank）** | **26.78 / 29.14 s** | **12.91 s** | 2.78 | 10.34 | **58.3–60.0 GiB** |
| **最终树（16 rank，两趟 + AvgDown pad + 8 GiB 守卫）** | **26.64 / 28.46 s** | **12.30 / 12.29 s** | 2.77 | 10.46 | **58.2–58.3 GiB（不再爬升）** |

最终树在两机上是 26.6 / 28.5 s、VAE 12.3 s（比 one-pass 再 −0.6 s），显存平在 58.2 GiB
（守卫前是 58.3 → 60.0 的爬升）。首步 423.8 s 里 323.8 s 是数据冷启动、88.3 s 是两台各自的首次编译+预热，
与稳态无关。

**单机的收益在多机下成立**（−15%，VAE −2.1 s）。但显存从 47.7 涨到 **58–60 GiB，逼近 61 GiB 上限**——
融合 norm 每次调用的全尺寸输出块在 16 rank 下的峰值更高，**六机部署前必须确认这条**（可选项：降低
`--vae-encode-batch`、或在 norm 前后限制并发编码数）。

### F13.2 两趟 RMSNorm（默认启用，逐位一致）

one-pass 的 tile 受 UB 限制只能用 1 KB 行宽（跨 713 KB 步长），实测只有 358 GB/s；两趟版（先归约出
每列范数到 fp32 scratch，再应用）用 16 KB 行宽、读两遍写一遍：

| shape | one-pass | two-pass | 提升 | 逐位一致 |
| --- | --- | --- | --- | --- |
| (8,160,356864) | 5.110 ms (358 GB/s) | **3.623 ms (756 GB/s)** | **−29%** | ✅ relL2=0 |
| (8,320,89216) | 4.225 (216) | **1.856 (738)** | **−56%** | ✅ |
| (8,640,11152) | 2.121 (108) | 1.646 (208) | −22% | ✅ |
| (8,160,22304) | 0.392 (291) | 0.649 (264) | 变慢（S 小于阈值，生产走 one-pass） | ✅ |

阈值 `TWO_PASS_MIN_S = 65536`（低于它的 shape launch 开销占优）。算术顺序与 one-pass 完全相同，所以
**latent 仍逐位一致**。

两趟内核首次在训练里调用时同样会八 rank 并发编译，`_two_pass_kernels` 原本没走编译锁，实测直接
**SIGSEGV**（twopass 首跑 rank 5/7 同时挂）。现在两条路径都过 `_locked_compile`（按 `reduce:`/`apply:`
分锁），这是 8 rank 共用 tilelang 磁盘缓存的硬性要求。

### F13.3 单机两趟 vs 单趟（同参数 A/B，5 步）

| 臂 | 稳态 step（2–4） | VAE | bwd | fwd | 峰值 reserved |
| --- | --- | --- | --- | --- | --- |
| twopass（默认） | 24.90 / 26.48 / 27.49 s | **10.32 / 10.06 / 10.08 s** | 10.28 / 10.27 / 11.04 | 2.80–2.83 | 54.0 → 56.1 GiB |
| twopass_off（`COSMOS3_NORM_TWOPASS=0`） | 25.27 / 27.97 / 27.26 s | 10.51 / 10.79 / 10.54 s | 10.14 / 11.38 / 10.07 | 2.81–2.82 | 53.0 → 53.3 GiB |

**VAE 段 −0.46 s/步（−4.4%）**，与单算子探针推算的 −0.3 s 一致。整步中位数 26.48 vs 27.26 s（−0.8 s），但逐步
`wait` 噪声有 ±1.5 s，只能确认方向、不能引用单步。**代价是峰值 reserved +2.5 GiB**：两趟路径用
`torch.empty_like` 申请输出、`torch.empty` 申请 fp32 scratch，走 torch 缓存分配器；one-pass 的输出由
tilelang 的 `out_idx` 分配。若六机上这 2.5 GiB 变成瓶颈，把 apply 内核也改成 `out_idx=[-1]` 即可对齐两条路径
的分配方式（需要重编译全部 shape，暂未做）。

## F14 profile 的另一面：VAE 已经 100% 占用，剩下的是四类"搬运"（2026-11-15）

`analyze_profile_deep.py`（新）在 F9 的 top-N 之外回答四个问题：有没有 idle、有多少 launch-bound 小算子、
相邻同类算子有多长连跑（融合候选）、host 侧算子构成。

**结论 1：VAE 编码没有 idle。** 2.482 s 的 span 里设备 busy 99.9%（区间并集），>20 µs 的空隙一个也没有。
所以 VAE 侧不存在"调度损失"，只有"算子本身慢"。

**结论 2：不是 launch-bound。** ≤10 µs 的 kernel 有 289 个（占调用数 9%），但只占 0.0% 的时间
（最大项是 conv 的 bf16→fp16 bias cast，1.21 µs × 278）。每个 kernel 都足够大。

**结论 3：elementwise 没有融合空间。** 相邻同类连跑统计：单发 153 次/75.0 ms，2 连跑 9 次/0.61 ms，
4 连跑 1 次/0.11 ms。**全部连跑只有 0.72 ms**，即使把每个边界融掉也只有 0.36 ms。VAE 的 elementwise
不是链式的（Silu/Add 各自独立），融它没有意义。

**结论 4：时间在四类"搬运"上**（合计 0.58 s/编码，23%）：

| kernel | 调用 | 时间 | 占比 | 性质 |
| --- | --- | --- | --- | --- |
| `aclnnConvolution_TransData` | 801 | 275 ms | 11.1% | conv 的 input/output 转 fractal；`probe_conv_transdata` 实测单次大 conv 是 **2 个**（输入+输出），weight 没有单独转换 |
| `aclnnCat_ConcatD` | 272 | 133 ms | 5.4% | feat_cache 的 `cat([cache, x])` |
| `aclnnInplaceCopy_TransposeAiCore` | 61 | 109 ms | 4.4% | `permute().contiguous()`（AvgDown3D 27、downsampler 24、attention 18） |
| `aclnnConstantPadNd_PadV3AiCore` | 73 | 59 ms | 2.4% | `F.pad`（含 AvgDown3D 的 0 填充 pad） |

四个来源都在 host 侧可枚举（`operator_details.csv` 里 `aclnnInplaceCopy` 322 次/148.7 ms、
`aclnnCat` 272/138.3、`aclnnConstantPadNd` 98/74.3）。它们都是纯数据移动（换实现不改数值），所以问题
不是"能不能换"，而是"换了能不能更快"——三个探针把每一类的底价都量了：

四类的可回收量（编码级，括号内是依据）：

| 来源 | 可回收 | 依据 |
| --- | --- | --- |
| TransData 的 weight 部分 | **0（已排除）** | `probe_conv_transdata`：`npu_format_cast(w, 33)`（FRACTAL_Z_3D）和 29（FRACTAL_NZ）转换本身成功，但随后的 `conv3d` 直接报 NPU op 错误；31（NDC1HWC0）被 Python 侧以 `unknown format type` 拒绝。ND/NDHWC 是 no-op，仍剩 2 个 TransData/次。**这条也不通** |
| AvgDown3D 的零宽 pad | ≈15–30 ms | `pad_t=0` 时 `F.pad` 仍整张拷贝（`PadV3`/`TensorMove`）；跳过不改任何值，已实施 |
| Transpose（AvgDown3D 27 次） | ≈0 | 实测带宽 bound：profile 里 1791 µs/次，按最宽 tensor ≈0.9 GB 算约 1.0 TB/s，自研 permute-copy 没有余量 |
| Cat（feat_cache） | ≈0 | `probe_layout_ops`：`cat` 实测 **1261 GB/s**（`copy_` 1201 GB/s），已到峰值；环形缓冲能省 1/3 但要跨 chunk 常驻 `[B,C,6,H,W]`×31 层，显存不可行 |

`probe_layout_ops` 的另外两个数：`F.pad(t+2)` 只有 **345 GB/s**（`zeros+copy` 307 GB/s，两者都慢，是 pad 类的通病），
而**零宽 pad 仍要 0.293 ms**（整张拷贝）——这就是已实施的那条。

## F14.1 显存：把回收阈值从"快满了"提到"留 8 GiB"

VAE 段每次融合 norm 调用都会申请一张全尺寸输出，micro-batch 尺寸一变就多留一个缓存块：heartbeat 里
**allocated 19 GiB / reserved 56 GiB**（61 GiB 卡），逐步爬升 54.1 → 55.4 → 57.1 → 57.5 GiB。原来的守卫是
`free < 3 GiB` 才 `empty_cache()`，等于留到只剩 3 GiB 才救。现在：

```python
keep_free = float(os.environ.get("COSMOS3_EMPTY_CACHE_GIB", "8")) * 2 ** 30
if free_bytes < keep_free:
    torch.npu.empty_cache()
```

一次 `empty_cache` 在两步之间只值几毫秒，但它把 37 GiB 的缓存空洞还给卡（`allocated 19 / reserved 56` 说明
空洞确实存在）。**6 步单机验证（`memcheck`，同参数、只差这个阈值）**：

| step | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- |
| 旧阈值（3 GiB）`mem` | 55.4 | 57.1 | 57.5 | — | — |
| **新阈值（8 GiB）`mem`** | **54.9** | **55.0** | **55.3** | **55.1** | **55.2** |

新阈值下 device_used **平在 55.1±0.2 GiB**（旧阈值 3 步涨 2.1 GiB 且还在涨），`peak_reserved` 53.5 → 56.0。
步时没有代价（25.0 / 28.2 / 27.5 / 27.5 s，step 5 的 50 s 是数据停摆）。**长跑显存风险解除。**

## F15 训练循环侧：DiT 的 norm 根本不是融合算子（2026-11-15，分析）

看 host 侧算子前先读模型代码：`Cosmos3OmniTransformer` 的 `hidden_act="relu2"` 把 norm 选成
`Cosmos3NemotronRMSNorm`（`transformer_cosmos3.py:164`），它的 forward 是**分解的 fp32 链**：

```python
h = hidden_states.float()                      # cast
variance = h.pow(2).mean(-1, keepdim=True)     # pow + reduce
h = h * torch.rsqrt(variance + self.eps)       # rsqrt + mul
return (self.weight.float() * h).to(dtype)     # mul + cast
```

每层 4 个（`input_layernorm` / `input_layernorm_moe_gen` / `post_attention_layernorm` /
`post_attention_layernorm_moe_gen`），28 层 → **112 次/前向**，每次搬运约 40 B/元素。按 113k token × 2048
算，单次 ≈9.3 GB，单次 ≈7 ms，112 次 ≈ **0.8 s/前向**，反向同量级。对比之下同一个文件里注意力用的
`RMSNorm`（diffusers）走的是 `torch_npu.npu_rms_norm`，已经是融合的——**只有 nemotron 这条路径漏了**。

这也是 profile 里 `aclnnPow` / `aclnnMean` / `aclnnRsqrt` / 大量 `Mul`+`Cast` 的来源。是否值得替换取决于
两件事，都由 `probe_dit_norm.py` 直接回答：

1. `torch_npu.npu_rms_norm` 与这条 fp32 链**是否逐位一致**（差别只可能在行归约顺序和 rsqrt 实现）；
2. 若不一致，自研融合 kernel（`dit_norm_kernel.py`）能否复刻；以及**反向**能否与 autograd 的链路逐位一致
   （kernel 没有 autograd，必须配一个 torch 算子实现的反向，这是风险点）。

**答案 1（决定性）：行归约复刻不了。** `probe_nemotron_norm.py` 在 [8192,2048] 上把 24 种候选归约
（1/2/4/8/16/32/64 lane、2–32 块、seq/tree 组合）逐一与 `mean(-1)` 比对：

| 候选 | 不一致行数 | max abs diff |
| --- | --- | --- |
| sequential / tree | 2879/8192 (35%) | 4.9e-4 |
| 64lane+seq/tree | 1705/8192 | 4.9e-4 |
| 32blk+seq+seq | 3580/8192 | 4.9e-4 |
| 其余（2/4/8/16 lane） | 7436–8192/8192 | 5e-3 |

**没有一个逐位一致**，最好也有 35% 的行在最后一位不同（fp32 1 ulp）。所以：**任何把 `cast→pow→mean` 融成
一趟的自研 kernel 都无法与上游逐位一致**，这条 5.3% 的路径在"逐位一致"的约束下关闭。（
`torch.rsqrt == 1/sqrt` 逐位成立，rsqrt 不是问题；问题只在归约顺序。）

顺带：`[time] reference chain 0.316 ms for 8192x2048` 的探针数字不能当性能结论——8192 行的中间张量还在
L2（192 MB）里，真实 5 万行时是 6.3 ms/次（见 F15.1）。

**答案 2：厂商融合算子只差 7e-7，而且反向可以做到逐位一致。** `probe_dit_norm.py`（[8192,2048]，随机数据）：

| 前向 | bf16 不一致元素 | relL2 |
| --- | --- | --- |
| 自研 TileLang kernel | 1452 万 / 1678 万（归约参数不对，原型作废） | 2.0 |
| **`torch_npu.npu_rms_norm`** | **12 / 16777216（7e-7）** | 1.1e-5 |

把 `npu_rms_norm` 包进一个 `Function`、反向用与 autograd 同序的 torch 算子重写，**反向对 x 和 weight 都逐位
一致（0/16777216 与 0/2048）**。所以这条路的代价精确地是"前向 7e-7 的元素偏 1 ulp"，收益是 6.3 ms → ~1 ms/次
（394 次/步 ≈ **0.9–1.3 s/步，3.5–5%**）。

`probe_norm_ulp.py` 在**生产尺寸**（[49152,2048] = 1.0 亿元素）上把"到底差多少"量化了：

| 前向实现 | bf16 不一致元素 | 比例 | relL2 |
| --- | --- | --- | --- |
| sequential 归约 | 436 / 100663296 | 0.0004% | 1.2e-5 |
| 64lane+tree 归约 | 104 / 100663296 | 0.0001% | 6.4e-6 |
| **npu_rms_norm** | **113 / 100663296** | **0.0001%** | 6.9e-6 |

即约**每 90 万个元素 1 个偏 1 ulp**。这是"用 7e-7 的差异换 3.5–5% 步时"的精确报价。

**在真实激活上复核偏差**（`COSMOS3_DIT_NORM_VERIFY=1`，逐次调用与 fp32 链对拍，2 步）：

```
[dit-norm-verify] 788 calls, 95,740 / 82,606,485,504 elements differ (0.00012%), worst per-call relL2 2.51e-05
```

即**每 100 万个元素只有 1.2 个不同**，单次调用最坏 relL2 2.5e-5——与合成数据探针（0.0001%）一致，
说明真实激活的分布没有放大偏差。这是"要不要开它"最直接的那个数。

**训练级实测（`--dit-norm-kernel`，6 步，与基线同参数，单机 8 卡）**：

| | 基线 | `--dit-norm-kernel` |
| --- | --- | --- |
| 步时（5 个稳态步中位数） | 27.01–27.08 s | **25.96 s（−1.05 s，−3.9%）** |
| fwd | 2.79–2.82 | **2.32** |
| 吞吐 | 9.48 样本/s | **9.86（+4.0%）** |
| MFU | 27.2% | **28.3%** |
| device used | 55.5 GiB | **51.34 GiB（−4.2）** |
| peak allocated | 48.2 GiB | 44.25 GiB |

5 个稳态步没有发散，loss 轨迹与基线保持 4 位有效数字一致（step-1 20.215309 vs 20.212885）。
**再跑到 20 步（19 个稳态）**：25.82 s 中位数、9.91 样本/s、fwd 2.31、bwd 10.55、VAE 10.02、53.04 GiB，
loss 从 20.5 平滑降到 3–11 的批间波动区间，**没有 NaN、没有尖峰**——这条也是唯一跑到 20 步的配置。

**A vs B 的 20 步 loss 对照**（同 seed、同 batch，唯一变量是 norm）：

| step | 1 | 2 | 3 | 4 | 10 | 14 | 17 | 20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A（逐位） | 20.2129 | 14.7049 | 12.8321 | 9.4516 | 4.3771 | 4.1479 | 24.8366 | 7.6624 |
| B（fused） | 20.2153 | 14.7110 | 12.8930 | 9.1951 | 4.0097 | 3.2511 | 22.5239 | 6.8940 |
| 差 | **+0.0024** | +0.0061 | +0.0608 | −0.2565 | −0.3674 | −0.8969 | −2.3127 | −0.7684 |

- **step 1 只差 0.012%**（正是 1.2 ppm × 1 ulp 的量级），step 2 差 0.04%；
- 从 step 3–4 起两条曲线开始按混沌方式分开——**任何非逐位改动（哪怕换个随机种子）都会如此**；
- 统计上不可区分：19 个稳态步的均值 A 8.337 / B 7.894、标准差 A 5.13 / B 4.72，**B 没有系统性变差**。


**16 rank（node1+node6）同样成立**：步时 **26.48 s 中位数**（基线 27.37，−0.89 s，−3.3%）、fwd 2.35（基线
2.84）、**device used 51.08 GiB（基线 56.22，−5.1）**、每 rank 32/26.48 = 1.21 样本/s → 16 卡 **19.3 样本/s**
（基线 18.7）。**单机与多机的收益幅度一致，没有多机衰减。**

**建议（这是本文件唯一一处需要拍板的地方）**：`--dit-norm-kernel` 的实测报价是
**−1.05 s/步（−3.9%）、+4.0% 吞吐、MFU +1.1 点、−4.2 GiB**，代价是**前向每 100 万元素 1.2 个偏 1 ulp、
反向与优化器路径逐位一致**。我的建议是**在六机上开启它**（用 `--no-dit-norm-kernel` 可随时回退），
因为 1 ulp / 1.2 ppm 的差异远在任何训练噪声之下，而 4 GiB 显存正好补上 SAC/跨步重叠一直缺的余量。
**但我没有替你翻默认**——你给的约束是"精度要对齐"，逐位 vs 1 ulp 属于策略判断，不是工程判断。

### F15.0 `mean(-1)` → `sum(-1)/H`：逐位一致，把一整趟全张量除法搬出反向（**默认开**）

`h.pow(2).mean(-1)` 的除法发生在 reduce 内部，但它的**反向**是 `grad.expand() / numel` —— 一次覆盖整个
`[tokens, 2048]` fp32 激活的除法。改写成 `h.pow(2).sum(-1) / H` 后，除法落在 `[tokens, 1]` 的结果（前向）
和 `[tokens, 1]` 的梯度（反向）上，expander 之后接的是 autograd 本来就要做的 `* 2h`。
**H = 2048 是 2 的幂，除法精确，所以 `sum / H == mean` 逐位相同。**

| 检查（49152×2048） | 结果 |
| --- | --- |
| 前向 | **bit_equal，0/100,663,296** |
| x 梯度 | **bit_equal，0/100,663,296** |
| weight 梯度 | **bit_equal** |
| fwd+bwd | 12.07 → **11.37 ms（+5.8%）**，按 394 链 ≈ **0.28 s/步** |
| 训练级 | **step-1 loss 20.212885**，与基线**完全一致** |

步级 A/B 里 0.28 s 淹没在 ±1 s 的噪声里（24.99/26.83 vs 24.94/27.43），但它是**严格更少的算子**且逐位一致，
所以进默认（`--dit-sum-mean-norm`，两个 bench 启动脚本已带）。
这也解释了 profile 里 `aclnnDivs_RealDiv` 的 706–822 次调用：约等于每个 norm 反向一次。

**profile 复核（同一台机器、同一配置，`--profile-step 2`）——机制确认：**

| | RealDiv 调用 | RealDiv 时间 | 一步 kernel 总时间 |
| --- | --- | --- | --- |
| `mean(-1)`（基线） | 706 | **145.3 ms** | 33,055 ms |
| **`sum(-1)/H`** | 1100 | **8.6 ms（−94%）** | **32,280 ms（−775 ms，−2.3%）** |

调用数反而变多（`sum/H` 给每个 norm 前向加了一次 `[tokens,1]` 的除法），但**时间塌了 17 倍**——因为剩下
全是 `[tokens,1]` 的小除法，全张量的那批没了。

**但步级没有可分辨的收益，原因值得记住**：单机单跑一对是 24.48 vs 24.94 s，看着像 −0.46 s；把 16 rank、6 步都
跑齐后是 **27.37 s 中位数 vs 27.24**（都在 ±0.4 s 噪声里）。**省下的 775 ms 是 kernel 时间，不是关键路径**——
那些除法跑在 vector 单元上、本来就和 cube 并行，把它们删掉并不会缩短墙钟。所以这条的价值是
"**为 vector 单元成为瓶颈的场景（更小 batch、更少卡、或 VAE 占比更高）提前腾出空间**"，而不是当下就变现。
教训：**kernel 时间之和的下降 ≠ 步时下降**，先看它和谁并行。

### F15.2 只融合 elementwise 尾巴：逐位一致，但只有 0.9%

归约复刻不了（答案 1），但链条末尾的 `h*r` → `w*u` → `bf16` 三步**全是 elementwise**，可以逐位复刻：
`dit_norm_tail.py` 的 kernel 算 `y = CAST_RINT(w * (h * r))`（fp32 中间量、同一舍入点），头
（cast/pow/mean/rsqrt）留给 torch，反向用与 autograd 同序的 torch 算子重写（并重算 `u = h*r`，换来每次调用
少存 800 MB 激活）。

`probe_dit_norm_tail.py`（49152×2048，与生产同形）：

| 检查 | 结果 |
| --- | --- |
| 前向 | **bit_equal，0/100,663,296** |
| x 梯度 | **bit_equal，0/100,663,296** |
| weight 梯度 | **bit_equal，0/2048** |
| 前向耗时 | 3.085 → 1.968 ms（−36%） |
| **fwd+bwd 合计** | **12.10 → 11.50 ms（+4.9%）→ 约 0.24 s/步（0.9%）** |

反向那 1.1 ms 的重算把前向的 1.1 ms 几乎吃掉了，所以净收益只有 0.9%——**在步时噪声（±1.5 s）之下**。

**但集成失败了（重要负结果）**：接进训练跑 3 步，**step 1 的 loss 就偏了（28.43 vs 20.47）、step 2 直接 NaN、
forward 从 2.8 s 涨到 67 s**。模块级逐位一致没有在集成后成立——198 个 norm 模块、梯度检查点重算、
ZeRO 分片参数、以及按 N 逐个编译的 kernel，每一个都可能是差异来源。**0.9% 的收益不值得沿着这条线索查下去**，
所以 `--dit-norm-tail` 已从 `train_sft.py` 撤掉（`dit_norm_tail.py` 与探针保留，作为记录）。
教训：**模块级 microbenchmark 的逐位一致只是入场券，不是结论**；`--dit-norm-kernel`（厂商算子那条）当初也是
靠 `probe_dit_norm.py` 在真实模块上验的，才敢说"反向逐位一致"。

另外 `probe_avgdown_equiv.py` 验证了 F14 的零宽 pad 改动：**4 个 AvgDown3D × 4 个 T 全部逐位一致，整编码
latent 也逐位一致**（relL2=0），编码 2369.4 → 2354.2 ms（−0.6%，省的就是那几张整张拷贝）。

### F15.1 稳态步 profile 实测：这条链 1.33 s/步（5.3%）

`--profile-step 2`（`AiCoreNone`，本次只放大 ~0%，步时 24.94 s 与不 profile 一致）抓到 41233 个 kernel。
从 CSV 里按 `Cast → Pows → ReduceMean → Adds → Rsqrt → Mul → Cast → Mul → Cast` 的模式匹配整条链，得到
**394 条链、合计 1333 ms**（去掉窗口里混进来的 matmul/add），即 25 s 步的 **5.3%**。单条链的样子：

| kernel | 时间 | 对应源码 |
| --- | --- | --- |
| `aclnnInplaceCopy_CastAiCore` | 0.99 ms | `hidden_states.float()` |
| `aclnnPowTensorScalar_PowsAiCore` | 1.31 ms | `.pow(2)` |
| `aclnnMean_ReduceMeanAiCore` | 0.70 ms | `.mean(-1, keepdim=True)` |
| `aclnnAdds_AddAiCore` + `Rsqrt` | 0.014 ms | `+ eps`、`torch.rsqrt` |
| `aclnnMul_MulAiCore` | 1.10 ms | `h * r` |
| `aclnnMul_MulAiCore` | 1.12 ms | `weight.float() * h` |
| `aclnnInplaceCopy_CastAiCore` | 1.10 ms | `.to(bf16)` |

**单次 6.3 ms，而两趟融合（先归约出 v，再一趟读写）只需读 0.4 GB + 读写 0.8 GB ≈ 1.0 ms**，即 6× 。
反向还有同量级的 elementwise（未计入这 1.33 s），所以整条链的 fwd+bwd 上限约 **2 s/步（8%）**——这是训练循环里
唯一还有数量级空间的地方。

同一份 profile 还给出两个和 F10 不同的结论：

* `step_trace_time.csv`：wall 25.34 s = computing 21.70 + free 1.80 + **非重叠通信 1.85 s**；通信总量 11.36 s，
  其中 **9.51 s 是重叠的**。F10 的"零重叠"是区间并集只和 matmul/conv 比对的结论，**通信其实大部分被
  elementwise/vector 掩盖了**，所以"省通信"的天花板只有 1.85 s，不是 10.5 s。
* 设备 idle 1.52 s（3713 个 >20 µs 空隙，最大几个 85/60/57 ms，都出现在 `Exp → Transpose` 之间）——
  这是 host 侧准备（`Preparing` 2.80 s）跟不上，属于调度问题，不是 kernel 问题。

**host 侧的账先摆出来**（`operator_details.csv` 的 `Host Self Duration`）：一步里 **256,899 条 host 算子
记录、合计 17.7 s**（占 25.3 s 墙钟的 70%）。但下面会看到，这些时间主要是同步/背压，不是派发：

| host 算子 | 调用 | device ms | **host ms** |
| --- | --- | --- | --- |
| `wait_event` | 552 | 23374 | 46.6 |
| `HcclReduceScatter` | 116 | 10861 | 18.1 |
| `aclnnInplaceCopy` | 7975 | 1374 | **1223** |
| `aclnnInplaceAdd` | 4620 | 1494 | **1082** |
| `aclnnInplaceZero` | 4095 | 196 | **915** |
| `empty_tensor` | **25691** | — | 532 |
| `aclnnMatmul` | 1221 | 3501 | 83 |

`aclnnInplaceZero` 每次调用 host 要 223 µs（device 只要 48 µs）；`empty_tensor` 一步 2.57 万次。
**下一步若要从调度侧拿收益，方向是砍 host 算子数量（allocator/zeros/inplace 合并），而不是再调 bucket 或流。**

按 host 时间排前两名更说明问题（本次导出没有 call stack，无法直接定位调用点）：

| host 算子 | 调用 | host ms | device ms |
| --- | --- | --- | --- |
| `aten::copy_` | 8950 | **7247** | 0 |
| `aten::_local_scalar_dense`（`.item()`） | 2467 | **1943** | 0 |

一步里 8950 次 `copy_`（每次 host 810 µs）和 2467 次 `.item()`，合计 9.2 s host 时间。两种解释必须区分开：
`.item()` 一定同步设备；`copy_` 的 host 时间也可能是**入队背压**（设备队列满时主机会阻塞），那样它反映的是
"设备是瓶颈"而不是"主机是瓶颈"。

**已定位（`scripts/hostprobe/sitecustomize.py`，通过 `PYTHONPATH` 注入、不改训练代码，逐调用点归因；
8 rank × 2 步，探针本身零成本：步时 24.99 s 与不开探针完全一致）：**

| kind | 调用点 | 每 rank 每步 | 时间 |
| --- | --- | --- | --- |
| `tolist` | `train_sft.py:884` `window.loss_values()` | 1 | **210 ms/次** |
| `item` | `zero/low_level_optim.py:657` `calculate_global_norm_from_list` | 2 | **42 ms/次** |
| `copy_` | `zero/low_level_optim.py:712` ZeRO 桶填充 | 301 | 28 ms |
| `tolist` | `flash_attention_npu.py:329/338` `_resolve_packed` | 112 | 4 ms |
| `item` | `packing.py:850/515` | 32 | 1 ms |

**结论：host 侧不是瓶颈，这些数字是同步/背压。** 前两项（一步 210 + 84 ms）都是"主机等设备排空"：
`loss_values()` 在步末对 loss 求 `.tolist()`、梯度裁剪前对 global norm 求 `.item()`；`copy_` 的 93 µs/次也不是
派发成本（正常 <20 µs），而是设备饱和时的入队背压。**砍 host 算子数量没有收益**，这条路关闭。

## F16 数据供给：约 10% 的缺口 + 一个每轮一次的 25 s 停摆（2026-11-15，测量）

稳态步的 `wait` 列并不总是 0：`memcheck`（64 episode）是 0 / 3.56 / 2.57 / **24.95** / 2.03 s。三个臂：

| 臂 | 设置 | 结果 |
| --- | --- | --- |
| 基线 | `--num-workers 4 --worker-threads 4` | 2–3.5 s/步的 wait，step 5 一次 ~25 s |
| 多进程 | `--num-workers 8 --worker-threads 2` | **更慢**：每 worker 线程少，单样本增强变慢，6 分钟没跑完 step 1 |
| 多进程×线程 | `--num-workers 16 --worker-threads 4` | **过载**：load 432（192 核），step 1 6 分钟没跑完 |
| 大 episode 集 | `--max-episodes 256`（排除 epoch 重启） | wait 2.73 / 2.37 s，**step 5 仍 24.89 s** |

结论：

1. **供给缺口约 10%**（4×4）：一个 worker 实测吃 ~3.7 核，32 个 worker ≈ 118 核，加上 8 个 rank 的 OMP 线程
   已接近 192 核上限。**再加进程/线程只会过载**（两个反例见上表），不是配置没调对。
2. **step 5 的 ~25 s 停摆是"部分 rank 的供给掉队"，不是 IO**。`io_sampler.sh`（每秒采 `/proc/vmstat`
   的 `pgmajfault`/`pgpgin` 与 `/proc/diskstats` 的读扇区）与一步 6 步的 run 同拍：

   | step | 1 | 2 | 3 | 4 | 5 | 6 |
   | --- | --- | --- | --- | --- | --- | --- |
   | major faults/s | 0 | 0 | 0 | 0 | **0** | 0 |
   | 磁盘读 MiB/s | 0 | 0 | 0 | 0 | **0** | 0 |
   | MemAvailable | 1687 | 1654 | 1649 | 1658 | **1648** | 1649 GiB |

   **整轮零 major fault、零磁盘读**（机器有 ~2 TB 内存，页缓存根本没被压）→ 之前"页缓存回读"的猜测**被否**。
   真正的形态是**按 rank 掉队**：step 5 各 rank 的 `wait` 是
   `4.46 / 3.48 / 13.14 / 22.17 / 16.14 / 2.75 / 1.65 / 21.11 s`——一半 rank 正常、一半卡十几秒，
   步时被最慢的 rank 决定。

   **epoch 重启也不是原因**：把 `--max-episodes` 从 64 提到 256、再提到 2048（每 rank 每 epoch 的 batch 数
   从 1 提到 64），2–3 s 的常规 wait 和 step 5 的 **25.70 s** 停摆都原样存在；掉队的 rank 子集每次还不一样
   （`2/3/4/7` 与 `3/4/5/7`），所以也不是固定分片的问题。

3. **停摆是周期性的（每 4 步一次），而且连 backward 一起变慢**。12 步序列（`waitseq`）：

   | step | 2 | 3 | 4 | **5** | 6 | 7 | 8 | **9** | 10 | 11 | 12 |
   | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
   | total s | 24.8 | 28.6 | 26.9 | **50.9** | 27.0 | 28.1 | 26.5 | **50.0** | 26.8 | 27.9 | 26.5 |
   | wait s | 0.0 | 3.7 | 2.5 | **26.6** | 2.5 | 3.7 | 1.9 | **9.0** | 2.4 | 3.5 | 1.9 |
   | bwd s | 9.9 | 12.3 | 10.6 | **34.7** | 10.8 | 12.0 | 10.7 | **33.7** | 10.8 | 11.7 | 10.5 |

   **backward 同时慢 3.3 倍**说明不是单纯的"数据慢"：整机的 CPU 被占住了。`npu-smi` 在容器外看不到
   训练进程的 HBM（一直是 3430 MB），所以排除"别的租户抢 NPU"这条只能靠训练自己的指标。

4. **根因是增强（ColorJitter），而且它吃线程**。`probe_augment_cost.py` 在真实尺寸
   （3 相机 × 33 帧 × 540×640 = 0.38 GiB/样本）上拆开测：

   | 变换 | 1 线程 | 4 线程 | 8 线程 |
   | --- | --- | --- | --- |
   | `ColorJitter(0.3,0.4,0.5,0.08)` | 5553 ms | 2093 ms | **1185 ms** |
   | 整条 Compose | 6333 ms | 2334 ms | 1371 ms |

   crop/resize 加起来 <1 ms（切片 + 同尺寸重采样）。**`ColorJitter` 占整条链的 90%**，而且**线程扩展性很好
   （1→8 线程 4.7×）**——这就解释了为什么 `--worker-threads 4` 有效、而 `8 worker × 2 线程` 反而更慢：
   **该加的是每个 worker 的线程数，不是 worker 数**。

5. **机制：整机 CPU 预算被占满，训练进程被饿着**。8 rank × 8 OMP 线程（64）+ 32 worker × 4 线程（128）
   = **正好 192 核**。worker 一进入重活期，训练进程就抢不到 CPU，所以**同一步里 backward 也慢 3.3 倍**。
   三个臂直接验证了这个机制：

   | 臂 | 停摆那步 wait | 停摆那步 bwd | 正常步 |
   | --- | --- | --- | --- |
   | 4 worker × 4 线程，OMP8（基线） | 26.6 s | 34.7 s | 25–28 s |
   | 2 worker × 16 线程，OMP8（worker 更吃 CPU） | **36.1 s** | **44.3 s** | 25–27 s（冷启动 234→143 s） |
   | 4 worker × 4 线程，**OMP4**（给 worker 让出 32 核） | **14.1 s** | **22.4 s** | 25–29 s（不变） |

   **`OMP_NUM_THREADS=4` 把周期性停摆减半且不伤正常步**（已写进两个 bench 启动脚本）。

6. **根治：把 `adjust_hue` 交给 inductor，逐位一致，停摆消失**。`probe_augment_compile.py` 先证明单步可编译
   （`F.adjust_hue` 992 → 598 ms，**bit_equal，0/102,643,200**），然后 `fast_color_jitter.py` 只重写
   `ColorJitter.transform`、保留 torchvision 自己的 `make_params`（同一 `randperm(4)`、同一组均匀抽样、
   同一顺序），实测 `probe_fast_jitter.py`：**3 个种子全部逐位一致，1165 → 780 ms（−33%，8 线程）**。

   **一个必须记住的坑**：把 hue 因子当 Python float 传进编译函数，dynamo 会按*值* guard，而 jitter 每次采样都换值
   → 变成每次调用重编译，实测 **3.7 s/次（比 eager 慢 3 倍）**。必须传 0-dim tensor（`_hue_math` 就是为此把
   因子张量化；HSV 往返的算子照抄 torchvision，逐位不变）。

   8 步 A/B（`--max-episodes 2048`，OMP4，4 worker × 4 线程）：

   | 臂 | step 2–8 平均 | step 5（原停摆点） | 最大步 |
   | --- | --- | --- | --- |
   | 基线 | 28.9 s | 38.7 s（wait 14.1 / bwd 22.4） | 38.7 s |
   | **`COSMOS3_FAST_JITTER=1`** | **26.8 s（−7%）** | **27.7 s（wait 3.0 / bwd 11.0）** | **27.7 s** |

   **周期性停摆消失，50 s 级尖峰没有了**；剩下 2–3 s 的常规 wait 仍然在（供给刚好贴着需求）。
   把 worker 线程从 4 提到 8 没有进一步收益（steps 2–8 均值 27.2 s，step 5 = 28.1 s），所以保持 4 线程。
   部署时把 `TORCHINDUCTOR_CACHE_DIR` 指到本地盘（如 `ext_cache/inductor`），每台先跑一次预热编译。

7. **剩下的供给缺口不在解码，也不在 compose/resize**。两个探针把路摸清了：

   - `probe_decode_threads.py`：PyAV 解 33 帧单线程 90–101 ms，`thread_type=FRAME` + 4 线程 76–89 ms
     （帧逐位相同）。只有 ~20%，还要额外吃 4 个线程——在已经占满 192 核的机器上**不划算，不做**。
   - `probe_item_stages.py`（生产路径：action 缓存 + 增强 + pad16，逐阶段计时）：

     | 阶段 | 每样本 | 占比 |
     | --- | --- | --- |
     | `_augment`（crop+resize+jitter） | 114 ms（编译后；首样本含编译 2.25 s） | 64% |
     | `_read_frames`（3 相机 mp4 解码 + /255） | 435 ms | 34% |
     | compose / quantise / resize_pad / normalise | 27 ms 合计 | 2% |

   **`resize_and_reflect_pad` 对 DROID 根本不重采样**（`scale = min(…, 1.0)` 恒为 1，只有反射填充），
   compose/量化/normalise 合计 27 ms。之前"compose/resize 是大头"的猜测**也否掉了**。
   真正的大头是**解码 435 ms + 增强 114–800 ms（取决于线程数）**，而增强已经编译过；
   剩下的解码是 mp4 逐帧解，没有低风险加速手段（见上）。

8. **uint8 直通能把 wait 压到 0.4 s，但设备侧又还回去**。collate/IPC 是那 2–3 s 的另一半来源
   （float 路径一批 32 样本要 stack 4.1 GiB 并过共享内存，uint8 只有 1.03 GiB）。实测
   `--video-uint8` + `COSMOS3_VIDEO_NORM_KERNEL=1` + fast jitter，8 步：

   | | float + fastjit | uint8 + kernel + fastjit |
   | --- | --- | --- |
   | wait | 1.8–3.0 s | **0.4–0.9 s** |
   | VAE | 10.0–10.1 s | 10.5–11.6 s |
   | 稳态步 | 26.85 s | 26.48 s |
   | 显存 | 55.6 GiB | 57.8 GiB |

   **净收益在噪声内（≈ −0.4 s），显存还多 2 GiB**，所以默认仍走 float 路径；uint8 的价值是"把停摆风险换成设备时间"，
   在数据更紧张（更多样本/更少 worker）的场景才值得换。

9. **VAE 的 encode batch 8 已经是最优点**（`probe_vae_encode_batch.py`，16 样本）：

   | batch | s/sample | 逐位一致 | peak |
   | --- | --- | --- | --- |
   | **8** | **0.294** | — | 18.3 GiB |
   | 12 | 0.314 | True | 28.4 GiB |
   | 16 | 0.405 | **False（relL2 6.1e-4）** | 47.1 GiB |

   更大的 batch 反而更慢（工作集超出 L2），而且 **16 的 latent 不再逐位一致**（conv 的 tiling 随 batch 变），
   显存也会让训练侧 OOM。**这条到此为止，不要再扫。**

10. **加深预取队列会更糟（负结果）**。`--prefetch-factor 6`（PyTorch 默认 2）8 步实测：
    step 4/5 = **29.44 / 30.20 s、wait 4.70 / 5.15 s、bwd 14.1**，比基线（wait 1.8–3.0、bwd 11.0）**全面变差**。
    原因是队列深了就把 6 × 4 worker × 4.1 GiB ≈ 98 GiB 的样本摁在主机内存里，page cache 被挤掉、反向也跟着慢。
    所以那 2–3 s 不是"队列不够深"，而是**每步 4.1 GiB 样本穿过 dataloader 的固有成本**；
    唯一能把它压到 0.4 s 的是 uint8 直通，但设备侧会把时间还回来（F16.8）。**两条路都试过，这是平衡点。**
    开关保留（`--prefetch-factor`，默认 0 = 用 PyTorch 默认），因为它是诊断这个问题的唯一旋钮。

11. **剩下的 2–3 s 常规 wait 对线程预算不敏感**：`OMP_THREADS=2`（再让出 48 核）实测
   step 2–5 = 25.01 / 27.09 / 27.23 / 28.35 s（wait 0.01 / 2.46 / 2.84 / 4.73），与 OMP4 持平——
   停摆仍然没有，但常规 wait 一点没变。所以它不是 CPU 预算问题，而是**每个 worker 串行处理 8 个样本
   （解码 0.44 + 增强 ≈0.8 s ≈ 1.2 s/样本）**留下的稳定缺口，`OMP_NUM_THREADS=4` 就够，不必再降。
   **数据侧到此为止**：再往下要么做帧缓存（25k episode × 0.1 GB 不可行），要么改配方（少解帧/换增强）。

## F17 多机复核：VAE 优化叠加成立，SAC 是"放不下"而不是"没效果"（2026-11-15）

用 node1+node6（16 rank，shard 8 + replica 2）把单机结论逐项搬到多机。

### F17.1 VAE 优化在 16 rank 上确实叠加

| | VAE 段 | 整步 |
| --- | --- | --- |
| F13.1 基线（无内核，16 rank） | 15.05 s | 31.5 s |
| **当前全套**（两趟 norm + AvgDown pad + fast jitter + OMP4） | **9.93 s** | **24.94 s** |

VAE 段 **−34%**，与单机同幅度（单机 10.0 s）。所以"多机上没效果"的说法至少在 16 rank 上不成立——
`vae_ms` 是逐 rank 的实际耗时，不受通信影响。

### F17.2 SAC（selective checkpointing）在多机上是显存问题

| 配置 | 结果 |
| --- | --- |
| `--checkpoint-mode selective`，encode-batch 8 | **`torch.OutOfMemoryError`**：HcclAllreduce 申请 420 MB 失败 |
| 同上 + `--vae-encode-batch 4`（把 VAE 峰值减半） | 跑到 **60.4/61.0 GiB（99%）**，step 2 卡死 |

SAC 保留 attention 输出、只重算 MLP，代价是 **+9.8 GiB**；32 样本/卡时这 9.8 GiB 不存在，所以它
**不是"变慢"，是"跑不起来"**。要让它成立只有三条路：① 开 `--dit-norm-kernel`（省 5.2 GiB，但那是精度取舍）；
② 更小的 per-rank batch（改配方）；③ 更多卡摊薄（每卡样本数不变，没用）。
另外 SAC 的收益上限本来就只有"省一次 attention 重算"≈1.1 s/步（fwd 里 FA 112 次 × 10.07 ms），
在 35 s 的 6 机步里约 3%。

### F17.3 多机独有的成本：backward 的通信与它对 vector kernel 的拖慢

16 rank 的 `step_trace_time`：computing 21.67 s、通信 **10.20 s（其中 8.88 s 重叠）**、free 1.95 s。

但重叠不是免费的：**同一批 vector kernel 在多机上明显变慢**——

| kernel | 单机 | 16 rank |
| --- | --- | --- |
| `aclnnDivs_RealDiv` | 706 次 / 145 ms（最大 1.69 ms） | 822 次 / **763 ms（最大 44.2 ms）** |
| `hcom_reduceScatter` | 1359 ms | 1031 ms，但出现 **119.6 ms 的孤立长尾**（straggler） |

这说明通信 DMA 与访存型 kernel 抢 HBM/L2：**cube 重的 kernel（conv/matmul/FA）不受影响**（均值几乎相同），
访存重的（div/zeros）被拖慢 4–5×。所以多机上"省算力"的优化会被"等带宽"吃掉一部分——这与 6 机基线文档里
"只有减少实际搬运字节的改动才有效"的判据一致。

### F17.3b 按预算切 SAC 也不行：4 GiB 预算换 0 收益、显存 98%

`--selective-budget-gib` 本来就是"按显存预算保留多少层的 attention 输出"的旋钮（`_SelectiveBudget.decide`：
`sac` 路径下 `max_saved_layers = budget / residual_bytes`）。默认 8 GiB 会 OOM，于是试着降到 4 GiB：

| 臂 | step 2 / 3 | bwd | 显存 |
| --- | --- | --- | --- |
| 基线（full checkpointing） | 24.94 / 27.43 s | 11.01 / 10.74–11.53 s | 55.3 GiB |
| **SAC，budget 4 GiB** | 25.46 / 28.41 s | **11.40 / 12.82 s** | **59.4–60.8 GiB（97–99%）** |

**反向没有变快（反而更高，在噪声内），显存却实打实涨了 4 GiB、逼近上限。** 原因也清楚：SAC 省的是
"不重算 attention"，而全量重算里 FA 只占 1.1 s/步，4 GiB 预算只买回 4/28 层 ≈ 0.16 s——**在 25–28 s 的步里测不出来**。
所以 SAC 在这套栈上不是"多机不叠加"，而是**它本身就不是一个速度优化**：它是"用显存换 batch"的工具，
32 样本/卡时既没有显存可换、也没有速度可拿。6 机基线文档里"反向不变慢、+9.8 GiB"的结论与此一致，
现在补上了"按预算切也不行"的实测。

### F17.3c 跨节点带宽几乎不掉——所以 6 机的通信问题不是带宽（`bench_hccl.py`，16 rank）

| 集合通信 | 8 rank（1 机） | **16 rank（2 机）** |
| --- | --- | --- |
| reduce_scatter 512 MiB | 119 GiB/s | **112.65 GiB/s（−5%）** |
| all_reduce 512 MiB | 64 GiB/s | **59.68 GiB/s（−7%）** |

**跨节点只掉 5–7%。** 512 MiB 的 all_reduce 只要 8.38 ms，而一步的梯度量级（0.3B 可训练参数 bf16 ≈ 0.6 GB）
摊到 all-reduce 上是**十几毫秒**——比 profile 里那 1.3–1.9 s 的"未重叠通信"小两个数量级。

**结论：多机暴露出来的那 1–2 s 不是带宽，是同步与长尾**（`hcom_reduceScatter` 出现 119.6 ms 的孤立长尾就是证据）。
这条把"6 机 shard 策略 A/B"的预期收益也压低了：既然 fabric 在 2 机上只掉 5%，shard 8 vs 16 的差别主要在
**集合通信的个数与同步点**，而不是字节吞吐——2 机上实测持平正好印证。**6 机仍然值得跑那一次 A/B，
但不要期待它是数量级的杠杆。**

### F17.4 可试的多机配置：`--hsdp-shard-size 16`（纯 ZeRO，无 replica 组）

| | step 2 / 3 | 显存 |
| --- | --- | --- |
| shard 8（replica 2） | 24.94 / 27.43 s | 55.3 GiB |
| **shard 16（纯 ZeRO）** | **24.73 / 27.16 s** | **53.3 GiB（−2 GiB）** |

2 机上持平、显存少 2 GiB、并且**去掉了跨节点 replica all-reduce**——6 机上 replica 组是 6 台，这一项才可能变贵。
**注意**：改 shard 会改变梯度归约顺序（step-1 loss 从 20.21 变成 21.22），这是部署决策，不是 kernel 优化；
6 机上的真实收益需要等 node5/7/8/9 空出来才能验。

## F18 最终标准基准（node1+node6，16 rank，2026-11-15）

配置：`bench_online_step_2node.sh`，32 样本/rank、grad-accum 1、HSDP shard 8 + replica 2、
4 worker × 4 线程、`OMP_NUM_THREADS=4`、`COSMOS3_FAST_JITTER=1`、`--vae-norm-kernel`（两趟）、
`--fast-vae-encoder`、`--vae-content-pad 16`、`--max-episodes 2048`，6 步。

| 指标 | 值 |
| --- | --- |
| 步时（最慢 rank 中位数） | **27.24 s**（5 个稳态步） |
| 吞吐 | **1.18 样本/s/rank → 18.8 样本/s（16 卡）** |
| 相位 | **VAE 9.92 s（36.4%）**、fwd 2.86、bwd 11.57、optim 0.48、wait 2.62 |
| 显存 | peak allocated 48.2 GiB、device used **56.2 GiB** / 61 |
| MFU | **27.0%**（transformer，含 4× checkpointing 的 FLOP 口径） |
| 每 rank 均衡 | 16 rank 的 step 全部 27.22–27.24 s、VAE 9.73–9.92 s |

**扩展性**：单机 8 卡 26.8 s/步、9.55 样本/s；两机 16 卡 27.24 s/步、18.8 样本/s
→ **同一步时下吞吐翻倍（96% 扩展效率）**，说明 VAE + 训练循环的优化在多机上是线性叠加的。

> 读数注意：`summarize_online_metrics.py` 只读单机的 `metrics/` 目录，16 rank 时会把它算成
> "9.40 样本/s、MFU 13.5%"——那是 8 个 rank 的样本数除以 16 卡的峰值。跨机要按 rank 数折算。

## F19 `--compile-blocks` 在本栈不可用（负结果，2026-11-15）

剩下的"主机派发空泡"（Preparing 2.4 s + idle 1.7 s/步）唯一的通用手段是把 DiT 层编译/图捕获。实测
`--compile-blocks`（`torch.compile(layer, backend=auto, fullgraph=True, dynamic=True)`）：

```
torch._dynamo.exc.Unsupported: Import failure
  (rank1, 第一个 forward)
```

dynamo 在 trace 的路径里遇到一个**函数内的 `import`**（`Unsupported: Import failure`），`fullgraph=True` 下直接
拒绝。`--compile-vae` 早前也是同一个原因被关掉（自定义 kernel 处 graph break）。要重开这条线，得先把
attention/policy 路径里的惰性 import 提到模块顶层——那是模型代码的改动，不在"精度对齐的算子优化"范围内，
所以**这条也记成负结果**，别再重复。

## F20 全部臂的总表（`collect_arm_summary.py`，2026-11-15）

`scripts/collect_arm_summary.py <logs>` 把每个臂的日志解析成稳态中位数（step 1 冷启动剔除），一张表看完整个
campaign。`--max-episodes 2048`、32 样本/rank、pad 16，除标注外均为 node6 单机 8 卡：

| 臂 | 步数 | step | fwd | bwd | vae | wait | 显存 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ditkernel`（`--dit-norm-kernel`，opt-in） | 2 | **25.20** | **2.29** | 11.23 | 10.06 | 1.81 | **50.4** |
| `summean`（`sum/H`，现默认） | 2 | 25.91 | 2.81 | 10.91 | 10.00 | 1.47 | 57.0 |
| `u8fast`（uint8 直通 + 融合 norm） | 7 | 26.28 | 2.81 | 11.94 | 11.20 | **0.43** | 57.6 |
| `aug2x16`（2 worker × 16 线程） | 7 | 26.75 | 2.82 | 10.82 | 9.97 | 2.35 | 52.8 |
| `twopass`（两趟 norm A/B 的 on 臂） | 4 | 26.98 | 2.82 | 10.66 | 10.07 | 2.40 | 57.3 |
| `datae2048`（256/2048 episode 对照） | 5 | 27.01 | 2.82 | 10.37 | 10.02 | 2.35 | 55.6 |
| `fastjit`（编译版 ColorJitter） | 7 | 27.08 | 2.79 | 10.96 | 10.00 | 2.47 | 55.5 |
| `omp4`（OMP_NUM_THREADS=4） | 7 | 27.16 | 2.83 | 10.66 | 10.12 | 2.29 | 55.6 |
| `prefetch6`（负结果） | 5 | 27.33 | 2.80 | 11.37 | 10.03 | 2.68 | 57.3 |
| `bucket12/48`、`async_off`（通信调度，全部无变化） | 4 | 27.35 | 2.81 | 10.4–10.7 | 10.5–10.6 | 2.2–2.4 | 55.6 |

**两条要读出来的信息**：

1. **稳态步的噪声约 ±0.4 s**（同一配置的不同 run 在 26.75–27.35 之间），所以除了 `ditkernel`（−1.9 s、−5 GiB）
   和 `u8fast` 的 wait（0.43 s）之外，**单项优化在步级是分不开的**——它们的效果是靠隔离测量（微基准、kernel
   profile、编码级 A/B）确认的，例如两趟 norm 的 −0.46 s 是在 VAE 段上看的，sum/H 的 −775 ms 是 kernel 总时间上
   看的。**不要用"臂与臂之间的步时差"去论证单项收益。**
2. `canvas16/base/wt4`（7.3–8.8 s）是早期 8 样本尺度的臂，不与本表同口径，仅作历史参照。

## 负结果与边界（避免重复投入）

1. **VAE 配置**：encode-batch / chunk / dtype 的组合空间基本是平的（0.449–0.472 s/样本，±2.5%），
   最快配置只值 0.1–0.2 s/步，且显存更高。不要再扫。
2. **optimizer/all-gather**：optim 段实测 0.29 s/步（0.9%），`--overlap-allgather` 的上限就是这 0.3 s；
   在 3 步中位数上很可能落在噪声里。它不是主项。
3. **梯度累积**：摊薄的是同样的 0.3–0.5 s/步固定成本，按每样本算是 0.3% 量级；官方 8192 全局 batch
   仍然需要它，但不要指望它提速。
4. **Transformer elementwise 融合**：按 113k token × 2048 估算，add+norm / relu² 的 GM 流量只有
   0.1–0.3 s/步（<1%）；`npu_rms_norm` 之类的替换不值得作为主线。
5. **conv 自研**：conv 本体已 74% MFU（326 TFLOP/编码 ÷ 1.246 s），没有空间；`channels_last_3d` 在
   torch_npu 上直接不支持（`Only contiguous_format or preserve_format is supported`），TransData 也消不掉。
6. **梯度通信调度**：bucket 12→48 无变化（bwd 10.41 s 两臂相同）；异步 reduce-scatter 无变化
   （10.67→10.69 s）。串行化是设备级的，调度层没有余地。
7. **VAE 的四类"搬运"全部到顶**：`cat` 1261 GB/s、Transpose ≈1.0 TB/s（都是带宽 bound，自研 kernel 没有
   余量）；pad 只有 345 GB/s 但整段只有 59 ms，值得做的只有"跳过零宽 pad"（已做）；conv 的 TransData
   **无法用预转 weight 格式消掉**（`probe_conv_transdata`：FRACTAL_Z_3D/FRACTAL_NZ 转换成功但 conv 拒绝，
   NDC1HWC0 被 Python 拒绝）。**VAE 侧到此为止。**
8. **DiT norm 的行归约复刻不了**（F15）：24 种候选归约全部不一致，最好也差 35% 的行 1 ulp。这条 5.3% 的
   路径在"逐位一致"下关闭；厂商 `npu_rms_norm` 只差 0.0001% 的元素，已做成 opt-in。
9. **首步冷启动与 data_starvation**：即便有 action 缓存，首步仍受 **32 个 spawn worker 的主模块重导入 +
   首 batch 增强**支配；action 缓存对两者都没有可见效果。`--worker-threads 4` 是对症的解药，
   但**"再加 worker"已被证伪**（F16：8×2 更慢、16×4 过载，机器已到 192 核上限）。

## 推荐与在线路径的现实下限

1. **action memmap 缓存**：所有长跑都开 `--action-cache`。逐位一致，构建 97 s，消除文件边界停摆；
   不解决首步冷启动（那是 worker spawn + 增强争用）。
2. **画布裁剪**：保守用 pad 64（relL2 2.48e-3，低于 bf16 存盘噪声；VAE −6.5%）；pad 16（−18.2%，
   relL2 1.48e-2）需要更长的 loss A/B，且 RoboLab serving 必须带同一参数。
3. **worker 线程**：按 `wt4` A/B 的结果，用 `--worker-threads 4`（PyTorch worker 默认 1 线程；OMP 环境变量
   到不了 worker）。目标配置：`--action-cache --worker-threads 4 --vae-content-pad <best>`，
   预期 step ≈ 28.5–29.5 s、MFU ≈ 25%，并且 step 5 类停摆消失。
4. **不要做**：VAE 配置继续扫、`--overlap-allgather`（上限 0.3 s）、grad-accum 提速（0.3%）、
   transformer elementwise 融合（<1%）。
5. **在线路径的现实下限**：VAE 13.2（裁画布后）+ fwd 2.76 + bwd 12.0 + optim 0.29 ≈ **28.3 s/步**，
   相对基线 32.78 s 约 **−13.7%**；48 卡外推约 **18.6 天**（相对 21.7 天）。再往下需要 VAE 结构
   （帧级复用/更快 Conv3D）或改配方（增强/分辨率），不是调度能拿的。
6. **跨步 VAE/bwd 重叠**：guide 08 明确警告"两条流争同一 HBM/L2，实际收益可能为 0"；在拿到
   profiler 时间线与同 seed loss 对拍之前，不作为承诺项。

## 六机部署清单（node6 验证通过后）

> **基线口径（2026-11-15，node1+node6 16 rank，6 步）**：27.24 s/步、18.8 样本/s、MFU 27.0%、
> VAE 9.92 s（36.4%）、fwd 2.86、bwd 11.57、optim 0.48、wait 2.62、device_used 56.2/61 GiB、
> 16 个 rank 全部 27.22–27.24 s。六机部署后先复现这一行，再谈改进。

1. **等 node5/7/8/9 的多机测试结束**，再把 `colossalai/booster/plugin/hybrid_parallel_plugin.py` 与
   `examples/worldmodel/cosmos3/` **和 `examples/worldmodel/cosmos3_harness/`** 一起同步到六台的本地盘副本
   （不要动共享 NFS 上正在被测试使用的脚本）。**两个目录必须同时同步**——`train_sft.py` 依赖 harness 的函数
   （`apply_checkpoint`），只同步一个会直接 `ImportError`，这一坑已经踩过两次。同步后用
   `md5sum train_sft.py cosmos3_harness/checkpoint.py droid_dataset.py` 三处对拍。
2. **action 缓存**：在每台本地盘各建一份（`build_action_cache.py`，97 s / 1.5 GB），或把 node6 的
   `action_cache/` 拷到各台同名路径；`--action-cache /mnt/disk0/jiangzs/nzf/action_cache`。
3. **worker 线程与 CPU 预算**：`--worker-threads 4`；**并加 `OMP_NUM_THREADS=4`**（F16：8 rank 的 OMP 线程
   与 32 个 worker 正好占满 192 核，把训练侧 OMP 从 8 降到 4 会让周期性停摆从 26.6 s 减到 14.1 s，
   正常步不变）。不要再加 worker：8×2 更慢、16×4 直接过载。
4. **画布 pad**：**上面那行 27.24 s 的基线是 `--vae-content-pad 16`**（relL2 1.48e-2，与 bf16 VAE 计算噪声
   同量级）。若选保守的 `--vae-content-pad 64`（relL2 2.48e-3，低于 bf16 存盘噪声），**实测**
   VAE 9.92 → **11.43–11.50 s（+1.5 s/步，约 +5.5%）**、显存 56.2 → 57.4 GiB。
   两种都必须让 RoboLab serving 带同一参数。**注意**：改 pad 会改变 VAE 的 (B,C,S)，融合 norm 内核要按新形状
   重编译一次（~20 个，数分钟），换配置时留出这段时间。
   **增强**：加 `COSMOS3_FAST_JITTER=1` 与 `TORCHINDUCTOR_CACHE_DIR=<本地盘>/ext_cache/inductor`
   （F16.6：逐位一致、稳态 −7%、周期性停摆消失；首次运行每 worker 编译一次，之后走磁盘缓存）。
5. **指标**：沿用 `summarize_online_metrics.py`（step/phase/吞吐/显存/MFU/per-rank），每步写
   `rank-*.jsonl`；长跑保留 step 1 与任何 data_starvation 事件。
6. **融合 VAE norm（`--vae-norm-kernel`，bench 默认已开）**：
   - 数值安全：latent 逐位一致（relL2=0），可无风险上线；两趟变体（S ≥ 65536 自动启用）同样逐位一致，
     比 one-pass 再省 **0.46 s/步**（VAE 段 −4.4%，F13.3）。
   - 显存：**+6.6 GiB（one-pass）/ +9 GiB（two-pass 相对不开内核）**（node6 单机，**未加守卫时**会爬到
     57.5/61 GiB）。已加自适应守卫（F14.1）：`free < COSMOS3_EMPTY_CACHE_GIB`（默认 8 GiB）就
     `empty_cache()`，六步验证 device_used 平在 **55.1±0.2 GiB**、步时无代价。六机照此配置即可，
     若 batch/帧数更大仍要先确认余量。
   - 启动：首次运行每个 (B,C,S) 编译一次（~12–16 s），**8 个 rank 由 `~/.tilelang` 上的 flock 串行化**
     （one-pass 与 two-pass 各一把锁；两趟路径最初漏了锁，实测直接 SIGSEGV），一次性多花 2–3 分钟。
   - 不要同时开 `--vae-compile jit`（torch.compile 会在自定义 kernel 处 graph break，收益归零）。
7. **AvgDown3D 零宽 pad 跳过**（`vae_fast_encoder.py`，默认生效）：逐位一致（16/16 模块·T 组合 +
   整编码 latent），编码 −0.6%。
8. **DiT norm 融合（`--dit-norm-kernel`，默认关闭）**：反向逐位一致，前向每 90 万元素差 1 个 ulp
   （0.0001%），收益 ~0.9–1.3 s/步（3.5–5%）。**是否开启是精度决策**，见 F15。
9. **uint8 数据直通（可选，`--video-uint8` + `COSMOS3_VIDEO_NORM_KERNEL=1`）**：
   - 它解决的是**数据供给**（wait 2.3–2.7 s → 0.4 s），在数据紧张的机器上有净收益（node1 −8.8%）；
     node6 上因设备侧归一化抵消，基本中性（F8.4），新内核把归一化压到 5.9 ms/调用后应转为小幅净收益。
   - 数值：bf16 输出与 host 逐位一致（F11）。
10. **六机上要做的第一个 A/B：`--hsdp-shard-size 48`（纯 ZeRO）vs `8`（replica 6）**。
    2 机实测持平（24.73 / 27.16 vs 24.94 / 27.43 s）、显存少 2 GiB，但 replica 组的代价随节点数增长，
    6 机才能定论；改 shard 会改变梯度归约顺序（step-1 loss 20.21→21.22），是部署决策不是 kernel 优化。
11. **不要在六机上再试这些**（都已实测否掉，别再花机时）：SAC（budget 8 直接 OOM、4 GiB 无收益且显存 97%）、
    跨步 VAE/bwd 重叠（两条流设备串行化，只藏住 2%）、`--compile-blocks`（dynamo `Import failure`）、
    `--prefetch-factor > 2`（wait 反而涨到 4.7–5.2 s）、解码多线程（20% 换 4 线程）、
    conv weight 预转格式（conv 拒绝）、VAE encode batch > 8（更慢且 16 不再逐位一致）。

## 下一步设计：VAE 与 backward 的跨步重叠（**已否定**，2026-11-15）

**结论先行：这条设计在本栈上不成立，不要再做原型。** `probe_stream_overlap.py` 把真实的 VAE 编码（1250.9 ms）
和一段 cube 主导的 matmul 负载（8192² × 60 次 = 245.8 ms）分别放在两条流上并发执行：

| 组合 | 时间 |
| --- | --- |
| 编码单独 | 1250.9 ms |
| matmul 单独 | 245.8 ms |
| 顺序执行 | 1487.3 ms |
| **两条流并发** | **1481.2 ms（只藏住 6.1 ms，2%）** |

设备侧不并行：两条流上的任务被串行化（与 F10 里 async reduce-scatter 无效的结论同源）。所以"把下一批的 VAE
放到 backward 旁边"最多只能填那 1.8 s 的设备空闲，**藏不住 10 s 的 VAE**，而代价是显存翻倍和一堆同步复杂性。
guide 08 的警告在实测里成立了。

（显存那边现在反而有余量：开 `--dit-norm-kernel` 后 steady 只用 50.4/61 GiB；但没有可藏的工作，就没必要动。）

---

以下为原设计记录（**保留但不实施**）。

### 原设计

### 为什么是它

> **口径更新（2026-11-15，F15.1 之后，请以本段为准）**：当前稳态步 25.3 s = computing 21.7 + free 1.8 +
> 暴露通信 1.85；其中 VAE 段 10.0 s、fwd 2.8、bwd 10.3。**设备 idle 只剩 1.8 s，所以"填 idle"不再是理由**；
> 这个设计现在的价值是"把 10 s 的 VAE（conv/vector 单元 + 大 DMA）从关键路径挪到 bwd（矩阵单元）旁边"，
> 收益取决于两类引擎能否真并行——这正是 guide 警告 HBM/L2 争用、必须先拿时间线验证的点。

基线 32.78 s 的设备账：VAE 15.19（99.9% 占用）+ fwd 2.76 + bwd 11.99 + optim 0.29 ≈ 30.2 s；
整步真在算的时间按 ops 草稿的分段占用（fwd/bwd 60–77%）估算约 25–26 s，**设备 idle 约 7 s（22%）**，
其中 data wait 2.9 s、fwd/bwd 内的通信/派发空泡约 4 s。重叠只能填 idle，不能减少工作量：
上限是这 4 s，而不是 VAE 的 15 s。

### 机制

```
main stream : [wait batch k] → fwd(k) → bwd(k) → optim(k) → fwd(k+1)
VAE stream  :                   [vae(batch k+1) on stream B] ──event──┘
```

- batch k+1 由 loader 预取；在 step k 的 fwd 之前把它的 VAE 发到 stream B（双缓冲 latent）；
- step k+1 的 `pack_batch` 只做 tokenize/packing，`latents=` 直接吃 stream B 的结果；
- 显存：VAE eb4 峰值 8.43 GiB + 模型峰值 40.3 GiB ≈ 48.7 GiB allocated，device 约 56 GiB，
  在 61 GiB 以内；eb8 的 14.89 GiB 会贴边，原型先用 eb4（VAE 慢 2%）。
- 风险：VAE 与 bwd 同抢 Cube/HBM，两者可能各慢 5–10%；净收益要按整步墙钟与 profiler 时间线判定，
  不能只看单段。验收：同 seed loss 一致、step 2–4 中位数下降 ≥1.5 s 才保留。

### 备选：不做重叠时的收益上限

- 数据 wait 2.9 s → 0（worker 数/线程调优）：step ≈ 29.9 s（−8.8%）。
- 画布 pad 32：VAE ≈ 13.9 s（−8.5%）：step ≈ 31.5 s（−4%）。
- 两者叠加：step ≈ 28.6 s（−13%），MFU 22.4% → 25.7%。
- 再加跨步重叠 2 s：step ≈ 26.6 s（−19%），MFU ≈ 27.6%。

---

## 复现命令（node6，容器内）

```bash
# 基线：8 卡、32 样本/rank、64 episode、8 步
bash .../scripts/bench_online_step.sh online_base 32 8 --max-episodes 64

# action 缓存（一次性）：success + failure，约 97 s
docker exec nzf-cai-dev bash -c 'cd /mnt/disk0/jiangzs/nzf/ColossalAI && PYTHONPATH=. \
  /mnt/disk0/jiangzs/nzf/cai-npu/bin/python examples/worldmodel/cosmos3/scripts/build_action_cache.py \
  --roots /mnt/disk0/jiangzs/datasets/Cosmos3-DROID/success /mnt/disk0/jiangzs/datasets/Cosmos3-DROID/failure \
  --out /mnt/disk0/jiangzs/nzf/action_cache --force'

# 带缓存的 64-episode 对拍
bash .../scripts/bench_online_step.sh cache64 32 6 --max-episodes 64 \
  --action-cache /mnt/disk0/jiangzs/nzf/action_cache

# 指标（step/vae/fwd/bwd/optim/wait/显存/MFU/per-rank）
docker exec nzf-cai-dev bash -c 'cd /mnt/disk0/jiangzs/nzf/ColossalAI && PYTHONPATH=. \
  /mnt/disk0/jiangzs/nzf/cai-npu/bin/python examples/worldmodel/cosmos3/scripts/summarize_online_metrics.py \
  --metrics-dir /mnt/disk0/jiangzs/nzf/bench_online/runs/cache64/metrics --world-size 8'

# F14：任何 profile 的"第二角度"（idle / launch-bound / 连跑 / host 算子），CSV 目录即可
python examples/worldmodel/cosmos3/scripts/analyze_profile_deep.py /tmp/prof_norm/npu-*/ASCEND_PROFILER_OUTPUT

# F14/F15 的四个探针（各 1–5 分钟）
python scripts/probe_layout_ops.py --checkpoint <ckpt>          # cat / pad 的带宽底价
python scripts/probe_conv_transdata.py --checkpoint <ckpt>      # conv weight 预转格式
python scripts/probe_nemotron_norm.py                           # DiT norm 的行归约能否复刻
python scripts/probe_dit_norm.py                                # npu_rms_norm 差别 + 反向逐位性
python scripts/probe_norm_ulp.py                                # 生产尺寸上的 1-ulp 比例
python scripts/probe_avgdown_equiv.py --checkpoint <ckpt>       # AvgDown 零宽 pad 逐位性

# VAE 配置扫描 / 画布扫描 / 调度 A/B
docker exec nzf-cai-dev bash -c '... bench_vae_sweep.py ...'
docker exec nzf-cai-dev bash -c '... bench_vae_canvas.py ...'
bash .../scripts/ab_online_scheduling.sh node6
bash .../scripts/ab_targeted.sh node6
bash .../scripts/run_integrated.sh node6

# F7：uint8 数据直通 A/B（node1）
bash .../scripts/run_uint8_ab.sh node1

# F8：融合 RMSNorm —— 逐位门禁 + 端到端 + 整步 A/B
docker exec nzf-cai-dev bash -c 'cd /mnt/disk0/jiangzs/nzf/ColossalAI && PYTHONPATH=. \
  /mnt/disk0/jiangzs/nzf/cai-npu/bin/python examples/worldmodel/cosmos3/scripts/probe_norm_kernel.py \
  --iters 10 --checkpoint /mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-Policy-DROID --batch 8'
docker exec nzf-cai-dev bash -c '... probe_tile_ops.py'   # CAST_RINT/标量运算语义
docker exec nzf-cai-dev bash -c '... probe_conv_shapes.py' # conv 形态与 MFU
bash .../scripts/run_norm_kernel_ab.sh node6
```

## F21：六机（48 卡）实测与多机税归因（2026-11-15）

六台机器（node1/5/6/7/8/9）全部空闲后，用 `scripts/run_6node_train.sh` 起了 48 rank 的正式训练（`cosmos3_6node_1115`）：HSDP shard 8 + replica 6、32 样本/rank、全量数据集、`--save-every 100`，其余开关与配置 A 完全一致。

### 结果：优化在多机上成立，多出来的 4–5 s 不在带宽

| 口径 | 16 rank | 48 rank |
|---|---|---|
| 步时中位数 | 27.37 s | 29.89 s（step 3–30）/ 28.38 s（step 50–52） |
| VAE 段 | 10.04 s | 9.86–9.92 s |
| forward | 2.84 s | 2.77–2.81 s |
| backward | 11.54 s | 12.1–14.3 s |
| 吞吐 | 18.7 样本/s | 51.4 样本/s |
| 单卡吞吐 | 1.169 样本/s | 1.07 样本/s |

VAE 与 forward 与 16 rank 逐项相同（差 <1%），两块 kernel 优化线性叠加；48 卡每卡掉 8%，MFU 约 24.6%。

### 归因：backward 末尾的集合通信在等数据最慢的 rank

同一步内各 rank 的 `perf.backward_ms` 相差 4–6 s，而 `skew` 恒为 1.00（所有 rank 同步进入下一步），说明差的是 **backward 的起点**：某个 rank 等数据晚了，其余 rank 先算完，在 `LowLevelZeroOptimizer.backward` 收尾的 `_reduce_grad` + `torch.npu.synchronize()` 处干等。逐 rank 数据（`scripts/pull_6node_metrics.sh` + `scripts/analyze_multinode_spread.py`）：

| step 区间 | 步时中位 | bwd 最小（≈纯计算） | spread | 占步时 |
|---|---|---|---|---|
| 3–9 | 30.87 s | 9.46 s | 6.31 s | 20.4% |
| 10–19 | 29.86 s | 9.43 s | 6.23 s | 20.9% |
| 20–29 | 29.44 s | 9.44 s | 5.42 s | 18.4% |
| 30–39 | 29.05 s | 9.46 s | 5.29 s | 18.2% |
| 40–49 | 28.67 s | 9.47 s | 4.75 s | 16.6% |
| 50–52 | 28.38 s | 9.53 s | 4.22 s | 14.9% |

`bwd 最小` 稳定在 9.43–9.53 s（纯计算），spread 却从 6.31 s 掉到 4.22 s，**说明其中约 2 s 是冷启动瞬态**（page cache 逐渐变热），剩余约 4.2 s（15%）是结构性等待。按节点统计数据等待中位数：

| step 区间 | node1 | node5 | node6 | node7 | node8 | node9 |
|---|---|---|---|---|---|---|
| 3–9 | 1.96 | **4.34** | 1.58 | 1.82 | 1.92 | **4.27** |
| 50–52 | 1.68 | **3.51** | 1.40 | 1.69 | 1.54 | 2.58 |

node5/node9 一直是最慢的两个数据源（初始 2.5 倍，现在仍约 2 倍），其余四台 1.4–1.7 s。

分阶段看（step 20 以后的中位数），**计算是均匀的，只有数据就绪时间不同**：

| 节点 | VAE | forward | backward 中位 | backward 最小 | 数据等待 | optim |
|---|---|---|---|---|---|---|
| node1 | 9.80 | 2.77 | 12.70 | 9.85 | 1.63 | 0.36 |
| node5 | 9.84 | 2.77 | 10.88 | 9.39 | **3.61** | 0.38 |
| node6 | 9.91 | 2.77 | 12.74 | 10.50 | 1.46 | 0.31 |
| node7 | 9.90 | 2.78 | 12.55 | 9.60 | 1.70 | 0.34 |
| node8 | 9.86 | 2.77 | 12.58 | 9.41 | 1.57 | 0.32 |
| node9 | 9.91 | 2.77 | 11.42 | 9.42 | **2.74** | 0.39 |

node5/node9 的 backward 中位数反而**更低**——因为它们到得最晚、不用等；其余四台的 12.6 s 里有约 2–3 s 是等它们。所以绑核的现实目标不是那 4.6 s 全部，而是把 node5/node9 多出来的约 2 s 数据等待压到其它四台的水平，对应约 7% 的步时。

### 两次 41–43 s 的异常步（step 5 与 step 9）

不是通信故障。node5 的全局 rank 9 单卡等数据 **17.3 s / 19.0 s**，其余 47 个 rank 在同一时刻全部空闲等集合通信（bwd 22–28 s，而 node5 其它 rank 只有 9.4 s）。六台采集器的 `power_w` 在窗口内同时从 ~330 W 掉到 ~105 W 并持续 13–17 s——设备真空闲。此后到 step 52 再未出现。

### 硬件差异：node5/node9 是 4 NUMA 域机器，且全机未绑核

六台都是 Kunpeng-920、4 路 × 48 核、2 TB 内存、同款 910B2，但 node1/6/7/8 暴露 **8 个 NUMA 域**（每域 24 核），node5/node9 只有 **4 个**（每域 48 核）。设备侧的 PCI NUMA 节点在六台上排布一致（C1/C2→socket3、81/82→socket2、01/02→socket0、41/42→socket1）。默认没有任何绑核：每个 rank 和每个 dataloader worker 的 `Cpus_allowed_list` 都是 `0-191`，单个 rank 的 16 GiB 内存散落在 2–5 个域上；实测一次 32 MiB 分配在未绑核时整个落在随机域（N7），绑到 `0-47` 后 8193 页全部落在该 socket（N1）。数据侧 CPU 也印证了效率差异：同样产出，node5 的 32 个 worker 平均占用 98 核、读盘 4.8 MB/s，node1 是 82 核、0.9 MB/s。

### 已实现、尚未上机：每 rank 绑到其 NPU 所在 socket

`examples/worldmodel/cosmos3/numa_affinity.py`：由 `npu-smi info` 的 Bus-Id + `/sys/bus/pci/devices/*/numa_node` 现算本机映射（不做硬编码表），取**整个 socket**（48 核）而不是 24 核半域，因为一个 rank 连 worker 约 30 个线程。`train_sft.py` 在 `main()` 第一行按 `LOCAL_RANK` 绑自己，之后 fork/spawn 的 worker 继承；`bench_online_step_6node.sh` 的 `NUMA_PIN=1` 负责生成并导出映射（表为空会报错退出）。六台的映射一致：

| rank | 0,1 | 2,3 | 4,5 | 6,7 |
|---|---|---|---|---|
| cpuset | 144-191 | 96-143 | 0-47 | 48-95 |

验证：单测 6 通过 2 跳过（Linux 专有两项在 macOS 跳过），`pytest examples/worldmodel/cosmos3/tests` 245 通过；容器内实测绑定生效、fork 子进程继承、内存随之落在目标 socket。默认关闭，不设 `NUMA_PIN` 时训练路径逐位不变。

### 绑核收益的先验测量（单进程探针，未重启训练）

`scripts/probe_numa_data.py` 在**同一台训练机空闲核上**（`nice -n 19`、2 线程、`OMP_NUM_THREADS=2`、编译版 jitter）重放 `__getitem__` 的全部阶段，逐轮切换亲和性，取第 2–3 轮的每样本 CPU 时间：

| 机器 | 未绑核 | 绑 socket A | 绑 socket B | 读帧（未绑核） |
|---|---|---|---|---|
| node1（8 域） | 8549 ms | 7845（−8.2%） | 7548（−11.7%） | 878 ms |
| node5（4 域） | 9630 ms | 9281（−3.6%） | 9061（−5.9%） | 1244 ms |

两点：**node5 未绑核时每样本要多花 9–17% 的 CPU**（与进程级实测的 82 核 vs 98 核一致），**绑到 socket 后能收回 4–12%**，且 node1 的两个绑核臂都稳定低于未绑核的两个臂。绝对值也对得上：探针的 8.5 / 9.6 核秒/样本，与训练进程实测的 9.0 / 10.7 核秒/样本同量级，说明探针有代表性。

代价与局限：探针在满载机器上跑（wall/cpu ≈ 0.5，即只拿到一半的调度机会），node5 的两轮对照本身差 7%，所以置信区间不窄；但方向在两台机器、两个绑核臂上一致（4:4）。绑核预期能收回 node5 相对其它节点的那约 1.9 s 数据等待差，对应 5–7% 步时。

### 判据（实验设计）

因为 spread 里有约 2 s 的冷启动瞬态，**比较必须落在同一 step 区间**（例如都取 step 30–50 的中位数），否则会把缓存变热的收益误记到绑核头上。预期收益上限是当前的 4.2 s spread（14.9%），现实目标是压掉 node5/node9 相对其它节点的 1.8 s 数据等待差。

## F22：算子级调研（npu-guide + tilelang-ascend + 论文）与候选排序（2026-11-15）

带着 F1–F21 的实测结论重读了 `deepinfer/docs/npu-guide`（18k 行，含 13 个已 clone 的参考仓库），重点核对三处与我们直接相关的判断。

### 核对 1：Conv3D 的位置——之前的结论被独立预算复核为正确

guide §4.2.2b 用官方判据重推了 910B 上的 Conv3D：`baseM×baseN=128×128` 时 MTE1 有 2.0× 余量（不是瓶颈），**真正的约束是 L1 容量只够 2 级流水**（C_in=384 时 167 KB/级）与 **`baseK ≤ 64KB/(baseM×2B)`**（baseM=128 → baseK ≤ 256，所以 `C_in=384/512` 必须切 2–3 段，不能一次吃完一个 tap）；时间维复用只能靠 L2。理想 tile 的稳态是 42,240 cycle 里 41,472 在 MMAD（**98.2% cube 占用**）。

对照我们 F9 的实测：conv 本体 1.246 s / 一次编码（batch 8）= **262 TFLOPS ≈ 74% MFU**，加 conv 内部 TransData 后 ≈60%。**两者不矛盾**：98.2% 是单 tile 稳态上界，我们有 233 次形状各异的 conv 调用（含 `conv_in` C=3、45 次 Conv2D），摊薄到 74% 是合理的。结论维持 F9：**手写 Conv3D 的收益上限只有那 0.275 s/编码（TransData），不值得**；真要动，唯一的低风险入口是逐形状找利用率最差的那些，而不是整体重写。

### 核对 2：数据搬运的账——我之前对 feat_cache 的估计过于乐观

guide 的优先级是「norm+residual → patchify+normalize → downsample 的转置写出」，理由都是「能一次读写解决却写了两次」。按 F9 的逐 kernel 表折算到步时（×4 次编码）：

| 项 | s/步 | 占步时 | 能否消掉 |
|---|---:|---:|---|
| conv 内部 TransData | 1.10 | 3.9% | 否：`channels_last_3d`/预转 fractal 权重两条路都已实测关闭 |
| feat_cache `cat` | 0.53 | 1.9% | **只能省 2/(T+2) ≈ 6%**——新帧本来就要拷一次，常驻缓冲只免掉两个缓存帧的重复拷贝（F9 里「省 2/3」的估计作废） |
| AvgDown 的 permute | 0.44 | 1.5% | 只有 27/61 次属于它；融合成「跨步读 + 顺序 8 元素求均值」可省一次全量拷贝，≈0.19 s/步（0.7%） |
| reflect pad | 0.24 | 0.8% | 可折进内核 store（F11 已给出路径），上限 ~0.2 s/步 |

**合计可动 ≈0.4 s/步（1.4%）**，且 AvgDown 融合会让 `mean` 的归约顺序变成我们自己写的——正是 F8 里「24 种顺序都对不上」的那类风险，收益却只有 0.7%。数据搬运这条路到此为止。

### 核对 3：训练循环侧的新线索——MC2 是唯一还没试过的通算重叠

F10 实测「通信与计算零重叠」，异步 reduce-scatter 也无效。guide §4.1 给出了原因和唯一的出路：910B 上通算重叠**只能靠 MC2**（AI Core 写消息区 → AICPU 调度 → SDMA 搬数），也就是把「matmul + 集合通信」写成**一个融合算子**（`AllGatherMatmul` / `MatmulReduceScatter`），通信由 SDMA 执行、不占 AI Core。我们的路径是 host 侧 `dist.reduce_scatter_tensor`，既不在 AI Core 上、也不在 SDMA 上，**处在两者之间的同步调用**，所以天然无法重叠。

这条的潜在收益是 F10 估的 2–3 s/步（7–10%），但代价是把 ColossalAI 的 ZeRO 梯度归约换成 `npu_mm_all_reduce_base` 一类的 MC2 算子——要改 autograd hook、算子形状契约和 HCCL group 传参，属于周级改动，且一旦数值路径变化就不再逐位可比。

### 另外两条从 guide 里捞到的、与我们直接相关的硬事实

- **`--cce-ftz` 默认开**（guide 反例 5）：视频 VAE 的 subnormal 会被冲成 0。我们的融合 RMSNorm 内核是 fp32 内部计算、最后 `CAST_RINT`，需要确认编译选项没有把 subnormal 抹掉——这条要写进新内核的验收清单（用含 subnormal 的输入对拍）。
- **tilelang-ascend issue #298**：编译器可能把 GM→L1 生成成两跳（GM→UB→L1）。我们的 VAE norm 内核走的是 UB 路径（`T.alloc_ub` + `T.tile.*`），不受这条影响；但如果日后写 conv/gemm 类内核，要先确认没有踩到。

### 候选排序（按「收益 ÷ 风险 × 可验证性」）

| # | 候选 | 预期 | 风险 | 验证方式 | 是否要动设备 |
|---|---|---|---:|---|---|
| 1 | `--dit-norm-kernel`（配置 B，已实现） | −3.3% 步（0.9 s） | 1.2 ppm 元素 1 ulp | 已完成 20 步 + 16 rank 复现 | 重启即可 |
| 2 | MC2 化梯度归约 | 7–10%（2–3 s） | 高，周级，改变数值路径 | 需先做 collectives-only 的可行性探针 | 需独占设备 |
| 3 | 逐形状找利用率差的 conv | ≤3%（未测） | 中 | 先补一次 `kernel_details` 级 profile | 需设备 |
| 4 | AvgDown 融合内核 | 0.7%（0.19 s） | 中（归约顺序） | CPU 等价 + 设备逐位 | 需设备 |
| 5 | VAE norm 的 wave 量化修复（C=640 的两个 shape） | 0.15%（0.04 s） | 低 | 单算子 benchmark | 需设备 |

**结论**：kernel 级优化已经走到「每一项都要先用设备量、且单项 ≤1%」的区间，唯一还够大的是 MC2（2–3 s）。设备被 48 卡训练占着，所以下一步无论是验证绑核、试 MC2 探针还是补 profile，都要先决定这台训练停不停。

## F23：48 卡这一轮的 checkpoint 不可续训（2026-11-15，已用脚本证实）

step 100 的首次存档跑了 **45 s**（17 GB，378 MB/s），本身符合预期。但用
`scripts/check_checkpoint_complete.py` 一验，发现**它不能用来续训**：

| 节点 | 状态 |
|---|---|
| node1（rank 0） | 已发布 `checkpoint-3f9d…`，清单 31 个文件 / 16.9 GiB，但**缺 rng-00008..00047** |
| node5/6/7/8/9 | 只有 `.incomplete-…` 暂存目录，**从未发布**（无 complete.json） |

原因在 `save_checkpoint`/`_publish_generation` 的设计假设：**每个 rank 把分片和 `rng-<rank>.pt` 写进自己的
staging，只有 rank 0 做 rename + 写 complete.json**。共享文件系统下这是对的（所有 rank 写进同一棵树，一次
rename 发布全部）；但我们是**每台节点本地盘各写各的**，于是 rank 0 的清单只覆盖 rank 0-7，其余五台的 staging
永远停在 `.incomplete-`。而 `validate_checkpoint` 要求清单里含 **全部 48 个 rng**，所以**两台机器都过不了校验**。

**影响**：这轮训练目前没有任何可用的重启点，崩溃或重启等于丢掉全部进度。之前「跑到 step 100 落了 checkpoint
再重启，代价小」的说法因此作废。

**两条修法**（都已核实过前提）：

1. **把 checkpoint 树放到共享盘**：`/home`（NFS，607 MB/s 实测）是唯一的跨节点共享存储，但只剩 **131 GB**，
   一代 17 GB，只够 7 代且必须剪枝；其余人的空间也会被挤压。零代码改动（把 `runs/<tag>/checkpoints` 换成
   NFS 目录的软链，所有 rank 即可写进同一棵树），但空间是硬约束。
2. **改成「每台节点自发布」**（改动比看上去大，**不作为首选**）：除了各 rank 自发布 + 放宽校验之外，
   还卡在**权重落点**上——`hybrid_parallel_checkpoint_io.py:303` 只让 `dp_rank == 0` 写 model，
   而 `load_sharded_model`（:426）是**每个 rank 各自从本地路径读全部权重**（ZeRO-2 下模型是复制的）。所以本地盘方案
   还得同时解决「rank 0 的 6.3 GB 权重如何到另外五台」（复制 5 份或改广播），否则校验过了也加载不了。
   这条要动 checkpoint IO 的加载语义，不做没有测试覆盖的改动。

在修好之前，**任何重启都会丢进度**；修好之后每 100 步一次的重启代价才回到「45 s + 一步」。

## F24：六机收尾、四机权重热启动与 node6 重新测量（2026-11-15）

六机（48 rank）在 step 130 左右停下（该轮无可用重启点，见 F23），改由 node5/7/8/9 四台（32 rank）从 step 100 的
**权重**热启动续训；node1/node6 空出来做算子开发（node6 为主力）。

### 权重热启动：先审计再动手

`scripts/prepare_warm_start.py` 做两件事：把训练 checkpoint 的 `model/` 改成 diffusers 约定
（`transformer/diffusion_pytorch_model-0000N-of-0000M.safetensors` + 重写 index + 复用基座的 `config.json`），
以及**逐位审计**：

| 检查 | 结果 |
|---|---|
| key 集合（vs 基座 transformer） | 549 / 549，0 缺失、0 多余、0 shape 不符 |
| dtype | 4 个 `time_embedder.*` 训练态是 bf16、基座是 fp32（训练全程 bf16，属预期） |
| **逐位**比对 | 322 个可训练张量按预期变化，**0 个冻结张量变化** |

热启动目录 `/mnt/disk0/jiangzs/nzf/checkpoints/Cosmos3-Edge-SFT100`（8.6 GB）。两个环境事实值得记：容器里
`/mnt/disk0/jiangzs` 是 **ro 挂载**（所以不能建在基座旁边）；四台之间**没有 SSH 互信**，分发改用 node1 起临时
HTTP + 四台 wget（内网，md5 对拍一致）。

### 启动脚本的两个静默坑

1. `run_6node_train.sh` 只透传 `MASTER_PORT`，**没传 `NNODES`**——四台启动时 step 脚本会按默认 6 节点告诉
   torchrun，然后永远等两台不存在的机器。已补 `-e NNODES=${#NODE_LIST[@]}`。
2. `bench_online_step_6node.sh` 的 `--checkpoint` 是硬编码的，加 `CKPT` 覆盖；顺带支持 `SAVE_EVERY`。
3. 训练必须带 `--no-reinit-action-head`，否则 `build_model` 会把训练过的 action head 重新初始化
   （`train_sft.py:501`）。

起跑后日志确认：`world size: 32`、`trainable 1.423B / frozen 1.946B`、`sum/H instead of mean in 198 norm`，
且没有 action-head 初始化那行。

### 32 卡跑到 step 156 之后：稳态与掉队仍在

| 区间 | 步时 p50 | 全 rank 最大 wait（p50） | bwd spread（p50） | 吞吐 |
|---|---:|---:|---:|---:|
| step 3–139 | 27.67 s | 3.77 s | 2.86 s | 37.0 样本/s |
| step 140–178 | 27.82 s | **4.11 s** | **3.15 s** | 36.8 样本/s |

也就是说 32 卡下掉队暴露仍有约 **4 s（11–12%）**，只是比 48 卡的 4.9 s 小。这条是当前最大的单项可动空间，
而它属于**调度**而不是算子：最慢 rank 的数据就绪时间 1:1 计入步时（F21 的回归斜率 0.99）。

### TransData 1.14 s/步：两条布局路重新验证，都关死

F9 之后又验了一次，因为它是 VAE 里第二大项（11.6%）：

| 尝试 | 结果 |
|---|---|
| `F.conv3d` 接受 NDHWC-strided 输入（`x.permute(0,2,3,4,1)`，内存不变） | 拒绝：conv 把 dim 1 当通道，`expected 160 channels, got 8` |
| `Tensor.to(memory_format=torch.channels_last_3d)` | 拒绝：`Only contiguous_format or preserve_format is supported` |

所以 conv 内部的 NCDHW↔NC1HWC0 转换（1.14 s/步）在这条栈上确实没有入口，只能等 torch_npu 支持 channels-last。

### 32 卡跑起来之后：48 卡那 8% 的税确实是 replica 掉队

同一个 recipe、同一份权重热启动，只把世界大小从 48 改成 32（replica 6 → 4），跑 139 步的稳态统计：

| 口径 | 16 rank | 48 rank | **32 rank** |
|---|---:|---:|---:|
| 步时 p50 | 27.37 s | 28.8 s | **27.67 s** |
| 每卡吞吐 | 1.169 样本/s | 1.07 | **1.157** |
| 全 rank 最大 wait（p50） | — | 4.90 s | **3.77 s** |
| backward spread（p50） | — | 4.63 s | **2.86 s** |

也就是说：**48 卡相对 16 卡的 8% 损失，在 32 卡上基本消失（只差 1%）**，而 VAE/forward 三项在所有规模下完全一致
（9.87 / 2.77 s）。这条把 F21 的归因钉死了——多机税是「replica 数越多、最慢 replica 的暴露等待越长」，不是带宽、
也不是 kernel。

### node6 重新测量（生产配置：fast encoder + 融合 norm，batch 8）

一次编码 5271 个 kernel 的归属：

| kernel | 占 device | 折算到步时（VAE 9.86 s） |
|---|---:|---:|
| Conv3DV2 | 53.3% | 5.25 s |
| conv 内部 TransData | 11.6% | 1.14 s |
| 融合 norm（main_kernel） | 11.1% | 1.09 s |
| Cat（feat_cache） | 5.8% | 0.57 s |
| Transpose | 4.5% | 0.44 s |
| Silu / Add / PadV3 | 3.3 / 3.1 / 2.1% | 0.33 / 0.31 / 0.21 s |
| Conv2D / Slice / ReduceMean | 1.9 / 1.5 / 0.5% | 0.19 / 0.15 / 0.05 s |

与 F9 的结论一致：**可融合的纯搬运合计约 0.96 s/步（3.3%）**，而且分散在 transpose/add/pad 三块；cat+slice
合计 0.72 s 卡在「conv 直接读缓存」这个必须重写 conv 的前提之后。

### 调研带来的三处修正

1. 本模型**没有 AdaLN**（`Cosmos3VLTextMoTDecoderLayer` 无 scale/shift/gate），所以「调制成 norm」的融合不成立；
   DiT 侧唯一有量级的 Nemotron norm 链已被 `--dit-norm-kernel` / `--dit-sum-mean-norm` 覆盖。
2. **910B 无 SIMT**：跨步 gather 的有效带宽只有 0.8–25%（MTE2 256 B 包），guide 的结论是这类算子要改算法而不是
   优化实现——这解释了此前 strided `mean` 慢 1.9×。AvgDown 融合必须走「连续行 DMA + UB 内重排」。
3. `--cce-ftz` 只影响 950；910B 原生支持 subnormal，之前记的担忧不适用于本机型。

## F25：AvgDown 融合——算法已证逐位可行，实现被 TileLang 挡住（2026-11-15）

### 结论先写

`scripts/probe_avgdown_order.py` 把「融合能不能逐位一致」这个问题关掉了：**fp32 累加与参考逐位一致，bf16 累加
不行**。三种配置各 3 轮、0 个元素不同；bf16 累加有 52% 元素不同、最差 1.6e-2。这同时解释了历史上那次
strided `mean(dim=(5,7))` 为什么 relL2 6.7e-2——**它是 bf16 累加的**，不是归约顺序的错。顺带纠正一个细节：
三个 AvgDown 模块的 group_size 分别是 **8 / 4 / 2**（不是统一的 8），对应 `160→160`、`160→320`、`320→640`。

### 实现受阻的四个具体构造（都在 910B2 + 本机 tilelang 上复现过）

| 构造 | 结果 |
|---|---|
| `T.reduce_sum(wide_2d, acc_1d, dim=-1)`，`[Wo, 2]` | 行 0 对、其余错；threads=1/2、带不带 `real_shape` 都一样 |
| `ub[0:W, 0:1]` 列切片到 `[W,1]` | ACL stream error；**行切片 `ub[c, 0:chunk]` 是好的**（融合 norm 内核就靠它） |
| `ub[0:W:2]` 步长切片 | UB 侧报错；**GM 侧 `x[0:W:2]` 被静默忽略**（拷了前 K 个连续元素）——四种里最危险的一种 |
| `T.tile.gather` + uint32 字节偏移 | 报错（本 build 的偏移/基址约定没有文档） |

没有跨步访问，pair 轴就只能靠 Python 侧 permute 去交错——而那正是这个内核要省掉的拷贝；或者 ~1e8 次标量操作。
所以 `vae_avgdown_kernel.py` 保留（算法与设计都在），但入口改成 `enable_avgdown_kernel_blocked()` 直接抛
`NotImplementedError`，模块顶部写明 STATUS: BLOCKED，**不接入训练**。

### 顺带做掉并测掉的一项：pad 在 uint8 上先做

`video_pad_norm.pad_normalize` 原来是「先归一化成 bf16，再 flip+cat 补边」，改成「先在 uint8 上补边，再归一化」，
并用 `_block_for(m)` 选一个能整除的块大小（补边后张量不再是 4096 的整数倍，原来的固定 BLOCK 会静默退回 torch 路径）。

| pad | 逐位一致 | 旧 | 新 |
|---|---|---|---|
| (4, 16)（生产） | ✅ | 22.69 ms | **19.65 ms（−13.4%）** |
| (3, 7) | ✅ | 16.39 ms | 16.21 ms（−1.1%） |

**但要说清楚量级**：这条路径整步只占 ~88 ms（0.3%），省 13% 就是 **0.04% 的步时**——它的真实价值是消掉了
「补边后 shape 不整除就静默退回 torch」这个坑，不是性能。

### 这一步对「kernel 级还能拿多少」的最终判断

- VAE 里可动的纯搬运合计 ~0.96 s/步（3.3%），分散在 transpose 0.44 / add 0.31 / pad 0.21，**最大那块被工具链挡住**；
- conv 逐形状实测 223 / 236 / 293 TFLOPS（峰值 353.9），最差的 1×1×1 `conv_shortcut` 只有 37.4、`conv_in` 74.7，
  但两者合计仅 0.19 s/步——**手写 conv 仍然不划算**；
- 因此 kernel 级的下一个真正有量级的杠杆不在 VAE，而在训练循环的通算重叠（F10 的 2–3 s/步）和 host 侧等待
  （17%），两者都需要改架构层面而非再写一个融合内核。

## F26：训练循环的未放大归属（node6，8 卡，3 步，--profile-step 2）

`analyze_step_kernels.py` 对 `profstep2` 的 profile 汇总（41627 个 kernel）。VAE 侧的 kernel 与 F24 的
9.86 s 口径逐项吻合（conv 4.98 / TransData 1.10 / main_kernel 1.05 / Cat 0.60 / Transpose 0.45 / Silu 0.31 /
Pad 0.23 / Conv2D 0.18），所以这批数字可以直接当设备时间用；训练循环侧仍受 profiler 放大影响。

| 类别 | 调用 | 设备时间 | 占比 |
|---|---:|---:|---:|
| communication | 501 | 10.64 s | 33.1% |
| elementwise | 25638 | 8.33 s | 25.9% |
| conv | 6532 | 6.28 s | 19.5% |
| matmul | 1243 | 3.54 s | 11.0% |
| other | 4348 | 1.33 s | 4.1% |
| **我们的融合 norm** | 1064 | 1.05 s | 3.3% |
| activation | 938 | 0.64 s | 2.0% |
| reduce | 918 | 0.34 s | 1.0% |

单个 kernel 的前几名：Conv3DV2 4.98（15.5%）、MatMulV3 3.48（10.8%）、**Add 2.01（5386 次）**、
**Mul 1.68（3074 次）**、FA-grad 1.28、FA-fwd 1.13、TransData 1.10、main_kernel 1.05、
reduceScatter 0.75（单次）、**Cast 0.62（2088 次）**、Cat 0.60、Transpose 0.45、**Pow 0.43（770 次）**、
Silu 0.31、ReduceSum 0.29、PadV3 0.23。

三条由此确定的判断：

1. **Pow 的 770 次全部来自 DiT 的 198 个 norm**（`h.pow(2)`，fwd + 重算 + bwd 各一遍），不是 loss——
   F12 把 loss 的 `.pow(2)` 换成 `.square()` 是对的，但那是另一处。改 norm 的 `pow` 属于 config B 的范畴
   （`npu_rms_norm` 内部完成），单改 `pow→square` 预计与 F12 类似、步时无感（kernel 时间 ≠ 墙钟）。
2. **Cast 0.62 s 与 Add/Mul 3.7 s 主要是 DiT 的 norm 链与残差**：norm 链已由 `--dit-norm-kernel`（config B）
   覆盖（−3.3% 步时），残差/RoPE/ReLU² 的可融合量按 token 体积估 <1%，与这里的数字不矛盾（这里的秒数被放大）。
3. **communication 33% 是唯一还够大的项**（F10 已证明它与计算零重叠，且异步无效）——出路只有 MC2
   （matmul + 集合通信写成融合算子，通信走 SDMA 不占 AI Core），属于架构级改动，不是再写一个内核。

### 所以 kernel 级这条线的终点

| 项 | 状态 |
|---|---|
| VAE conv 5.25 s/步 | 逐形状 223–293 TFLOPS（峰值 353.9），手写不划算 |
| VAE TransData 1.14 s/步 | 两条布局路（NDHWC-strided 输入、`channels_last_3d`）都被 torch_npu 拒绝 |
| VAE 融合 norm 1.09 s/步 | 我们的内核，已按 756 GB/s 调过 |
| VAE 数据搬运 2.3 s/步 | AvgDown 那块被 TileLang 的跨步/二维归约限制挡住（F25） |
| DiT elementwise | norm 链已有 config B；其余碎项 <1% |
| 通信 33% | 只有 MC2 能动 |

## F27：绑核 A/B 在真机上成立（node6，8 卡，各 5 步）

`NUMA_PIN` 的整条链路（`npu-smi` → sysfs → socket cpuset → `os.sched_setaffinity`）在真实训练启动路径里验证通过：

```
NUMA_PIN numa_w8 rank=0 map=144-191;144-191;96-143;96-143;0-47;0-47;48-95;48-95
[numa] local rank 0 pinned to cpus 144-191 (48 cores)
```

两个臂除绑核外完全相同（同 recipe、同 2048 episodes、5 步、同一台机器先后跑）：

| 臂 | 步时 p50 | VAE | forward | backward | wait（中位） | wait（全 rank 最大） |
|---|---:|---:|---:|---:|---:|---:|
| 不绑核 | 26.69 s | 9.92 | 2.79 | 10.38 | 1.82 s | **2.50 s** |
| **绑 socket** | **26.14 s** | 9.88 | 2.78 | 10.12 | 1.51 s | **1.86 s** |

**步时 −0.55 s（−2.1%），而且落点正是预测的那一处**：VAE/forward 完全不变（设备工作没变），
**wait 中位 −0.31 s、全 rank 最大 −0.64 s**，backward 也跟着降 0.26 s（CPU 争用变小）。
与先验测量一致：未绑核时单进程每样本 8.5 核秒、绑到 socket 后 7.5（−12%），node5 未绑核时比 node1 多花 9–17% CPU。

**置信度说明**：每臂只有 4 个稳态步，步时差的 0.55 s 刚过 ±0.4 s 的噪声底；但机制（wait 下降、vae/fwd 不动）在
32 个 rank-step 上一致，方向可信。下一步是在四机（replica 4、max-wait 4.11 s）上验证——那里要等的正是最慢
节点，收益预期比单机更大，但需要把生产训练停一次。

### 四机验证的可行路径（等用户点头再动）

绑核在单机已经证明能把 wait 从 1.82 压到 1.51（全 rank 最大 2.50 → 1.86）。四机上要等的是**最慢节点**，
预期收益更大：32 卡当前的 max-wait 是 4.11 s，而 node5 一直比 node7 慢 0.85 s。

代价可以做到很小，因为**「干净停止」会写一代 checkpoint**（`control` 机制），而 F23 的缺陷只影响 `--resume`，
不影响**权重热启动**（F24 已验证这条路）。所以流程是：

1. 通过 run-control 干净停止四机训练 → rank 0 写出 `checkpoint-<id>/model`（约 45 s）；
2. 用 `prepare_warm_start.py` 把它组装成新的热启动目录（~1 min）；
3. 在两个调度配置上各跑 5 步（`--hsdp-shard-size 8` vs `32`，都带 `NUMA_PIN=1`）；
4. 用胜者热启动恢复生产训练，丢的只是停止前最后一次保存之后的步数。

两个待答问题：**(a)** 绑核在 replica 4 下是否把 4.11 s 的 max-wait 压下来；**(b)** 纯 ZeRO（shard 32）去掉
replica all-reduce 后，暴露等待是否比 shard 8 更小——F10/F14 在 16 卡上测过是平手，但那时只有 2 个 replica。

### 三个调度臂的对比（node6，8 卡，各 5 步，同一 recipe）

| 臂 | 步时 p50 | VAE | forward | backward | wait 中位 |
|---|---:|---:|---:|---:|---:|
| A 不绑核 + OMP4 | 26.69 s | 9.92 | 2.79 | 10.38 | 1.82 s |
| B 绑核 + OMP4 | 26.14 s | 9.88 | 2.78 | 10.12 | 1.51 s |
| **D 绑核 + OMP6** | **25.64 s** | 9.87 | 2.77 | **9.75 s** | 1.48 s |

两处改动**作用在不同阶段、可以叠加**：

- 绑核 → **wait** 1.82 → 1.51（数据侧 CPU 效率），backward 顺带 −0.26；
- OMP4 → OMP6 → **backward** 10.12 → 9.75（计算阶段的主机侧争用变小），wait 基本不动；
- 合起来 **−1.05 s/步（−3.9%）**，而 VAE/forward 三项全程不动，说明动的都是主机侧而不是设备工作。

注意这与 F16 的旧结论不冲突：OMP8 当年是在**没有绑核**的情况下把 192 核占满导致周期性停摆；绑核之后每个 rank 只在自己
那 48 核里，OMP 上限的约束变了，所以 OMP6 反而更好。OMP8 的臂还在跑，用来找拐点。

### 调度臂 C 否掉：8 worker × 2 线程触到「每设备进程数上限」

`--num-workers 8 --worker-threads 2`（同样 16 个线程、更细粒度）在第一个 step 就挂：

```
[PID 127915] Not_Supported_Specification(E30003): The number of started user processes on the device
exceeds the limit.
RuntimeError: DataLoader worker (pid 127915) is killed by signal: Aborted.
```

worker 里只要有一次 NPU 上下文初始化（inductor 的 hue 图或任何 tensor 操作），就会占用该设备的进程名额；
8 rank × 8 worker = 每设备最多 9 个上下文，超过上限。**4 worker/rank（= 5 上下文/设备）是安全档位**，
这也解释了为什么更细的流水线粒度在这条栈上走不通——是硬约束，不是调参问题。

## F28：把调度收益搬到四机生产（干净停止 → 热启动 → 绑核 + OMP6）

### 先彩排，再动生产

「停止 → 存档 → 热启动」这条往返在 node6 上用一次性跑完整验证过：

| 步骤 | 结果 |
|---|---|
| 写 `control/request.json {"action":"stop"}` | trainer 在步边界看到：`phase=stopping, step=3, stop_at_step=3` |
| 干净退出并发布 checkpoint | `checkpoint-5772f3a4…`，31 个文件 / 16.9 GiB，model 8 分片 + optimizer 14 + 8 个 rng，`check_checkpoint_complete` 判 **RESUME-SAFE** |
| `prepare_warm_start.py` 重组 | 549 个张量覆盖基座，7 个分片 |
| 从它热启动跑 3 步 | `trainable 1.423B / frozen 1.946B`，step 1 loss 8.74（权重确实被加载，不是全新起点） |

顺带记一个分发坑：容器（root）组装的目录是 `root:root drwxr-x---`，宿主机起的 HTTP 服务读不了，返回 404；
`chmod -R a+rX` 之后才通。node1 上组装的同款目录属主是 niuzifan，所以上一轮没暴露这个问题。

### 切换结果：−6.7% 步时，max-wait 从 3.85 降到 2.42

同一采样位置（step 3–8，两边都是刚冷启动之后的头几步，排除 step 2 那个「索引后整批数据在缓存里」的特快步）：

| | 步时 | VAE | forward | backward | max-wait |
|---|---:|---:|---:|---:|---:|
| 旧：未绑核 + OMP4 | 27.92 s | 9.87 | 2.78 | 11.37 | **3.85 s** |
| **新：绑核 + OMP6** | **26.06 s** | 9.78 | 2.77 | **10.25 s** | **2.42 s** |

**−1.86 s（−6.7%）**，比单机四臂的 −3.9% 更大——因为四机要等最慢的节点，绑核把每个节点都往前推。

把窗口拉长，用**同一采样位置**逐段对比（旧跑在新跑同位置也会漂，所以不能拿旧跑的总中位数当基线）：

| 窗口 | 旧步时 | 新步时 | 差 | 旧 max-wait | 新 max-wait |
|---|---:|---:|---:|---:|---:|
| step 3–9（7 步） | 28.02 | 26.27 | −1.74（−6.2%） | 3.97 | 2.57 |
| step 10–20（11 步） | 28.65 | 27.16 | −1.49（−5.2%） | 4.34 | 2.05 |
| step 21–24（4 步） | 28.61 | 27.46 | −1.15（−4.0%） | — | — |

**收益稳定在 −1.1~−1.7 s（−4.0%~−6.2%）**，且不随步数衰减——不是头几步的缓存假象。backward 的 spread 也从
3.79 s 降到 2.68 s（bwd 最小值 9.45 → 9.50 不动，说明纯计算没变，降的是等最慢 replica 的那段）。

**规模效率回到 100%**：新配置下 32 卡每卡 1.168 样本/s，与 16 卡的 1.169 持平（48 卡未绑核时是 1.07，
32 卡未绑核时 1.157）。多机税基本被调度吃掉了。

另外，**半 socket 绑核（每 rank 独占 24 核）不如全 socket**：node6 上 25.84 s vs 全 socket 的 25.64 s，
在噪声内但方向偏慢——两个 rank 共享 48 核比各占 24 核更好，说明这条流水线要的是容量而不是更细的局部性。
VAE 与 forward 保持不动（9.78 / 2.77），说明变量干净，动的全是主机侧。

逐节点 wait（中位 / p90）进一步印证机制：

| 节点 | 旧 | 新 |
|---|---|---|
| **node5（4 NUMA 域，长期最慢）** | **2.58 / 4.56** | **1.77 / 2.11** |
| node7 | 1.85 / 2.43 | 1.61 / 2.29 |
| node8 | 1.95 / 3.24 | 1.52 / 1.72 |
| node9 | 2.30 / 3.43 | 1.66 / 2.35 |

node5 的 **p90 wait 从 4.56 s 降到 2.11 s**，节点间 wait 极差从 0.73 s 收到 0.25 s——这正是「绑到 NPU 所在
socket」应该产生的效果：4 NUMA 域的机器上两个 rank 共享一个 socket，不绑定时线程和内存都会跨域。

### 四机切换（step 263）

1. 四台各写 stop → 训练在 step 263 干净停止，node5 发布 `checkpoint-9ac7d28e…`（17 GB）；
2. node5 组装 `Cosmos3-Edge-SFT263`（549 张量）→ HTTP 分发到 node7/8/9，四台 md5 一致（`d6eb4d9a`）、各 8.6 GB；
3. 以 `NUMA_PIN=1 OMP_THREADS=6` 重起 32 卡（tag `cosmos3_4node_numa`），其余参数与上一轮完全相同。

对照组是同一份 recipe、同一批机器上 step 140–178 的稳态：步时 27.82 s、max-wait 4.11 s、bwd spread 3.15 s。
验收标准：**步时 ≥ 0.5 s 的下降，且 VAE/forward 不动**；若 max-wait 反升，则说明四机的瓶颈不在数据侧，
绑核要退回、只保留 OMP。

## F29：shard 8 vs 纯 ZeRO（32）——步时打平，纯 ZeRO 省 2.9 GiB

同一份 step-27 热启动、同样的绑核 + OMP6，只把 `--hsdp-shard-size` 从 8 换成 32（replica 4 → 无 replica 组），
同采样位置 step 3–8 对比：

| | 步时 | VAE | forward | backward | optim | max-wait | bwd spread | device used |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **shard 8（保留）** | **26.06 s** | 9.78 | 2.77 | 10.25 | **0.33** | **2.42** | **1.24** | 54.96 GiB |
| shard 32 | 26.20 s | 9.77 | 2.78 | 10.39 | 0.49 | 2.69 | 1.43 | **52.02 GiB** |

**步时打平（shard 32 反而慢 0.14 s，在噪声内），但两处结构性差异值得记：**

- **optim 0.33 → 0.49 s**：纯 ZeRO 下优化器步要跨 32 个 rank 做参数 all-gather，代价可测地变大；
- **显存 54.96 → 52.02 GiB（−2.94 GiB）**：优化器状态与梯度的分片细 4 倍，省下来的显存是实打实的。

结论：**时间上保持 shard 8**（本轮目标是步时），shard 32 作为「显存换时间」的备选记在这里——如果以后要提
batch 或开更大的 selective checkpointing 预算，这 2.9 GiB 就是可用余量。

### 顺带修掉一个启动器误判

`run_6node_train.sh` 的空闲检查用 `pgrep -f "[t]rain_sft.py"`，而上一轮遗留的 collector 命令行里有
`--entry train_sft.py`，于是四台都被误判为忙、拒绝启动。已改成先取 PID 再用 `ps` 过滤掉 `cosmos3_monitor` 的行。

## F30：MC2 通算融合的可行性评估——结论是不值得做（2026-11-15）

torch_npu 2.13 确实暴露了 MC2 算子：`npu_mm_all_reduce_base`、`npu_mm_reduce_scatter_base`（以及量化变体），
所以工具是有的。但把它用到我们的梯度归约上有两个硬前提：

1. **必须把反向的 matmul 换成 MC2 算子**：MC2 是「先算 A·B 再集合通信」的融合体，它要的是矩阵乘的输入；
   而 ZeRO 的 `_run_reduction()` 拿到的是已经累加好的 `.grad` 桶，两者对不上。要接就得改每一层 linear 的
   backward（autograd Function 或 shardformer policy），并且让 ZeRO 的桶逻辑改成「MC2 直接产出分片梯度」。
2. **数值路径会变**：归约顺序与「先累加到 .grad 再 reduce」不同，逐位一致性会丢失。

**收益上限已经被前面的调度工作压掉了**：F10 时代暴露的通信是 2–3 s/步，那时 bwd spread 是 3.79 s；
绑核 + OMP6 之后 **bwd spread 只剩 1.24 s**（bwd 最小值 9.50 不动，说明降的是等待段）。MC2 能动的就是这 1.24 s，
即 **≤4.5%**，而且要付周级改动 + 数值路径改变两个代价。

**结论：MC2 不再值得做。** 调度把它的上限从 7–10% 压到了 4.5% 以下，风险收益比反过来了。

### AvgDown 的第二条路（torch 两段式）也是负结果

「先对 (t,h) 求和、再对 w 对求均值」的两段式可以不碰 permute：

```python
v = x.view(b, c, t//ft, ft, h//fs, fs, w//fs, fs)
total = v.float().sum(dim=(3, 5)).mean(dim=-1) / (ft * fs)     # 全部 fp32，和是精确的
```

数学上与参考等价（bf16 的 8 个数在 fp32 里求和是精确的，所以任何顺序都逐位一致——这也解释了
`probe_avgdown_order.py` 里 seq/pair/sum_div 全部相等）。但实测在 160→160 的最大那档上：

| | 时间 |
|---|---:|
| 参考（permute + mean） | 1.25 ms |
| 两段式 | **1.65 ms（+32%）** |

**更慢**，而且 160→320 那档需要另一套分组（group_size=4 时 (a) 轴参与输出通道划分，两段式会把它消掉）。
这条也关掉：AvgDown 的 0.44 s/步在「TileLang 内核受阻 + torch 两段式更慢」之后，确认拿不到。

### 越往后收益越大：step 150–170（21 步）−9.8%

同采样位置、连续 21 步的对照：

| 窗口 | 旧步时 | 新步时 | 差 | 旧 max-wait | 新 max-wait |
|---|---:|---:|---:|---:|---:|
| step 45–60（16 步） | 28.34 | 27.05 | −1.28（−4.5%） | 4.36 | 1.82 |
| **step 150–170（21 步）** | **28.31** | **25.55** | **−2.76（−9.8%）** | **4.50** | **1.91** |

**收益随训练推进而变大**：采样器越往后走、数据越冷，未绑核那一版的 wait 从 4.36 涨到 4.50（数据侧 CPU 效率
一差，冷读的代价就被放大），而绑核 + OMP6 的版本稳在 1.8–1.9。也就是说调度改动不只是把基线平移，**它改变了
斜率**——长跑里这一点比头几步的数字更重要。

### 最干净的一次对照：step 45–60（16 步）

`cosmos3_4node_final`（绑核 + OMP6）与旧的未绑核跑在同采样位置、连续 16 步的对比：

| | 旧 | 新 | 差 |
|---|---:|---:|---:|
| 步时 | 28.34 s | **27.05 s** | **−1.28 s（−4.5%）** |
| backward | 11.95 | 11.61 | −0.34 |
| 全 rank 最大 wait | **4.36** | **1.82** | **−2.54** |
| VAE | 9.87 | 9.77 | −0.10 |

样本最长的一次，结论与前面两个窗口（−5.2%、−6.2%）一致：**调度收益稳定在 −4.5%~−6%，取 −5% 作为口径**。
注意 wait 降了 2.54 s 而步时只降 1.28 s——设备侧的积压吸收了一半，这也解释了为什么单看 wait 会高估收益。

### 回退后的一过性现象（记下来避免误判）

从 shard 32 切回 shard 8 时，新跑（`cosmos3_4node_final`）的 step 4–9 出现 backward 比 `cosmos3_4node_numa`
同期高 **+2.3 s**（12.4–13.8 vs 10.2–11.2），而 **wait 反而更低、VAE/forward 完全一致**，显存也一样（54.95 vs 55.09 GiB，
排除第二个作业）。到 step 10 之后 backward 自己收敛回 11.1–11.8，与 numa 跑 step 10–20 的 11.62 一致。

结论：这是**连续切换导致的 page cache 抖动**（shard 32 那 8 步把数据读到别的位置，紧接着的新跑前几步读的是冷数据，
体现在 backward 里的集合等待上），不是配置退化。判读多机数据时，**重启后的前 8–10 步不能当稳态**——这已经是本次
第二次被它误导（第一次是 step 2 的「24.5 s」）。

## F31：config B 在当前配置下的实测（node6，绑核 + OMP6，5 步）

在已经上了绑核 + OMP6 的基础上再加 `--dit-norm-kernel`（厂商 `npu_rms_norm` 替换 Nemotron norm 链）：

| 臂 | 步时 p50 | forward | backward | VAE |
|---|---:|---:|---:|---:|
| 绑核 + OMP6（当前生产） | 25.72 s | 2.80 | 10.12 | 9.88 |
| **+ config B** | **24.75 s** | **2.31** | 10.10 | 9.90 |

**−0.97 s（−3.8%）**，全部落在 forward（2.80 → 2.31，−0.49 s），与历史测量（−3.3%）一致，而且与调度收益
正交、可以叠加：调度 −4.5~−6% + config B −3.8% ≈ **合计 −8%~−9%**。

代价仍是精度策略问题：前向 1.2 ppm 的元素偏 1 ulp（真实激活 8.26 亿元素里 95,740 个不同，单次最坏 relL2 2.5e-5），
反向与优化器路径逐位一致，同 seed 20 步不发散、16 rank 复现过。**默认仍不开启**，等用户按「上游精度对齐」的
口径拍板。

### 臂 F：关掉 empty_cache 守卫——中性，保留

`COSMOS3_EMPTY_CACHE_GIB=1`（空闲 3.6 GiB > 1，守卫不再触发），其余同生产配置：

| 臂 | step 2–5 中位 |
|---|---:|
| 守卫开（生产） | 25.72 s |
| 守卫关 | 25.88 s |

**+0.16 s，噪声内，5 步内也没有 OOM**。也就是说那次 `torch.npu.empty_cache()` 的代价量不出来，而它挡的是
「每步 +2.1 GiB 爬升」那种风险——**保留**，不值得为 0 收益去掉保护。


## F32：绕开 TileLang 四构造的尝试——逐位一致做到了，但更慢 73%

F25 说四个构造（二维 `reduce_sum`、列切片、步长切片、`tile.gather`）都卡在「在核内拆开最内层 w 对」。绕法是
把这件事交给 torch：内核只用连续行读把组内 (t,h) 成员求和、写出 **fp32 全宽**行，torch 再对最内层连续的 w 对
做 `.mean(-1)`（该轴 stride=1，不需要 permute）。

**逐位一致性：做到了。** 160→160（group=8）在 5709 万个元素上 `torch.equal` 全等。道理是 bf16→fp32 精确、
至多 8 个 bf16 值在 fp32 里求和也精确，所以加法顺序不可能改变结果——这也解释了 `probe_avgdown_order.py` 里
seq / pairwise / ×(1/n) 三种写法全部相等。

**但速度不行**（生产形状，batch 8）：

| 配置 | 参考（permute+mean） | 内核 + torch 收尾 | 差 |
|---|---:|---:|---:|
| 160→160（group 8） | 10.01 ms | 17.30 ms | **+72.9%** |
| 160→320（group 4） | 10.13 | 25.51 | +151.9%（且我的通道解码有误，全错） |
| 320→640（group 2） | 2.86 | 14.46 | +405.8%（同上） |

原因不在算法而在 DMA 形态：参考里的 permute 是**一次大块连续拷贝**，非常 DMA 友好；而内核每个 block 要按
「隔行」的粒度搬 656 元素的小段，一次一行，setup 开销吃掉一切。**结论：那次 permute 本身不是瓶颈**，
0.19 s/步（AvgDown 占 transpose 的 27/61）拿不回来，AvgDown 这条路彻底关闭。

（group 4/2 的解码错误没有继续修，因为即使修对也仍是更慢，不值得。）

### prefetch 深度在绑核 + OMP6 下重测：仍然更慢

当年否掉深队列（`--prefetch-factor 6`）是在未绑核 + OMP4、192 核占满的背景下。现在每 rank 绑在自己的 socket、
OMP6，CPU/内存分布完全不同，所以重测了 `--prefetch-factor 4`：

| 臂 | step 2–5 中位 |
|---|---:|
| prefetch 默认 2（生产） | 25.72 s |
| prefetch 4 | **26.39 s（+0.67）** |

结论与当年一致：**深队列更差**。原因不是 CPU 预算，而是预取队列把更多样本摁在主机内存里、挤掉 page cache，
反而让后续读盘变冷。这条彻底关闭。


### num-workers 6 × 2 线程：中性，不采用

上次 8 worker 撞在每设备进程上限（E30003）。6 worker + 1 rank = 7 个上下文/设备，能跑：

| 臂 | step | wait | bwd |
|---|---:|---:|---:|
| 生产（4 worker × 4 线程） | 25.64 | 1.48 | 9.75 |
| 6 worker × 2 线程 | 25.43 | 1.43 | 9.80 |

步时 −0.21 s 在噪声内，而 **wait 只动了 0.05 s**——如果真是数据流水线变宽带来的收益，wait 应该有可见下降。
所以这条算中性，不采用（worker 越少、设备上下文越省，也更稳）。

### 调度侧的完整地图（全部实测）

| 项 | 结论 |
|---|---|
| 绑核到 NPU 所在 socket + OMP6 | **采用**：−5%~−10%（随后期推进放大），每卡 1.245 样本/s |
| 半 socket 绑核 | 更差（容量 > 局部性） |
| shard 8 vs 纯 ZeRO 32 | 时间打平；32 省 2.94 GiB 但 optimizer +0.16 s |
| prefetch 4 / 6 | 更差（挤 page cache，与 CPU 预算无关） |
| num-workers 8 | 硬件上限（每设备进程数） |
| num-workers 6 × 2 | 中性 |
| empty_cache 守卫 | 中性，保留 |
| 监控 collector（5 s 间隔） | 中性，保留 |
| MC2 通算融合 | 上限已被压到 1.24 s（≤4.5%），架构级 + 改数值，不做 |
| config B（`--dit-norm-kernel`） | −3.8%，等精度取舍拍板 |

### 最终确认：step 230–268（34 步）−8.2%

| 窗口 | 旧步时 | 新步时 | 差 | 旧 max-wait | 新 max-wait |
|---|---:|---:|---:|---:|---:|
| step 150–170（21 步） | 28.31 | 25.55 | −2.76（−9.8%） | 4.50 | 1.91 |
| **step 230–268（34 步）** | **27.98** | **25.68** | **−2.30（−8.2%）** | **4.33** | **1.90** |

34 步的同位置对照是本次样本最大的一次，收益稳定在 **−8%~−10%**，折合 **39.9 样本/s、每卡 1.247 样本/s**，
高于此前任何配置（16 卡最好 1.169、48 卡 1.07）。

### 结项：还需要用户决定的一件事

kernel 与调度两侧的可动项已经全部实测完毕（F25–F32），唯一未采用的是 **config B**（`--dit-norm-kernel`，
−3.8%，前向 1.2 ppm 元素偏 1 ulp、反向与优化器逐位一致、20 步稳定、16 rank 复现）。它不违反精度约束与否，
是项目口径问题——在用户明确之前不启用，默认运行保持严格逐位对齐。

## F33：图编译这条路在本栈上不通（调研 + 实测）

用户要求「不考虑有偏差的选项，从其他角度继续」。第一个新角度是**图编译/算子融合**（inductor、torchair），
理论上能把 DiT 里那条 elementwise 长链（profile 里约 3.4 s/步）融掉、并省掉逐 op 的派发开销。实测：

| 后端 | 结果 |
|---|---|
| `inductor`（默认） | `InductorError: Device npu not supported` |
| `inductor` + `npu_backend="mlir"` | `ImportError: torch_mlir is not installed` |
| `inductor` + `npu_backend="ascendc"` | `TritonMissing: Cannot find a working triton installation` |
| **`torchair`（GE 图模式）** | **可用且逐位一致**（`torch.equal` 通过），但在 5 个 elementwise 串联、[8192,2048] 的测例上 **0.36 ms vs eager 0.19 ms——更慢** |

也就是说：这个容器里既没有 triton-ascend 也没有 torch_mlir，inductor 三条路都缺依赖；torchair 能跑但小图上是
负收益（GE 的固定派发开销 > 逐 op 异步派发的开销）。**图编译这条线关闭**，`--compile-blocks` 当年那个
「trace 里有函数内 import」的失败也不必再修。

## F34：`pow(2)` → `square()`——逐位一致，白拿一个 kernel 家族

profile 里 `aclnnPowTensorScalar` 每步 770 次（约 0.43 s，放大后），全部来自 198 个 DiT norm 的 `h.pow(2)`。
先验证两种写法在设备上是否逐位相同（`probe_pow_square.py`，bf16 与 fp32，混合 1e-3/1e3 量级）：

```
bf16  pow==square True  pow==mul True   bad=0
fp32  pow==square True  pow==mul True   bad=0
```

**完全一致**，所以把 `_sum_mean_forward` 里的 `h.pow(2)` 换成 `h.square()` 是无损的，而 `square` 走乘法路径、
不进超越函数。改完用 `probe_sum_mean_norm.py` 复验：前向 0/100,663,296 不一致、x 梯度 0、w 梯度一致。

### square 改动的步时效果：中性

| 臂 | step 2–5 中位 | forward | backward |
|---|---:|---:|---:|
| 原样（`pow(2)`） | 25.72 s | 2.80 | 10.12 |
| **改 `square()`** | 25.90 s | 2.81 | 10.42 |

**墙钟中性（+0.18 s，噪声内）**，又一次印证 F12 的教训：把 770 个 Pow kernel 换成 Mul（kernel 时间确实更低）
并没有改变步时——那些 kernel 不在关键路径上。改动保留，因为它逐位一致、单指令、并且让 profile 里少一个
`aclnnPowTensorScalar` 家族；但不计入性能收益。

## F35：融合 add+RMSNorm——厂商算子有，但融合并不比「add + rms_norm」快

用户允许把精度边界放宽到 bf16 噪声量级，于是先查了 torch_npu 的融合算子，`npu_add_rms_norm` 确实存在：

```python
y, rstd, residual = torch_npu.npu_add_rms_norm(x1, x2, gamma, eps)   # gamma 必须与 x 同 dtype
```

- **residual 与 `x1 + x2` 逐位相等**；
- y 与「bf16 加 → fp32 归一化」的参考差在 **33.9% 元素上、max rel 1.3e-2**——正好是 bf16 噪声量级（bf16 eps 7.8e-3），
  符合新口径；
- 但**速度上融合没有优势**（[8192, 2048]）：

| 实现 | 时间 |
|---|---:|
| 分解链（cast/square/sum/div/add/rsqrt/mul/mul/cast） | 0.422 ms |
| `add` + `npu_rms_norm`（两算子） | **0.131 ms** |
| `npu_add_rms_norm`（融合） | 0.142 ms |

融合版比两算子版还慢 0.011 ms。也就是说**再写一个 TileLang 的 add+norm 内核没有意义**——它最多只能追平厂商的
单算子，而厂商已经做了。同时 `T.reduce_sum` 在宽 tile（[64,2048]、[16,512]、[8,128]）上全部失败，
自研的行归约只能靠树形 slice 相加，成本高于收益。

**结论：直接上 config B（`--dit-norm-kernel`，用 `npu_rms_norm` 替换整条 norm 链），不做 add 融合。**

## F36：配置 B 上线 + 配方层的第一个判断（并行策略）

用户允许 bf16 噪声量级的偏差后，把 config B（`--dit-norm-kernel`）接进四机生产：干净停在 step 404 → 热启动
（SFT404，四台 md5 一致）→ 以 `--no-reinit-action-head --dit-norm-kernel` 重起（tag `cosmos3_4node_ditk`），
绑核 + OMP6 保持不变。进程命令行确认两个开关都在。

### 序列并行（`--sp-size 2`）先算了一笔账，不做

策略层支持 Ulysses 上下文并行（`all_to_all`），但对我们这条流水线是负收益：

- SP=2 时 DP 变成 16，每步全局 batch = 16×32 = **512**（现在是 1024），而**每个 rank 的 VAE 仍是 32 个样本**；
- 所以 VAE 段 9.8 s 不动、DiT 段砍半（fwd+bwd ≈ 12.8 → 6.4），步时约 16–17 s 但只出 512 样本 → **约 30 样本/s**，
  比现在的 40 样本/s 差得多；
- 想用「samples-per-rank 翻倍」补回来，VAE 会跟着翻倍（19.6 s），更差。

**SP 是省显存的工具，不是提吞吐的工具**——在 VAE 占 38% 的流水线里，把 DiT 切小只会让 VAE 的占比更高。

### 叠加后的最终效果（同采样位置 step 3–8，四机 32 卡）

| 配置 | 步时 | forward | backward | max-wait | 显存 | 吞吐 |
|---|---:|---:|---:|---:|---:|---:|
| 起点：未绑核 + OMP4 | 27.92 s | 2.78 | 11.37 | 3.85 | 55.5 GiB | 36.7 样本/s |
| **+ 绑核 + OMP6**（首次上线那轮） | 26.06 s | 2.77 | 10.25 | 2.42 | 55.0 | 39.3 |
| **+ config B**（当前生产） | **25.16 s** | **2.26** | **9.89** | 2.46 | **50.4** | **40.7** |

拆开看两项的贡献与叠加性：

- **调度（绑核 + OMP6）**：27.92 → 26.06，**−1.86 s（−6.7%）**，全部落在 wait（3.85 → 2.42）与 backward；
- **config B**：26.06 → 25.16，**−0.90 s（−3.5%）**，全部落在 forward（2.77 → 2.26），与调度的作用面不重叠；
- **合计 −2.76 s（−9.9%）**，接近线性相加；显存同时从 55.5 降到 50.4 GiB（config B 省 5.1 GiB）。

口径提醒：这里没有用 `cosmos3_4node_final` 的 step 3–8 做对照，因为那一轮紧接在 shard-32 实验之后、前 8 步带着
page-cache 抖动（F28 已记录），用它会高估 config B 的收益。

### 配方层：放大 batch 也是负结果

node6 上 `--samples-per-rank 40`（其余同生产：绑核 + OMP6 + config B）：

| 配置 | 步时 | 每节点吞吐 |
|---|---:|---:|
| 32 样本/rank | 24.75 s | **10.34 样本/s** |
| 40 样本/rank | 31.82 s | 10.06 样本/s（−2.7%） |

步时涨了 28.6%，比 batch 的 25% 还多——VAE 与 DiT 都随样本线性增长，而更大的工作集把 page cache 和显存余量
又吃回去了。**保持 32 样本/rank。**

配方层到这里的两条结论：序列并行会把全局 batch 砍半而 VAE 不变（30 vs 40 样本/s），放大 batch 是 −2.7%。
剩下的只有「改数据增强配方」这一条，而它改变训练分布本身，属于训练质量取舍，不是纯效率优化。

## F38：监控面板从「一次拉全量」改成「按标签页取数」（2026-11-16）

现象：重构后的界面点击一个 run 后要等很久，而且开着页面时四个节点的训练机 CPU 一直被监控啃着。测量把原因拆成三层，
都是**每 5 秒把十个接口全量重拉**带来的：

| 接口 | 改前 | 改后 | 说明 |
|---|---:|---:|---|
| `/system`（默认） | 1.34 MB，3.8–7.3 s | **10 KB，0.13 s** | 进程表 + 每卡 500 点曲线占了 90% 字段，而「拓扑」页只需要最新值 |
| `/system?processes=1&points=180` | —— | 508 KB / gzip 76 KB，0.4–0.6 s（满载机器上有 2 s 级抖动） | 只有「系统」页需要进程表和曲线；`&host=<节点>` 可再降到 0.17 s |
| `/throughput?source=ranks` | 1.05 MB | 0.21 MB（600 点） | 32 rank × 2000 列远超热力图能显示的密度 |
| `/control` | 1.9–3.8 s | 0.2–0.6 s | 之前是排在两个大响应后面等 GIL/队列，本身只要 1 KB |
| `/api/runs` | 106 KB（0.25 s） | 106 KB → gzip 11 KB | 6 个 run 的列表本身不重，重的是它没压缩 |
| `/series`（16 条） | 715 KB | → gzip 244 KB | 数值 JSON 的压缩比约 3× |

三处改动：

1. **服务端**：`/system` 的进程表和设备曲线改成 `?processes=1&points=N` 显式请求，默认只回最新值与每卡 `process_count`
   （`system.py:SystemHistory.snapshot`）；加 `GZipMiddleware(minimum_size=2048)`，数值 JSON 在隧道上省 3–10 倍；
   `nonfinite_leaves` 从「每个叶子都拼一次 f-string 的递归」改成迭代 + 只有真遇到 NaN/Inf 才拼路径（同形状
   payload：171 ms → 85 ms；这是每个响应都跑一次的开销）。
2. **前端**：`TAB_NEEDS` 表声明每个标签页需要什么（概览=series/events/checkpoints，曲线=series+throughput，
   系统=system(processes/points)，拓扑=ranks+system…），`loadRunData` 只并发取当前页需要的；
   切标签页时如果该页数据没取过就补一次再重绘；5 秒轮询也只刷新当前页。首屏不再等最慢的那个请求。
3. **静态资源版本号** `?v=18 → ?v=19`，避免浏览器继续用旧 `app.js`。

改后：首屏 7 个请求并发 **804 ms**（此前被 `/system` 的 4–7 s 拖住），轮询带宽从约 700 KB/s 降到约 50 KB/s
（gzip 后），四个训练节点上的监控 CPU 占用相应下降。

**复现**（node5 本机）：
```
T=<token>; for u in /api/runs/$T /api/runs/$T/control "/api/runs/$T/system?points=0" "/api/runs/$T/system?processes=1&points=180"; do
  curl -s --compressed -o /dev/null -w "%{time_total}s %{size_download}B $u\n" "http://127.0.0.1:18790$u"; done
```

## F37：多机续训落地——每个节点各自发布一代 checkpoint（2026-11-16）

F23 记过这个坑：每节点本地盘 + 只有 rank 0 发布 → 非 0 节点的 `model/`、`optimizer/` 是空目录，没有
`complete.json` 也没有 `training_state.pt`，`--resume` 在每台机器上都被校验拒绝，只能热启动（权重在、优化器/
步数/采样游标全丢）。这一节把它补上。

根因是两层，都在「谁写盘」上：

1. `HybridParallelCheckpointIO.save_sharded_model` 先 `if self.dp_rank != 0: return`——这里的 `dp_group` 就是 HSDP 的
   shard group（8 卡一组），所以每个 shard group 只有一个 rank 真正落盘；`control_saving` 又要求
   `replica_rank == 0`，于是 model/optimizer 的 **index 和 param-group 文件只有全局 rank 0 写**。非 0 节点上
   只有 `save_state_dict_shards` 在 master=False 时 `del shard; continue` 留下的空目录。
2. harness 的 `_publish_generation` 只在 rank 0 上跑：算 sha256、写 `complete.json`、rename、换 `latest.json`。

改法（**不动 ColossalAI**，全部在 harness/scripts 里）：

- `_save_leader(checkpoint_io)`：`dist.get_rank(checkpoint_io.global_dp_group) == 0`，即每个节点一个 writer；
- save 前对 leader 把 `checkpoint_io.replica_rank` 置 0，让该 replica 的 model/optimizer 分片与 index 写进本机
  staging（replica 之间权重本来就相同，各写一份是对的）；
- 发布改成每节点各自做：本机 leader 对 staging 做 sha256、写 `complete.json`（新增 `ranks` 字段，
  `FORMAT_VERSION=3`）、rename 成 `checkpoint-<id>`、原子换本机 `latest.json`；`training_state.pt` 也由每个节点
  leader 写（各 rank 内容一致）；
- 校验按节点：只要求**本 rank** 的 `rng-<rank>.pt` + `training_state.pt` + model/optimizer 存在（format <3 仍按
  老规矩要求全部 rank 的 rng）；sha256 只在节点 leader 上全量做，其余 rank 只查大小——否则 8 个 rank 会把 17 GB
  各读一遍。`COSMOS3_CKPT_VERIFY=size` 可整体跳过哈希（载入本身仍会拒绝截断/结构非法的 shard）；
- 启动脚本加 `RESUME=`：`RESUME=<旧 run>/checkpoints bash run_6node_train.sh <新 tag>`，四台各自读本地副本；
  `run_6node_train.sh` 在有 RESUME 时不再 `rm -rf` 输出目录。

实测（node1+node6，16 rank，`--max-iter 6 --save-every 3`，2026-11-16）：

| 检查项 | node1（rank 0-7） | node6（rank 8-15） |
|---|---|---|
| `latest.json` | `checkpoint-4bf3791c…` | 同 |
| `complete.json` | world 16、format 3、ranks [0..7]、31 文件、16.9 GiB | world 16、format 3、ranks [8..15]、31 文件、16.9 GiB |
| 内容 | 8 model + 14 optimizer + 8 rng + training_state | 同（各自 replica 的一份） |
| `check_checkpoint_complete.py` | RESUME-SAFE as rank 0 | RESUME-SAFE as rank 8 |

续训（`RESUME=<a 的>/checkpoints`、新 tag、`--max-iter 9`）：两台都通过指纹 / world 16 / 本 rank rng 校验，日志
第一行直接是 **`[step 7]`**（心跳里 step 6 = 存档步，即权重/优化器/采样游标都从 6 继续），loss 5.6 / 6.1 与 A 的
step 6（5.2 / 5.8）同量级，而不是从零起步的 ~21。第一步仍然要 570 s，但那是数据冷填充（wait 553 s），与 A 的
560 s 一致，和续训无关。

**磁盘代价**：每节点每次存档 17 GB（此前只有节点 0 写，但那份多机用不上）。7.1 TB 盘、当前 2.7–3.9 TB 可用；
100 步一存、1 万步就是 1.7 TB/节点。保留策略用 `scripts/prune_checkpoints.py <run> --keep N [--apply]` 手工做
（默认 dry-run，永不删 `latest.json` 指向的代和 staging 目录），不在训练循环里自动删。

**顺带修掉的一个真 bug**：`train_sft.py` 是手工 `booster.boost(...)`，模型上没有 `_colossal_booster` 属性，而
`apply_checkpoint` 一直靠 `model._colossal_booster` 找 booster——所以这条入口的 `--resume` 从来没跑通过（热启动
绕开了它）。现在改成显式传 `booster=`，`checkpoint.py` 里也改成 `getattr` + 明确报错。第一次续训尝试就是死在
`AttributeError: 'HybridParallelModule' object has no attribute '_colossal_booster'` 上，这也是「不跑到才算验证
过」的例子。

## F39：历史 step 只剩 15 个 rank、batch 480 —— 中继的 `cp -f` 把监控骗了（2026-11-16）

现象：在面板里点 step 1000 的 checkpoint 打开 step 详情，标题写着 `Batch 480 samples · world 32`，per-rank 表只有
15 个 rank（node5 自己的 0-7 一个都没有）。数据本身没坏：`rank-000.jsonl` 有 2505 行、从 step 1 到 2503。

三层原因，从「看到什么」到「为什么缺」：

1. **那个 batch 不是全局 batch。** rank 0 写的聚合记录在设计上只能读到自己节点的 rank 文件
   （`complete: false`、`ranks: [0..6]`、`reporting_world: 7`），监控知道这一点，于是用 mirror 收上来的 rank 记录
   重算（`aggregate_from_ranks`）——所以 batch = 它手上有的 rank 数 × 32。15 × 32 = 480。
2. **监控手上为什么只有 15 个 rank 的历史。** 首见一个 rank 文件时只读最后 512 KiB（`_read_rank_files`），面板
   14:30 重启过，2.4 MB 的文件因此从 ~step 1600 起才有数据；node5 自己的 8 个 rank 就停在这个窗口里，所以一个都
   不在。
3. **另外 16 个 rank 为什么反而有 step 1000。** 跨节点遥测由 `/tmp/relay_out.sh`（各节点→NFS staging）和
   `/tmp/relay_in.sh`（staging→面板主机）搬运，两边都用 `cp -f`——**先截断再写**。实测盯 `rank-009.jsonl`
   10 ms 一次：满长 2,455,442 B 的文件被看到过 **131,072 B**。监控的 `JsonlTail` 把「文件变短」判定为「被重写」，
   于是 `reset()` 从 0 重读整份 2.4 MB —— 这一下就把 step 1000 读进来了。24 个中继 rank 里 16 个至少被撞上过
   一次（于是有 step 1000），其余 8 个和被中继跳过的 node5 自己 8 个就没有。同一机制还让这些 rank 的曲线点重复：
   几个 rank 的序列在 ~890 个 step 上堆到 3000-5000 个点。

四处修改：

| 位置 | 修改 |
|---|---|
| 中继 | `scripts/relay_metrics.sh in\|out`（进仓库，替换 /tmp 里那两份）：先 copy 到同目录的 `.relay.*` 临时名，再 `mv` 到位——同目录 `mv` 是原子的，读者要么看到旧文件要么看到新文件 |
| `readers.py:JsonlTail` | inode 变化但没变短 = 原子替换，保留 offset；变短时先等一个 `shrink_grace_s=3s` 宽限期，仍在变短才当成真重写 |
| `series.py:ThroughputStore` | 每个 rank 的序列只在 step 严格递增时才 append，重读不再制造重复点 |
| `state.py:_read_rank_files` | 首见时 ≤8 MiB 的 rank 文件 **整份读**（2.5k step ≈ 2.4 MB），重启不再丢历史 |

改完实测：step 1000 变成 **32/32 ranks、batch 1024**，32 个 rank 的序列都从 step 1 开始、~2580 点无重复；再把
`rank-009/016/024` 每 10 ms 采样 60 s，最小长度都在 2.47 MB（改前几秒内就能抓到 131 KB 的窗口）。
