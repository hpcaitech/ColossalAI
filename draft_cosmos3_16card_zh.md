# Cosmos 3 双机 16 卡：node5 环境、VAE fp16 实测与官方 DROID 配方的工期推算

本篇记录三件事：node5 上把训练环境配成与 node1 一致的过程与结论、2 节点 16 卡上 VAE 用 fp16 与不用 fp16 的实测差别、
以及按官方 `action_policy_droid_nano` 配方（10000 iter × 全局 batch 8192）推算这套 16 卡需要多久。

## 一、node5 的环境：所谓"安装"只有两步

| 项 | node1 | node5 | 结论 |
| --- | --- | --- | --- |
| 主机名 | npu-910b2-46 | npu-910b2-6 | 都是 8×910B2 |
| 驱动 | 25.2.1 / V100R001C21SPC009B220 | 同左，逐字一致 | 跨节点 HCCL 不需要处理版本差异 |
| 镜像 | `ascendai/torch-npu:2.13.0rc1-cann9.1.0-910b-openeuler24.03-py3.12` | 同左（本地已有） | 不需要重新拉 |
| `/home` | NFS `192.168.0.110:/` | **同一份 NFS** | 代码、venv、脚本天然共享 |
| `/mnt/disk0/jiangzs` | 本地 NVMe | 本地盘上的同名副本 | checkpoint 8.6 GB、DROID success 707 GB 齐备 |
| RoCE | 10.52.66.122–129 | 10.52.66.90–97 | 同一 /24，link 全 UP |
| 控制面 | 192.168.0.46 (bond1) | 192.168.0.6 (bond1) | 唯一两节点互通的网段 |

**代码不需要 clone。** `/home/niuzifan` 是同一份 NFS，node5 看到的就是 node1 那棵树；
用 `md5sum` 在两个容器里核对过 `train_sft.py` 与 `packing.py`，字节相同。
唯一要新建的是容器：`/home/niuzifan/scripts/dev-container-n5.sh`，与 node1 的 `dev-container.sh` 同一份定义
（`--privileged --network host --shm-size 32g --user 0:0`、显式挂驱动目录），只改了一处——
`ASCEND_PROCESS_LOG_PATH` 指到 `ascend-log/node5`，否则两个容器会往同一个 NFS 目录写进程日志。

两处踩过的坑：

1. 容器里 `git` 报 `dubious ownership`——仓库属主是宿主上的 `niuzifan`，容器里是 root。加一条
   `git config --global --add safe.directory <repo>` 即可（写在共享 HOME 里，两节点都生效）。
2. node5 宿主机上没有 `git`，也不要把 venv 拿到宿主机上跑：venv 的 `bin/python` 指向镜像里的解释器路径，
   宿主机上不存在。**一切训练命令都在容器内执行。**

## 二、双机 16 卡 HCCL

启动方式：每节点一个 `torch.distributed.run --nnodes=2 --node_rank=N --nproc_per_node=8`，
master 用 `192.168.0.46`，容器是 `--network host` 所以地址直接可用。

```bash
export HCCL_SOCKET_IFNAME=bond1     # 控制面。数据面由驱动按 hccn 里每卡的 10.52.66.x 自动选 RoCE
export HCCL_CONNECT_TIMEOUT=600
```

验证：16 rank 的 `all_reduce` 全部 `ok=True`，sum=136.0 = 16×17/2，两节点进程都以 0 退出。
也就是说跨节点的集合通信本身没有障碍，后面测到的差异不是"连不上"，而是"连上了但慢"。

## 三、测法

- 三次运行，其余完全相同：`--vae-dtype float32`（6 步）、`--vae-dtype float16`（6 步）、
  `--vae-dtype float16 --grad-accum 4`（3 步）。前两次是 A/B，第三次用来解通信与计算的占比。
- 16 卡 = 2 节点 × 8，`--samples-per-rank 2 --grad-accum 1` → **全局 batch 32**。
- 每行日志由 launcher 打上 `[HH:MM:SS]`，取相邻 `[step N]` 的间隔作为一次完整 iteration
  （取数等待 + VAE 编码 + 前反向 + ZeRO 更新）。第 1 步含预热，统计时从第 2 步起算。
- 显存用 `COSMOS3_REPORT_PEAK_HBM=1`，读 caching allocator 的高水位（`npu-smi` 在卡忙时读不到）。
- 两组的唯一变量是 `--vae-dtype`，数据、canvas（736×544）、33 帧、checkpoint 完全相同。

## 四、VAE fp16 vs fp32（16 卡实测）

| | VAE fp32 | VAE fp16 | 差 |
| --- | --- | --- | --- |
| 单 iteration（32 样本） | **5.20 s** | **4.20 s** | −1.00 s（−19%），**1.24×** |
| 逐 step 间隔 | 6/5/5/5/5 s | 4/4/4/4/5 s | |
| 整机吞吐 | 6.15 样本/s | **7.62 样本/s** | +24% |
| 每卡峰值显存 | 16.97 GiB | **12.30 GiB** | −4.67 GiB（−27.5%） |
| loss 区间（6 步） | 3.09–13.54 | 3.08–17.61 | 同量级，无 NaN/Inf |

显存那 4.67 GiB 基本就是 VAE 权重与激活从 fp32 降到 fp16 的量（单样本编码峰值 6.11 → 3.11 GiB），
它直接把"每卡能塞多少样本"的天花板抬高：16.97 → 12.30 GiB 意味着同样 64 GiB 卡上多出 4.7 GiB 余量。

时间上只省了 19% 而不是 VAE 单机 profiler 给出的 2.46×，原因是**一步里 VAE 只占一部分**：
按单卡 profiler，2 个样本的 fp32 编码是 2.46 s、fp16 是 1.00 s，理论上应省 1.46 s；实测省 1.0 s。
日志时间戳只有 1 s 分辨率，5 个间隔的均值本身就在 ±0.5 s 量级，所以这个差额（0.6–1.4 s）
与 1.46 s 并不矛盾——**这一步的收益量级是"每样本省 0.5 s 左右"，不是"整步快 2.46 倍"**。
**但这 1.0 s 是白拿的**——精度门禁已经做过：64 个真实 DROID 窗口上，fp16 latent 与 fp32 的差
是配方自带 bf16 量化的 0.58–0.64 倍，bf16 编码则是 4.7–5.0 倍（见 `draft_cosmos3_vae_optimization_zh.md` 第五节）。

改动只有两处，且对 fp32 路径零影响：

- `train_sft.py` 增加 `--vae-dtype {float32,float16}`，`build_vae(checkpoint, dtype)` 把 dtype 传给 `from_pretrained`；
- `packing.py` 里 `vae.encode(batch["video"].to(device, torch.float32))` 改成跟随编码器自身 dtype
  （`next(vae.parameters()).dtype`），否则 fp16 权重配 fp32 输入会报 dtype 不匹配。

## 五、16 卡相对 8 卡的真相：每卡吞吐反而降了

| | 每卡样本 | 单 step | 整机吞吐 | 每卡吞吐 |
| --- | --- | --- | --- | --- |
| 单机 8 卡（此前实测，fp32） | 2 | ~4.0 s | 4.00 样本/s | 0.500 样本/s |
| 双机 16 卡（本次，fp32） | 2 | 5.20 s | 6.15 样本/s | 0.385 样本/s |
| 双机 16 卡（本次，fp16） | 2 | 4.20 s | 7.62 样本/s | 0.476 样本/s |

8 卡那一行是此前同一口径的实测（`draft_cosmos3_training_pipeline_zh.md`），代码版本早一些，只能当量级参考。
可靠的那一半是 16 卡自己的分解：每个 optimizer step 的固定通信约 **0.9 s**，占 fp16 单步的 21%
（由下面的 grad-accum 对比解出）。也就是说 16 卡每卡吞吐比 8 卡低两成左右，
16 卡的价值不是"同样的活干得更快"，而是**同样的墙钟时间里做两倍的全局 batch**：
整体吞吐 6.15 / 4.00 ≈ 1.54×，不是 2×。
fp16 在 16 卡上把每卡吞吐拉回 0.476 样本/s，基本追平 8 卡 fp32 的 0.500。

这也解释了为什么单机 8 卡时 HSDP/SP 都不是瓶颈：这个负载的每卡工作量太小，任何跨节点同步都会立刻显形。

## 六、官方 DROID 配方与工期推算

官方配方（`NVIDIA/cosmos-framework`，`examples/toml/sft_config/action_policy_droid_nano.toml`，逐字）：

| 项 | 值 |
| --- | --- |
| 并行 | HSDP `data_parallel_shard_degree=8` × `data_parallel_replicate_degree=32` = **256 rank**（GB200 参考机：64 节点 × 4 卡） |
| 每卡样本 | `max_samples_per_batch = 32`（res480） |
| 梯度累积 | `grad_accum_iter = 1` → 全局 batch **8192** |
| 迭代数 | `max_iter = 10000` → 合计 **81920 万样本 = 81.92M** |
| 精度 / lr | bf16 / 2e-4，cosine `cycle_lengths=[100000]` |
| 激活重算 | `mode="full"`，`save_ops_regex=["fmha"]` |
| 其他 | 默认 `compile_tokenizer.enabled=true`（对 Wan VAE 做 torch.compile），每 1000 步存一次 |

把本次实测的每样本成本线性外推（这是保守口径，下面给出通信摊薄后的修正）：

| 配置 | 每样本墙钟 | 一次 8192 样本更新 | 10000 次迭代（16 卡墙钟） | 折合 GPU·h | 每卡吞吐 |
| --- | --- | --- | --- | --- | --- |
| 16 卡，VAE fp32 | 0.163 s | 22.2 min | **3697 h = 154 天** | 59,100 | 1385 样本/h |
| 16 卡，VAE fp16 | 0.131 s | 17.9 min | **2987 h = 124 天** | 47,800 | 1714 样本/h |

### 通信摊薄后的修正（第三次实测）

上面的口径把每个 32 样本 micro-step 都要付的 ZeRO 开销也算进了每一样本，而官方是每 8192 样本才付一次。
再跑一次 `--grad-accum 4`（其余不变，fp16），就能把这笔固定成本解出来：

| | micro-step 数/更新 | 一次更新实测 |
| --- | --- | --- |
| `grad-accum 1` | 1 | 4.20 s |
| `grad-accum 4` | 4 | 15 s / 13 s（均值 14.0 s） |

解 `m + c = 4.2`、`4m + c = 14.0`：**每个 micro-step（32 样本）≈ 3.3 s，每次 optimizer step 的固定通信 ≈ 0.9 s**，
后者占单步 21%，是跨节点并行最主要的固定开销（官方配方把它摊薄 256 倍，见下）。
（两次 g=4 的间隔给了 m ∈ [2.9, 3.6] s 的区间，下面按区间给结论。）

官方每 256 个 micro-step 才更新一次，所以一次 8192 样本的更新是
`256 × 3.3 + 0.9 ≈ 838 s = 14.0 min`，10000 次约 **2330 h ≈ 97 天**（区间 87–107 天）。
fp32 的 m 换成 4.3 s，同式得 1094 s/更新 → **约 127 天**。

| | 线性上界 | 摊薄修正 | 差距 |
| --- | --- | --- | --- |
| fp32 | 154 天 | **127 天** | −18% |
| fp16 | 124 天 | **97 天** | −22% |

**结论：这套 2 节点 8+8 跑完官方 DROID 配方，fp16 约 97 天（保守上界 124 天），fp32 约 127 天；
fp16 省约 30 天，比例 1.3×。**

三条必须说清楚的限定：

1. **`m` 与 `c` 只由两个点解出**，时间戳分辨率 1 s，所以给出的是 87–107 天的区间而不是一个数；
   要收窄就得多跑几个 `grad-accum` 档位。
2. **256 卡那个数字是线性扩展假设**，没有实测。跨节点规模上去以后 RoCE 带宽和 HCCL 拓扑会改变系数，
   而且官方是 HSDP（节点内 8 卡分片、节点间 32 路复制），我们这里是 16 路全分片，通信模式本来就不同。
3. **报告 Fig. 8 写 DROID post-training 用 58K 样本，与 TOML 的 10000×8192 对不上。**
   本文按 TOML 推算，因为 TOML 是可直接执行的配方，且它折算出的 GPU·h 与"256 卡跑一次 SFT"
   的量级自洽（fp32 约 48,600 GPU·h，256 卡约 8 天）。

## 七、结论

1. node5 的"安装"实际只有起容器一步；代码、venv、脚本由同一份 NFS 天然一致，数据在本地盘同名路径上。
2. 双机 16 卡 HCCL 通，`HCCL_SOCKET_IFNAME=bond1` 是唯一需要设的变量。
3. VAE fp16 在 16 卡上：单步 5.20 → 4.20 s（1.24×），峰值显存 16.97 → 12.30 GiB（−27.5%），loss 无异常。
4. 16 卡的每卡吞吐比单机 8 卡低约两成（8 卡基线取自更早的日志，量级参考）；整体收益约 1.5× 而不是 2×，
   固定通信约 0.9 s/更新。
5. 官方配方（10000 iter × 8192）在 16 卡上：fp16 约 **97 天**（保守上界 124 天）、fp32 约 **127 天**；
   要在一周量级跑完需要 256 卡以上。fp16 的收益在工期上放大成约 30 天。
6. 想再压工期，优先级依次是：继续做薄每个 micro-step 的 3.3 s 里 VAE 那约 1.0 s（RMSNorm 融合、padding 下沉
   与去 clone 均已完成：fp16 下编码从 0.926 s 降到 0.70 s/次（1.32×，默认窗口随之为 8 帧），
   相对 fp32 原生 2.175 s 共 3.08×，见
   `draft_cosmos3_vae_optimization_zh.md` 第十三、十四节)、
   开 `--compile-vae` 并照官方做 rank 分摊的 AOT 编译、以及把 ZeRO 的跨节点通信改成官方那种节点内分片 + 节点间复制的 HSDP。
   注意**不要**照抄官方"每次调用 24 帧"的分块编码：910B2 上实测只有 1.04–1.06×，而显存涨 2–3 倍，
   本机最优是 12 帧（见 `draft_cosmos3_vae_optimization_zh.md` 第十一节）。

## 附：复现命令

```bash
# 连通性
dev-container.sh /home/niuzifan/scripts/hccl_2node_inner.sh 0      # node1
dev-container-n5.sh /home/niuzifan/scripts/hccl_2node_inner.sh 1   # node5

# 训练（node1 起 rank 0，node5 起 rank 1）
dev-container.sh   /home/niuzifan/scripts/train_2node_inner.sh vaefp16 0 --vae-dtype float16
dev-container-n5.sh /home/niuzifan/scripts/train_2node_inner.sh vaefp16 1 --vae-dtype float16
```
