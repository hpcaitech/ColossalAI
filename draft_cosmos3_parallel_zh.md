# Cosmos 3 SFT 的并行策略：八卡现状与调优路线

对象：`examples/worldmodel/cosmos3` 的 SFT，Ascend 910B2 ×8（单机）/ ×16（node1+node5）。
对照：上游 `action_policy_droid_nano` recipe（256 卡）。

## 一、当前八卡用的是什么并行

`train_sft.py` 里的插件：

```python
plugin = HybridParallelPlugin(
    tp_size=1, pp_size=1, zero_stage=2,
    precision="bf16", max_norm=GRAD_CLIP,
    custom_policy=Cosmos3OmniPolicy(backend="npu"),
)
```

world_size = 8 时 `dp_size = 8 / (tp × pp × sp) = 8`，所以：

| 维度 | 取值 | 含义 |
| --- | --- | --- |
| TP | 1 | 无张量并行，每卡完整参数 |
| PP | 1 | 无流水线 |
| SP | 1 | 无序列并行 |
| DP | 8 | 纯数据并行 |
| ZeRO | stage 2 | 梯度 + 优化器状态按 DP 组切 8 份，**参数不切** |

每卡静态显存（实测口径）：

| 项 | 单卡 | 8 卡 ZeRO-2 |
| --- | --- | --- |
| bf16 参数（3.369B，含冻结） | 6.28 | 6.28（复制） |
| fp32 master（1.423B 可训练） | 5.30 | 0.66 |
| bf16 梯度（1.423B） | 2.65 | 0.33 |
| Adam m/v（fp32） | 10.60 | 1.33 |
| VAE（fp32，冻结）+ 缓冲 | 2.64 | 2.64（复制） |
| 合计 | 27.48 | **≈ 11.2 GiB** |

通信与数据：

- 反向传播中按桶做 reduce-scatter，桶大小 `zero_bucket_size_in_m=12`（12M 元素 ≈ 25 MB bf16），
  `overlap_communication=True` 是默认值，通信与反向重叠。
- 每微步梯度总量 2 × 1.423B × 2 B ≈ 5.7 GB（全组），每卡约 2.5 GB。
- 数据按 `DistributedSampler(num_replicas=dp_size)` 切分；**VAE 每卡各编各的样本**，不做任何共享。
- ZeRO-2 不支持 `no_sync`（插件里显式断言），所以 `grad_accum=32` 时**每个微步都同步一次梯度**，
  不是攒满 32 个微步再同步。

上游 256 卡的 recipe 是 HSDP：`data_parallel_shard_degree=8` × `data_parallel_replicate_degree=32`，
每卡 32 样本、`grad_accum_iter=1`。**单节点内的 shard 组正好是 8 卡**，所以我们的八卡配置就是上游一个
节点分片组，差别只在累积次数（我们用 `32 样本 × 32 累积`，上游用 `32 样本 × 1`）。

### Policy 侧做了什么

`Cosmos3OmniPolicy` 在 tp=1 时只做一件事：在 `postprocess` 里装
`ColossalTwoWayAttnProcessor`（双向注意力 kernel），不改任何模块 —— 也就是说**纯 ZeRO 跑法下
policy 不参与并行切分**。

tp>1 时它才做列/行并行重写：`to_q/to_k/to_v/add_q_proj/add_k_proj/add_v_proj` 列并行，
`to_out/to_add_out` 行并行；约束是 q 头数 16、kv 头数 8 都要能被 tp 整除且 kv 头数 ≥ tp
（`policy.py:96-108`），所以 tp ∈ {1, 2, 4, 8}。策略里没有 pipeline 和 sequence 切分逻辑。

## 二、八卡上为什么只有 DP 是合理的

单卡相位实测（2 样本、开重算）：VAE 编码 2.40 s、forward 0.22 s、backward 0.48 s、优化器 0.29 s，
合计 3.41 s。**VAE 占 70%，而且是逐样本的纯推理工作。**

| 方案 | VAE 会发生什么 | 结论 |
| --- | --- | --- |
| DP | 每卡编自己的样本，随卡数线性摊薄 | 唯一不浪费的方案 |
| TP=t | dataloader 只按 dp 切，同一 TP 组的 t 张卡拿到**同一批样本**，`pack_batch` 在每张卡都跑一遍 → VAE 成本 ×t | 直接亏 t 倍（VAE 占 70%） |
| PP=p | 各 stage 拿同一批数据，VAE 同样每 stage 重复 | 同上 |
| SP | ColossalAI 的 SP 挂在 tp 组上，tp=1 时不存在 | 不可用 |

TP 只有在"latent 只编码一次再广播给 TP 组"的前提下才不亏，而那样也最多把非 VAE 的那 30% 拿回来。
所以八卡的调参空间只剩：

1. ZeRO stage（0/1/2）——只影响 30% 的模型部分
2. 每卡样本数 S——VAE 每样本成本随 S 略降（1.26 → 1.20 s/sample），模型部分摊薄发射开销
3. 激活重算——决定 S 的上限（3.45 GiB/样本）
4. reduce 桶大小

**理论上限：把所有非 VAE 部分优化到零也只快 30%，实际可拿到的是 10% 量级。** 真正的 2–3× 只能来自把
VAE 移出关键路径（预计算 latent，代价是冻住数据增强），不来自并行策略；而"用另一个 stream 让 VAE 与
训练步重叠"也没那么大空间 —— 单卡 profile 里设备占用率已经是 91%，重叠最多拿回空转的那 9%（见第三节）。

## 三、八卡实测矩阵

条件：8×910B2、recipe 窗口（480p / 33 帧）、开激活重算、1 步预热 + 3 步计时、每卡 2 个 dataloader
worker、`torch.npu.synchronize()` 分段计时。单位秒。

| 配置 | 卡数 | 每卡样本 | data | pack(VAE) | fwd | bwd | opt | 合计 | 全组 samples/s | peak alloc | peak reserved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ZeRO-2 | 8 | 8 | 0.32 | 9.58 | 0.77 | 2.76 | 0.35 | **13.79** | **4.64** | 38.8 | 46.7 |
| ZeRO-2 | 8 | 6 | 0.27 | 7.07 | 0.65 | 2.06 | 0.35 | 10.40 | 4.62 | 31.9 | 37.7 |
| ZeRO-1 | 8 | 8 | 0.33 | 9.81 | 0.91 | 2.50 | 0.34 | 13.89 | 4.61 | 38.8 | 46.4 |
| ZeRO-2 | 8 | 10 | 0.43 | 15.71 | 1.05 | 3.31 | 0.38 | 20.88 | 3.83 | 45.7 | 55.7 |
| ZeRO-0 (DDP) | 8 | 8 | — | — | — | — | — | **OOM** | — | — | — |
| ZeRO-2 | **1** | 6 | 0.22 | 7.10 | 0.64 | 1.93 | 0.28 | 10.16 | 0.590 | 48.3 | 60.3 |

结论：

1. **数据并行效率 97.8%**（S=6 时 8 卡每卡 0.577 vs 单卡 0.590 samples/s）。HCCL 单机通信完全不是瓶颈，
   每微步 2.8 GB 的 reduce-scatter 被反向重叠掉了。
2. **ZeRO-1 与 ZeRO-2 只差 0.7%，在噪声范围内**。ZeRO-2 省显存（静态 11.2 vs 13.9 GiB），没理由换。
3. **ZeRO-0 在 8 卡跑不动 S=8**：DDP 不切优化器状态，静态 27.5 GiB + 激活 27.6 GiB 直接顶到 60.96 GiB。
4. **每卡 8 个样本就是最优**：S=10 时 VAE 每样本从 1.20 s 涨到 1.57 s（保留显存 55.7 GiB，allocator
   开始抖动），全组吞吐反而掉 17%。
5. 每样本边际成本 1.70 s，其中 VAE 1.20 s（71%）、模型 0.44 s、固定开销 0.23 s。

**所以八卡的"速度最优"就是现在这套**：ZeRO-2 + 每卡 8 样本 + 激活重算，再往上调只有百分之一二的量级。
剩下的优化空间不在并行上 —— 单卡 profile 里设备占用率已经是 91%（3.09 s 设备时间 / 3.41 s 步时），
VAE 与训练步做流重叠最多也只能拿回那 9%。

而"把 VAE 编译快一点"这条路也基本关上了：另一路在同一个节点上的实测是，`torch.compile(backend="npu")`
冷编译要 18 分钟（GE/TBE 串行编 2415 个算子的图），热缓存后第一次调用还要 95–140 s。单步 13.8 s
的训练里不可能挂这样的编译开销，除非把编译产物离线缓存下来复用 —— 那是另一件事。

### 一个工程坑：/dev/shm

8 卡 × 4 个 dataloader worker、每卡 32 样本时，一个 collate 出来的 batch 是 1.27 GB（uint8 视频），
32 个 worker × prefetch 2 直接把 32 GB 的 `/dev/shm` 撑爆（`RuntimeError: unable to allocate shared
memory(shm) ... No space left on device`）。两个办法：容器开 `--shm-size 256g`（宿主机有 2 TB 内存），
或者把 worker 数降到 1–2。实测 2 个 worker 时解码 0.32 s/步，已经完全被 13.8 s 的步时盖住。

## 四、多机多卡（node1 + node5，16 卡）

硬件事实：

- node1 `10.52.66.122–129`、node5 `10.52.66.90–97`，**同一 RoCE 网段**，HCCL 可跨节点；管理网
  192.168.0.46 / 192.168.0.6 可作 rendezvous。
- node5 有同样的 torch-npu 镜像，但**当前没有容器在跑**；`/mnt/disk0` 是各自的本地盘，
  node5 上**没有** checkpoint（8.6 GB）和数据集（success 585 GB）。
- ColossalAI 的 `HybridParallelPlugin` **没有 HSDP**（没有 shard/replicate 两级分组），只有一个 dp 组，
  所以跨 16 卡时 ZeRO 会跨节点切分，等价于 FSDP-16，而不是上游的 8×32。

预期与代价：

- 每卡静态 ≈ 6.28 + 18.55/16 + 2.64 ≈ **10.1 GiB**，样本上限比单机略升。
- 每微步 reduce-scatter 2.84 GB，跨节点部分走 RoCE；按 100 Gb/s 估计约 0.2 s/微步，
  相对 8 样本 ~12 s 的微步是可接受的（<2%）。
- 单机 8 卡的并行效率已经 97.8%，而跨节点多出来的只是这部分通信，所以 16 卡预计能到
  **8.5–8.9 samples/s**（16 × 0.58 × 0.92–0.95）。
- 吞吐仍然由 VAE 决定：16 卡也只是把 8192 的全局 batch 从 29 分钟压到约 15 分钟。

落地顺序：

1. node5 起容器（同一个 `dev-container.sh`，改 `NAME`），复制 8.6 GB checkpoint；数据集先只放几十个
   episode 做验证，或把 node1 的目录临时导出成 NFS。
2. 16 卡 `hccl_check.py` all_reduce 冒烟（`--nnodes=2 --node_rank`，master 用 192.168.0.46）。
3. 16 卡一步训练（`--samples-per-rank 8 --max-iter 1`）。
4. 再谈吞吐；此时瓶颈仍是 VAE。
