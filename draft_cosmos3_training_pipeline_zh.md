# Cosmos 3 SFT 一步训练的全流程：阶段、搬运、两个连续 iteration

对象：`examples/worldmodel/cosmos3/train_sft.py` 在 8×910B2 上的真实训练路径（ZeRO-2 / HSDP / Ulysses CP 都走同一条）。
数据取自 2026-11-06 的实测：`[pack] samples=2 AR=284 DM=7104`、`[model] trainable 1.423B / frozen 1.946B`、
单步约 4 s（每卡 2 样本、480p、33 帧、`grad_accum=1`）。

---

## 0. 先说清楚"一个 iteration"指什么

主循环是 `for it, batch in enumerate(dataloader)`，每转一圈取一个 **micro-batch**（每卡 `--samples-per-rank` 个样本）。
每 `--grad-accum` 圈才调用一次 `optimizer.step()`；变量 `step` 和 `--max-iter` 都只在 optimizer step 时递增。

所以本文把 **一个 iteration 定义成一次 optimizer step**，它包含 G 个 micro-step：

- recipe（官方 `action_policy_droid_nano`）：G = 32，每卡 32 样本；
- 今天的 8 卡冒烟：G = 1，每卡 2 样本。

这样定义才有意义，因为**相邻 iteration 之间真正的前置条件发生在 optimizer step 的边界上**——参数 all-gather 之后，
下一次 forward 才能开始。

---

## 1. 参与者：谁在算，各自在哪

| 角色 | 位置 | 职责 |
| --- | --- | --- |
| DataLoader worker × N（spawn 进程） | CPU | PyAV 解 33 帧 × 3 视角、增广、拼画布、resize + reflect pad、归一化、读 action/state |
| 主进程 | CPU | 组 prompt、tokenize、发起 H2D |
| 本 rank 的 NPU | 1 张 910B2 | VAE 编码、transformer 前后向、loss、optimizer、参数 all-gather |
| HCCL | 卡间 | 三套进程组，见下 |

进程组（由 `HybridParallelPlugin` 建立）：

- **dp 组**：`world / sp_size` 个 rank。`DistributedSampler(num_replicas=dp_group.size())` 按它切片，所以同一个
  CP 组里的两张卡拿到**同一批样本**（各算一半序列），这正是 CP 要的语义。
- **sp 组（CP）**：只承载两处 all-to-all 和 gather 汇合。
- **shard 组**：ZeRO 的 reduce-scatter 和参数 all-gather 都在它内部完成。HSDP 时它只是 dp×sp 的一段
  （`hsdp_shard_size` 个连续 rank）。
- **replica 组（仅 HSDP）**：shard 组之间同偏移的 rank，负责第二级的**已切分**梯度 all-reduce。

另外两个默认值后面会反复用到：`overlap_communication=True`（梯度归约与反向重叠）、
`overlap_allgather=False`（参数 all-gather 在 step 末尾**同步**完成）。梯度 bucket 是 12M 元素。

---

## 2. 阶段划分（按计算任务）

| # | 阶段 | 在哪算 | 做什么 | 通信 | 备注 |
| --- | --- | --- | --- | --- | --- |
| S1 | 索引与采样 | CPU（一次性） | 扫 LeRobot parquet 建窗口索引；`DistributedSampler` 按 dp 组切片 | 无 | `shuffle=False`，打乱在索引里做 |
| S2 | 视频解码与增广 | **CPU worker 进程** | PyAV 解码 33 帧 × 3 视角 → float → 随机裁剪增广 → 拼三视角 → `closest_canvas` + resize + reflect pad → `/127.5-1` | 无 | 与上一 iteration 的 NPU 计算**天然重叠**（DataLoader 预取） |
| S3 | H2D 拷贝 | CPU → NPU | `batch["video"].to(device, float32)`，`[B,3,33,H,W]` | 点对点，非集合 | 2 样本 480p ≈ 316 MB |
| S4 | VAE 编码 | **NPU**（冻结，`no_grad`） | Wan VAE encoder，内部按 4 帧一块循环、`feat_cache` 传因果状态；取 `latent_dist.mode()` → bf16 | 无 | 单步最大的一块（此前实测约 70%） |
| S5 | prompt + tokenize | **CPU 主进程** | 每样本拼 JSON（任务/时长/画布/宽高比）→ tokenizer → ids | 无 | 纯 CPU、串行，**不与 NPU 重叠** |
| S6 | 打包与嵌入 | **NPU** | `embed_tokens`、`proj_in`（latent patchify + 线性）、`time_proj/time_embedder`、`action_proj_in`、`rotary_emb`，再 scatter 进 joint 序列 | 无 | 产出 AR 塔 `und_seq`（284 token）与 DM 塔 `gen_seq`（7104 token）；同时在 policy 上 `set_packed_meta` |
| S7 | CP 切分 | NPU 本地 | `_cp_shard_fn`：两路各 pad 到 cp 的倍数，`chunk`，本 rank 取连续一段 | **无** | 只在 `--sp-size > 1` 时存在 |
| S8 | decoder × L 层 | **NPU** | 见 2.1 | **CP 时每层约 9 次 all-to-all**（前向） | 反向再来一遍（见 S11） |
| S9 | CP 汇合 | NPU + HCCL | `_cp_gather_fn`：两路 all-gather 后去掉 padding | 2 次 all-gather | 反向是 reduce-scatter |
| S10 | 解码头 + loss | **NPU** | `proj_out` / `action_proj_out`、unpatchify、flow-matching loss | 无 | **只有 DM 塔的 token 进 loss**，AR 塔只是条件 |
| S11 | backward | **NPU** | 解码头 → gather 反向（reduce-scatter）→ 逐层反向（checkpointing 时先重算前向）→ 交换的反向 | 每层约 9 次 all-to-all（反向） | 与 S12 重叠 |
| S12 | 梯度归约 | HCCL | 每个 bucket 满即 `reduce_scatter_tensor`（ZeRO-2）；HSDP 再对**已切分**梯度做 replica all-reduce | shard 组内；HSDP 加 replica 组 | `overlap_communication=True`，与反向重叠 |
| S13 | grad norm | HCCL | 每个 param group 一次**标量** all-reduce，用于 clip=1.0 | shard 组内 | 每 optimizer step 一次 |
| S14 | optimizer step | NPU 本地 | fp32 master weights 更新（HybridAdam；action head 学习率 ×5） | 无 | 只更新本 rank 的 shard |
| S15 | 参数 all-gather | HCCL | 把更新后的 bf16 参数按 bucket 收集回每个 rank | shard 组内 | **同步**（`overlap_allgather=False`），阻塞下一次 forward |

### 2.1 一层 decoder 里逐项展开

前向，每层：

1. `input_layernorm(und_seq)` / `input_layernorm_moe_gen(gen_seq)`——token 级，分片上结果与全量一致；
2. attention：
   - 7 个投影：`to_q/to_k/to_v`（AR）、`add_q_proj/add_k_proj/add_v_proj`（DM），Edge 还有
     `k_norm_und_for_gen`（对 AR 的 key 再归一化一次给 DM 用）；
   - qk-norm + mRoPE，都是 token 级，本地算；
   - **7 次 all-to-all**（`_exchange`）：AR 的 q/k/v、DM 的 q/k/v、以及 Edge 的 AR-k-for-gen。
     它把「序列分片、头完整」换成「序列完整、头分片」；
   - 裁掉 S7 补的 padding，用**全局**的每样本边界调用双向 kernel（`two_way_flat_attention`）；
   - 补回 padding，**2 次 all-to-all**（`_exchange_back`）换回「序列分片、头完整」，再过 `to_out` / `to_add_out`；
3. 残差；
4. 两个 MLP：`mlp(und)` 与 `mlp_moe_gen(gen)`，各自残差——本地，无通信；
5. checkpointing：
   - `--checkpoint-mode full`：整层交给 diffusers 的 `_gradient_checkpointing_func`，反向整层重算；
   - `selective`：把两个 MLP 包进 `torch.utils.checkpoint`，attention 的输出留着不重算；
   - `none`：全留。

**一个重要的观察**：all-to-all 是**按张量逐个**发的，不是融合的一次。前向 7 次 + 2 次 = 9 次/层，反向同样 9 次。
pack 越长，这些每次只有 MB 级的集合通信的启动开销占比越高。这是 CP 路径上最直接的优化点（把 q/k/v 拼成
一个 buffer 发一次、AR/DM 两路合成一次）。

---

## 3. 阶段之间的数据搬运与通信总表

量按 8 卡、每卡 2 样本、480p、33 帧估算。

| 从 → 到 | 内容 | 量 | 发生在 |
| --- | --- | --- | --- |
| 磁盘/NFS → 页缓存 | 33 帧 × 3 视角视频 + parquet | ~80 MB/样本（uint8） | S2，预取 |
| worker 进程 → 主进程（共享内存） | collate 后的 batch | ~158 MB/样本（float32 视频） | 每个 micro-batch 开始时 |
| CPU → NPU（PCIe） | `video` | ~316 MB/micro-batch | S3 |
| NPU 本地 | VAE latent | ~2 MB/样本（bf16） | S4 出口 |
| 主进程 CPU → NPU | index / timestep / action 等小张量 | KB 级 | S6 |
| NPU ↔ NPU（CP 组） | q/k/v 分片 | 每层 9 次，每次 MB 级 | S8 前向、S11 反向 |
| NPU ↔ NPU（CP 组） | 汇合的输出分片 | 2 次 all-gather | S9 |
| NPU ↔ NPU（shard 组） | 梯度分片 | 总量 2.85 GB（每 rank 发 2.85 GB、收 356 MB） | S12 |
| NPU ↔ NPU（replica 组，仅 HSDP） | 已切分梯度 | 每 rank 356 MB | S12 |
| NPU ↔ NPU（shard 组） | 标量 norm | 几个 float | S13 |
| NPU ↔ NPU（shard 组） | 参数 all-gather | 总量 2.85 GB（每 rank 发 356 MB、收 2.85 GB） | S15 |

注意冻结的 1.946B 参数**不参与**任何 ZeRO 通信——它们不进 optimizer，也就不在 shard 里，只随模型复制在每张卡上。

---

## 4. 两个连续的 iteration

记第 k 次 optimizer step 的第 j 个 micro-step 为 `(k, j)`，j = 0…G−1。

### iteration k

```
(k, 0)
  ├─ 前置：iteration k−1 的 S15（参数 all-gather）已经返回；
  │         iteration k−1 最后一次 backward 的激活已经释放（checkpointing 的 saved tensors）
  ├─ S2  取 micro-batch：worker 早就解好了（预取），这里只是从队列取
  ├─ S3  H2D：video 上卡
  ├─ S4  VAE 编码（NPU，no_grad）
  ├─ S5  组 prompt + tokenize（CPU，主进程，阻塞等它）
  ├─ S6  打包 + 嵌入（NPU），同时 policy.set_packed_meta(本批的 AR/DM 边界)
  ├─ S7  CP 切分（sp>1）
  ├─ S8  逐层前向（每层 9 次 all-to-all）
  ├─ S9  CP 汇合（2 次 all-gather）
  ├─ S10 解码头 + loss
  └─ S11 backward 开始；梯度 bucket 满一个就发一次 reduce-scatter（S12），与剩余反向重叠

(k, 1) … (k, G−1)
  └─ 重复上面的 S2→S11；每次 backward 都往 S12 里添自己的那部分梯度

(k, G−1) 的 backward 返回后
  ├─ S12 收尾：所有 bucket 的 reduce-scatter 完成（HSDP 时还有 replica all-reduce）
  ├─ S13 每个 param group 一次标量 all-reduce 求全局梯度范数
  ├─ S14 optimizer step：更新本 rank 的 fp32 master shard
  └─ S15 参数 all-gather：把更新后的 bf16 参数收集回所有 rank（同步，阻塞）
```

### iteration k+1

```
(k+1, 0)
  ├─ 前置：iteration k 的 S15 已经返回  ← 唯一的跨 iteration 硬依赖
  ├─ S2  取 micro-batch（worker 在 iteration k 的 NPU 计算期间已经解好）
  ├─ S3  H2D
  ├─ S4  VAE
  ├─ S5  tokenize
  ├─ S6  打包 + set_packed_meta（覆盖上一批的元数据）
  ├─ …
  └─ 与 iteration k 完全同构
```

### 相邻 iteration 的前置条件，逐条

| 依赖 | 谁等谁 | 为什么 |
| --- | --- | --- |
| 参数 | k+1 的 S6/S8 **必须**等 k 的 S15 | forward 读到的是 working param，S15 之前它还是旧值。S3/S4 对参数没有数据依赖，现在之所以也排在 S15 之后，纯粹是单线程的程序顺序 |
| 显存 | k+1 的 S3/S4 必须等 k 的最后一次 backward 释放激活 | 同一张卡的显存 |
| 数据 | k+1 的 S2 **不等** k 的任何阶段 | DataLoader worker 独立进程，预取 |
| 元数据 | k+1 的 `set_packed_meta` 必须晚于 k 的所有 forward（含 checkpointing 的重算） | metadata 挂在 processor 上，是跨 iteration 的可变状态；重算发生在 backward 里，所以必须等 backward 结束 |
| 梯度 | S13 必须等 S12 全部完成；S14 必须等 S13 | norm 要覆盖全部 shard |
| bucket | S12 的每次 reduce-scatter 只依赖产生它的那段反向 | 所以能和反向重叠 |

**关键路径上的串行段**（NPU 在这些时刻是空的或等着的）：

1. **S3 + S5**：H2D 和 tokenize 都是串行的，NPU 在 S5 期间完全空闲；
2. **S4**：VAE 在每个 micro-batch 的前向之前串行执行，不能和上一个 micro-batch 的 decoder 重叠；
3. **S15**：同步 all-gather，阻塞下一个 iteration；
4. **S2 的首次命中**：worker 还没预热时（第一个 batch）会真的等解码。

---

## 5. 由此看出的、VAE 之外的优化点

按"改动大小 / 预期收益"排：

1. **让 S15 与下一个 iteration 的 S3/S4 重叠**。ZeRO 的参数 all-gather 是同步的
   （`overlap_allgather=False` 是默认值），而 VAE 编码**不需要 transformer 参数**——这是天然的重叠窗口：
   把 all-gather 异步发出去，VAE 算完再等它。改动只在插件的一个开关，收益是每步省掉一次 2.85 GB 接收的等待。
2. **把 tokenize 挪进 Dataset（worker 进程）**。现在 S5 是主进程里的纯 CPU 串行段，挪进 worker 后与 NPU 重叠，
   且不改变数值（prompt 只依赖样本自身的字段）。
3. **H2D 传 uint8**。现在 worker 已经把视频转成 float32 并归一化，搬 316 MB；改成搬 uint8 的 79 MB、
   在卡上做 `/127.5 - 1`，省 4 倍 PCIe 流量和一次主机侧 dtype 转换。
4. **CP 的 all-to-all 融合**。每层 9 次 → 3 次（qkv 合一、AR/DM 合一），反向同理。序列越长、层数越多收益越大；
   我们现在的 pack 只有 7389 token，收益主要体现在启动开销上。
5. **grad norm 的标量 all-reduce 合并**。每个 param group 一次，可以并成一次。
6. **VAE 本身**：`--compile-vae`（静态 shape）、更低精度、或把编码摊到不参与训练的卡上（数据流水线并行）。
   这是唯一能真正减少 VAE FLOPs 的方向；单纯"重叠"在同一张卡上不会凭空多出算力。

---

## 6. 与实测对齐

| 项 | 值 | 来源 |
| --- | --- | --- |
| 每 micro-batch 的 pack | AR = 284、DM = 7104（2 样本/卡） | `[pack]` 行 |
| 参数 | 可训练 1.423B / 冻结 1.946B | `[model]` 行 |
| 单步耗时 | ~4 s（2 样本/卡，`grad_accum=1`，热缓存） | 三档对比日志 |
| 峰值显存 | full 18.01 / selective 23.97 / none 33.85 GiB | 同上 |
| VAE 占比 | 约 70% | 此前的单步拆解实测 |
