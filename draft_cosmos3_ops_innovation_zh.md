# Cosmos 3 在 Ascend 910B2 上的算子级优化创新：VAE 编码与训练流水线

本文回答一个问题：在 910B2 上，Cosmos 3 的训练还剩下哪些**算子级**的优化空间，按什么顺序做、每项值多少、哪些不值得做。

它建立在四份已有草稿之上（`draft_cosmos3_vae_optimization_zh.md`、`draft_cosmos3_training_pipeline_zh.md`、`draft_cosmos3_profile_zh.md`、`draft_cosmos3_16card_zh.md`），不重复其中已实现或已实测的内容，只做增量设计；硬件约束取自 `deepinfer/docs/npu-guide`，torch_npu 算子能力以官方文档核实为准，代码位置取自本仓库当前 checkout。

---

## 〇、摘要与结论

### 0.1 三句话

1. **杠杆不在 Transformer 的逐元素融合，而在优化器与 VAE 的调度重叠。** 实测一步 3.413 s 的相位拆解是：pack（VAE 编码 + 提示词 + 打包）2.402 s（70.4%）、反向 0.475 s（13.9%）、优化器 0.291 s（8.5%）、前向 0.217 s（6.4%）。把前向里所有可融合的逐元素链全部融掉，理论上限约 0.03–0.05 s/步（约 1.5%）。真正的两项：优化器段在 8 卡真机上是 **332 → 203 → 104–109 ms**——先换 `NpuFusedAdamW`（−129 ms），再复用 cast buffer + 一次多张量 grad norm（−95 ms），合计 **−223 ms / −67%**（见 6.1.2/6.1.3），与算子账（AdamW 138 → 13 ms、grad norm 64 → 4 ms、cast/回写 24 → 2 ms）吻合；把 VAE 编码与参数更新段重叠（P5）值 **0.1–0.3 s/步（3–9%）**，上限是整步实测的设备 idle 约 0.32 s。**按真机实测修正：优化器可回收 8 卡 0.223 s/步，加上在线 VAE 重叠的 0.1–0.3 s/步，合计 0.32–0.52 s/步——这才是全部算子工作的天花板；VAE 内部的算子项加起来只有 0.06–0.12 s/步。**
2. **VAE 剩下的空间是"布局与搬运"，但大头绑在 227 个卷积各自的 pad/cat 上，入口处的融合只能拿走很小一块。** fp16 之后 0.499 s 里 Conv3D 仍占 45%（0.225 s），算力利用率约 41%——继续压需要手写 im2col Conv3D（P3，0.02–0.05 s/样本，成本 4–8 周）；而入口 kernel（P1）只消掉 conv_in 这一层的 pad/cat 与 patchify，值 0.005–0.010 s/样本，权重预转（P2）值 0.002–0.005 s/样本。真正低成本的是残余逐元素（P4，0.008–0.016 s/样本）。**VAE 侧全部算子项加起来约 0.04–0.08 s/步，不如把它整段与 optimizer/HCCL 重叠（P5，0.1–0.3 s/步，上限是整步 idle）。**
3. **有一条代数恒等式必须先算清楚，但它不是生产路径：patchify(2×2) + conv_in(3×3×3) 严格等价于一个 stride-2 的 Conv3D(3×6×6)。** 数学上 MAC 数不变（每输出 324），权重可一次性预置换（纯 Python 验证 max|Δ| = 3.55e-14；时间索引必须是 `t+dt−2`，写成 `t+dt−1` 会整体错位）。但 910B 的 `NC1HWC0` 把通道固定补齐到 16 lane：融合后 C_in=3 的 issued K = `16 × 108 = 1728`，而原路径 C_in=12 只有 `16 × 27 = 432`——**4× 的 issued MAC**。conv_in 这一层只有 0.015 s（fp32），4× 的增量上界约 0.018 s，再减掉省下的 patchify，**净亏约 0.005–0.013 s/样本**。正确做法是把 patchify/归一化/填充折进**数据入口 kernel**（保持 conv_in 的 12 通道），恒等式只保留为该 kernel 的数值 oracle。这条"用 issued-K 算术否掉一个漂亮恒等式"的过程，是本文方法论的样板。

### 0.2 杠杆排序（在线 VAE 口径；离线后的重排见 0.4 与第六节）

| # | 优化对象 | 实测基线 | 可拿回的量级 | 证据等级 | 成本 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 优化器去逐 tensor + 多张量 cast/grad norm + `NpuFusedAdamW`（D05/P01） | 1 卡段 0.264–0.291 s；8 卡段 0.332 s | **实测：8 卡 332 → 104–109 ms（−223 ms / −67%，两轮合计）；1 卡换类 −32 ms** | measured（6.1.2/6.1.3） | 中 | P0 |
| 2 | VAE 编码跨 iteration 与训练步重叠（P5） | VAE 1.0 s/2 样本；整步设备 idle 约 0.32 s | 0.1–0.3 s/步（3–9%）；与 T5 的 all-gather 重叠是同一笔账，不可相加 | measured 占用率 + inferred | 中 | P0 |
| 3 | host 侧 `aten::to` / loss / tokenize（D08） | host 2.35 s/步、设备空转约 0.32 s/步 | **先定位**；若确认在关键路径，上限 0.1–0.2 s/步（inferred，不作承诺） | measured 现象 + inferred | 中 | P0（先定位） |
| 4 | VAE 残余逐元素二次融合（P4） | 逐元素 8% + Transpose 2%（fp32 口径） | 0.008–0.016 s/样本 → 0.016–0.032 s/步 | inferred | 中 | P1 |
| 5 | VAE 数据入口一体化（P1：uint8+归一化+patchify+填充一个 kernel） | 只消 conv_in 一层：patchify 0.012 s + 1 次 Pad + 1 次 Cat（全链 Pad/Cat 的大头属于其余 226 个卷积） | 0.005–0.010 s/样本 → 0.01–0.02 s/步；另省 H2D 4× | measured 占比 + derived（红队修正） | 中 | P1 |
| 6 | VAE 布局：权重预转 + conv_in 直落 NDC1HWC0（P2） | TransData 0.040 s（fp16），权重侧按字节远小于激活侧 | 0.002–0.005 s/样本 → 0.004–0.01 s/步 | measured + derived（红队修正） | 低/中 | P2 |
| 7 | 手写 causal Conv3D（P3，仅 down0/down1） | Conv3D 0.225 s @ 全族约 41%；down0/down1 本就更高效 | 0.02–0.05 s/样本 → 0.04–0.10 s/步 | inferred | 很高（4–8 周） | P2 |
| 8 | Transformer 逐元素融合（T2） | 前向 0.217 s、每层约 60 次 launch | 0.03–0.05 s/步（0.9–1.5%） | derived（GM 流量账） | 高 | P2 |
| 9 | P6：整链路 NDHWC 后用 `npu_rms_norm` | 与 P4(a) 同一条 affine 链 | 0.002–0.008 s/样本 | inferred | 中 | P2 |
| — | patchify + conv_in 代数融合（作为卷积替换） | conv_in 整层 fp32 约 0.015 s | **净亏** 0.005–0.013 s/样本（4× issued MAC） | derived（C0=16） | — | 不做 |
| — | two-way K/V 交错消除 | 每层 60.5 MB 流量 = 1.06 ms/前向 | 约 3.2 ms/步（含反向 cat） | measured 流量 + derived | 高（要改内核） | 不做 |
| — | FP8 / bf16 VAE / 直接 channels_last / torch.compile | — | 0（不可用、精度不过门禁或环境缺失） | measured/文档 | — | 不做 |

这张表就是本文的结论：**先做第 1–3 项（优化器、重叠、host 审计）；第 4–6、8–9 项都是"顺手做"的清理；第 7 项是唯一有量级、但成本也最高（4–8 周）的研究项；最后三行明确不做。** 注意第 4–9 项全部加起来（约 0.06–0.12 s/步）仍小于第 1–2 项（合计 0.32–0.52 s/步）——**调度与重叠才是 VAE 那 70% 的正确拿法，不是继续在编码器里抠算子。**

### 0.3 与已实现工作的边界

已实现且已实测、本文不再重复论证的：VAE fp16（1.229 → 0.499 s，2.46×）、12 帧分块编码（1.04–1.06×，本机最优）、RMSNorm 融合（fp16 0.927 → 0.748 s，1.24×）、HSDP/Ulysses CP、file_system 共享内存策略。**SAC 不在这个列表里**：本 checkout 的 `train.py` 只有 full checkpointing，3.413 s 的实测也没开重算（见 4.4）。本文只提它们的**改进版**或**后续增量**。

### 0.4 部署模式变更（2026-11-09）：VAE 改为离线预提取

后续可能把 VAE 编码移到训练循环之外（离线预计算 latent，训练时直接读 `--latents`）。这会改写本文的优先级，先给出结论与新的相位账。

**新的单步预算**（沿用 `--latents` 的实测口径：3.41 s → 1.05 s）：

| 相位 | 离线 latent 后的时间 | 占 1.05 s | 与在线模式的差别 |
| --- | --- | --- | --- |
| VAE 编码 | 0 | 0% | 整段移出训练循环 |
| 数据加载 + pack + loss | 约 0.07 s | 6.4% | 只读 latent，不再读 158 MB/样本的像素 |
| forward | 0.217 s | 20.7% | 不变 |
| backward | 0.475 s | 45.2% | 不变 |
| optimizer + zero_grad | 0.291 s | **27.7%** | 不变，但占比从 8.5% 升到 27.7% |

三条直接推论：

1. **优化器从"8.5% 的边角"变成约 25–28% 的主项**。按上表 0.291 s 是 27.7%；1 卡复测段 0.264 s（≈25%），8 卡 0.332 s。真机可回收量分两轮拿到：1 卡换类 **−32 ms**，8 卡换类 + 多张量 cast/grad norm 合计 **−223 ms**（332 → 104–109 ms，约占离线整步的 10%）。它仍是 P0——这是目前唯一在真机上验证过收益的项。
2. **VAE 重叠（P5/5.3）失去意义**：训练步里已经没有 VAE 可重叠。保留 5.3 的分析只作为"如果回到在线 VAE"的备查；新的重叠对象变成"下一步的数据预取 / 参数 all-gather"，量级由 4.5 的 0.01–0.11 s 决定，而不是 0.32 s 的整步 idle。
3. **VAE 侧算子项（P1/P2/P3/P4/P6）不再计入训练步时**，它们的价值变成**离线数据集生产时间**：707 GB、约 17.9k 个样本，按 0.5 s/样本单卡算约 2.5 小时，8 卡约 19 分钟。这是"一次性/每次换增强配置时"的成本，不是每步成本。因此它们从 P0/P1 降到"离线工具链优化"，只在需要缩短数据集周转或做大规模消融时才优先。
4. **离线 latent 的代价要单独记账**：增强必须在编码前定型（失去在线增强的多样性），latent 的存储是 48 通道 × 16× 压缩，约 3.4 MB/样本（12 帧），17.9k 样本约 61 GB——比 707 GB 原始数据小一个量级，这是一个正向的附带收益。

新的优先级见 6.1–6.4（已按本节重排），统一的测试口径见 6.6。

---

## 一、口径、基线与证据分级

### 1.1 硬件与软件口径

| 项 | 值 |
| --- | --- |
| NPU | Ascend 910B2（Atlas A2，DAV_2201），单机 8 卡；双机 16 卡经 RoCE |
| Cube 峰值 | bf16/fp16 353.9 TFLOPS；HBM 1.6 TB/s |
| 片上内存 | L0A/L0B 各 64 KB，L0C 128 KB（fp32 累加），L1 约 480 KB 可用，UB 192 KB/AIV |
| 精度支持 | fp16/bf16/fp32/int8/int4b_t；**无 FP8/HiF8/MXFP8/MXFP4**（950 才有），INT8 与 FP16 同速 |
| 软件 | CANN 9.x + torch_npu 2.13；triton-ascend 未安装 |
| VAE | Wan2.2-TI2V-5B `AutoencoderKLWan`，48 通道 latent，16× 空间 / 4× 时间压缩，编码器通道 [160,160,320,640,640] |
| 主干 | 28 层，hidden 2048，MLP 9216（SwiGLU），16 Q 头 / 8 KV 头，head_dim 128；打包 AR 284 + DM 7104 = 7388 token（2 样本） |

一条必须写在最前面的口径警告：**草稿之间的参数量不自洽。** 28 层 ×（attention 约 0.88 M + MLP 约 37.7 M）本身已超过 3.369 B 的"总参数"，而按 hidden 2048 / MLP 9216 反推的可训练参数也不止 1.423 B。因此本文所有推导一律用**每层 FLOP、每 token 字节数、相对比例**，不引用绝对参数量；凡必须引用处都标注"草稿口径"。

### 1.2 测法与相位基线

单卡 2 样本一步 3.413 s，每段前后 `synchronize()`：

| 相位 | 时间 | 占比 | 设备占用 |
| --- | --- | --- | --- |
| pack（VAE 编码 + 提示词 + 打包） | 2.402 s | 70.4% | 97–104% |
| transformer forward | 0.217 s | 6.4% | 60–77%（含反向段整体） |
| backward | 0.475 s | 13.9% | 同上 |
| optimizer + zero_grad | 0.291 s | 8.5% | 28–43% |
| 合计 | 3.413 s | 100% | 每步真在算 3.09 s，约 91% 占用 |

单步 21,362 个 kernel，中位 15 µs，35.9% 短于 10 µs；conv 族 54.7%、matmul 7.5%、copy/pad/cat 11.9%、FlashAttention 4.6%、逐元素约 13%。host 侧 `aten::to` 花 2.35 s/步而对应设备时间仅 35 ms/步——这是全文唯一一个"现象已确证、调用点未定位"的项，优先级给 P0 是因为它的量级可能超过其他所有算子项之和。

### 1.3 证据分级

本文每个结论标一个等级，读者据此决定要不要先复测：

- **measured**：本机实测，附数字与出处草稿。
- **derived**：由实测数字按硬件公式推导（如 GM 流量 ÷ 1.6 TB/s、FLOP ÷ 峰值），推导过程可复算。
- **inferred**：基于官方文档或同类实现的推断，本机未验证。
- **speculative**：设计假设，需真机闭环。

凡"预期收益"没有标 measured 的，都不应当作承诺；第 6 节给每项配了验收口径。

本文的数字经过一轮红队复核（硬件/数值可行性 + 经济性两个视角）：P1、P2、P4、P6、T1 的收益按复核值下调，§3.3 的净亏收窄到 0.005–0.013 s/样本。**凡与早期草稿不同处，以本文为准**；复核中被直接否决的子项（如 P4 的"残差进 conv epilogue"）已从设计里删除，不再列为选项。

---

## 二、方法论：从 21,362 个 kernel 到可融合的算子链

### 2.1 六种动作，按成本从低到高

1. **换精度**：fp32 → fp16/bf16。VAE 已完成（2.46×），且 fp16 是唯一过精度门禁的选择（bf16 误差是配方量化噪声的 4.7–5.0 倍）。
2. **换布局**：ND ↔ NZ ↔ NC1HWC0，让搬运在 MTE2 里飞行完成，消掉 TransData。VAE 的 801 次 TransData 主要来自这里。
3. **融合逐元素链**：norm + rope + cast + residual，减少 GM 往返与 launch。收益 = 省掉的字节 ÷ 1.6 TB/s，加上省掉的 launch × 单次发射成本。
4. **融合到矩阵乘的 epilogue**：投影输出直接写成交错/归一化后的布局。910B 上 bf16 matmul+bias 没有官方融合算子（`npu_linear` 不支持 bf16，`npu_fused_matmul` 只有 950），所以这一层要么用 fp16，要么手写 Ascend C。
5. **换算法**：im2col Conv3D、单核双 mask 注意力、在线 softmax 合并。成本最高，只在第 1–4 项做完后还有余量时做。
6. **重叠与调度**：多流、通信与计算重叠、VAE 与 optimizer 重叠。不减少设备工作量，只回收 idle。

### 2.2 五类 bound 与判定

| bound | 判据 | 本文对应现象 |
| --- | --- | --- |
| Cube | 达成率 = FLOP ÷ (峰值 × 时间) | VAE Conv3D 约 41%；Transformer 前向约 45% |
| Vector | 逐元素/规约占设备时间 | VAE 逐元素 13%、Div 2.7% |
| MTE2（GM↔L1） | 有效带宽 ÷ 1.6 TB/s | TransData 17.4%（fp32）/ 8%（fp16） |
| MTE1（L1↔L0） | im2col 与 NZ 转换是否跟得上 Cube | Conv3D 的 baseK ≤ 64 KB/(baseM×2B) |
| Launch/host | kernel 中位时长与 host 派发时间 | 21,362 kernel/步、35.9% <10 µs；`aten::to` 2.35 s/步 |

**融合判据（本文的统一裁决规则）**：一条融合只有在同时满足下面两条时才进 P0/P1——

1. 省掉的 GM 流量 ≥ 该相位总流量的 5%，或省掉的 launch ≥ 该相位 launch 数的 10%；
2. 不引入新的 Cast/TransData，且反向可表达（训练路径）。

按这条规则，SwiGLU 融合（7.6 GB/前向，占前向流量约 12%）过第 1 条但收益绝对值只有 4.8 ms，属于"正确但不急"；two-way 交错消除（1.06 ms/前向，占 3.5%）两条都不过，直接不做。

### 2.3 Amdahl 的正确用法

相位占比决定上限：设某相位占 step 的 p，该相位内提速 s 倍，则 step 提速为 1 / ((1−p) + p/s)。

- 前向 p = 6.4%：即使前向快 2×，step 只快 3.3%。
- optimizer p = 8.5%：快 5×（291 → 58 ms），step 快 7.4%。
- pack p = 70.4%：假设快 1.5×（2.402 → 1.60 s），step 快 23%——**但这只是算术**：VAE 段设备占用 97–104%，算子层面没有 1.5× 的空间（3.0 的 kernel 剖面已说明），唯一能做的是把整段与别的相位重叠（5.3），而重叠的上限是整步 idle 0.32 s。

这就是为什么本文把算子工作的重心放在 optimizer 与调度重叠，而不是注意力融合。**当前向只占 6.4% 时，任何"只优化前向"的方案，收益上限都是 3.3%。**

---

## 三、VAE 编码前端：五项算子级创新

### 3.0 先看算子图景

fp32 单次编码 1.229 s 的 kernel 剖面（`kernel_details.csv`，4381 个 kernel）与 fp16 之后的分项预算：

| 族 | fp32 次数 | fp32 时间 | fp32 占比 | fp16 后占比（按 bf16 剖面） | fp16 折算秒数 | 归属 |
| --- | --- | --- | --- | --- | --- | --- |
| Conv3D | 233 | 0.681 s | 55.5% | 45.1% | 0.225 s | P3（或保持） |
| TransData | 801 | 0.213 s | 17.4% | 8.0% | 0.040 s | P2 |
| Pad | 227 | 0.063 s | 5.1% | 7.8%（含 Cat） | 0.039 s | P1 |
| Cat | 272 | 0.034 s | 2.8% | 同上 | 同上 | P1 |
| Div | 198 | 0.034 s | 2.7% | 2.7% | 0.0135 s | P4 |
| Transpose | 37 | 0.024 s | 2.0% | 2.0% | 0.010 s | P4 / P1 |
| 逐元素 | 1240 | 0.099 s | 8.0% | 约 13% | 0.065 s | 已融合一部分 |
| Conv2D | 45 | 0.015 s | 1.2% | — | — | — |

（fp16 的秒数是把 bf16 剖面占比乘到实测 0.499 s 上得到的**近似值**，不是逐 kernel 实测；证据等级 derived。**口径警告**：本表的 0.499 s 是单次编码口径，而 3.1/3.4/3.5 里的百分比用的是 12 帧窗口的 0.374 s/样本 基线（0.748 s ÷ 2 样本），两者不是同一配置，引用时不要互换；RMSNorm 融合省下的 0.179 s 也是 12 帧窗口口径。）

三项结构性事实决定了后面所有设计：

1. **时间集中在 level 0/1**：down_blocks.0 占 44.6%、down_blocks.1 占 26.3%，两级合计 71%；mid 与 down_blocks.3 已是尾数（6.2% / 5.8%）。任何需要手写内核的方案，只可能在 level 0/1 回本。
2. **搬运占比与卷积数学同量级**：TransData + Pad + Cat + Div + Transpose = 30.0%（fp32）/ 20.5%（fp16）。这解释了为什么 fp16 只拿到 2.46× 而不是 4×——矩阵乘快了，搬运没同比变快。**但"占比"不等于"可回收"**：搬运里只有 patchify 与 conv_in 的输入转换能在入口 kernel 消掉；227 次 Pad 与 272 次 Cat 的大头绑在各自的卷积上，只有手写卷积（P3）才能拿走（见 3.1 的归因边界）。
3. **单卡内部没有调度空隙**：设备占用 99.9%、kernel 首尾相接。所以 VAE 侧的多流只在"与别的相位重叠"时有意义（见第五节），在编码内部没有可回收的 idle。

### 3.1 P1：数据入口一体化——uint8 + 归一化 + patchify + 填充写进一个 kernel（P1）

**现状。** 视频以 fp32 进 H2D（每样本 [1,3,33,544,736] = 158.6 MB），到设备后 `patchify` 先做一次 7 维 permute + contiguous，接着每个 `WanCausalConv3d.forward` 都做一次 `F.pad`（227 次 Pad），有 cache 时还要 `torch.cat([cache_x, x], dim=2)`（272 次 Cat）。`F.pad` 同时是 `channels_last_3d` 被平台挡住的原因：`constant_pad_nd` 只接受 contiguous/preserve 格式。

**必须先说清楚的归因边界。** 这 227 次 Pad 与 272 次 Cat 是**每个卷积各一次**，分别属于各层自己的输入填充与 cache 拼接；P1 只把**第一层 conv_in** 的那一次 pad/cat 与 patchify 合并掉。其余 226 次 Pad 与 271 次 Cat 只有手写卷积（P3，把 halo 折进 im2col 的 L1 偏移）才能消掉。因此下面"收益"一节只按 conv_in 这一层记账——早期草稿把全链 Pad/Cat 的 0.109 s 全算给入口 kernel，是错的（红队复核，高估约 5–9×）。

**设计（一条 SIMT kernel，形态参照 DeepInfer 的 `rgb_pad_to_patches_tl`）。** 输入是 uint8 的 `[B,3,T,H,W]`（HWC 内层，DataLoader 直接发 uint8），对每个输出元素反查 patchify 的源像素：

```
输出通道 c = 4*cc + 2*j + i        // i 为 H 相位，j 为 W 相位（与官方 permute(0,1,6,4,2,3,5) 一致）
源像素 u = X[b, cc, t-2, 2h-2+i, 2w-2+j]
fp32 计算 v = float(u) * (1/127.5) - 1，再一次性 cast 到 fp16
写入预分配的 [B, T+2, H/2+2, W/2+2, 16] fp16 缓冲
```

写入的缓冲区是 **NDC1HWC0 语义**（C1=1、C0=16）：12 个有效通道 + 4 个零 lane，空间边界与时间前 2 帧保持 0，因此下游不再调用 `F.pad`，也不再需要 `feat_cache` 的 clone+cat——后续 chunk 直接从 uint8 源重读 `t0-2`/`t0-1`。缓冲按 `(B,T,H,W,dtype)` 缓存复用，只写内区。

**索引映射已验证**：在纯 Python 上按上面的反查公式构造 buffer，再与 `patchify(F.pad(((u/127.5)-1)))` + conv1 逐元素比，`max|Δ| = 0`（要求 H、W 为偶数；奇数会被 patchify 丢掉最后一行/列，本模型的 544×736 不受影响）。注意时间索引是 `t-2`、空间是 `2h-2+i`，与 3.3 的 `t+dt−2` 是同一套偏移，写成 `t-1`/`2h-1` 会整体错位。

**为什么补齐到 16 lane 不增加卷积代价。** `LoadData3DParamsV2` 要求 `NC1HWC0` 且 C0 固定为 16；conv_in 的 12 个通道本来就只能按 16 lane 处理。所以"12 有效 + 4 零"与硬件行为一致，唯一代价是入口 kernel 的写放大 `16/12 = 1.33×`（约 0.002 s），换来的是 conv_in 的输入转换消失。

**归一化零点的语义必须写进契约。** 归一化后的 0 对应 uint8 的 127.5，uint8 域不可表示。填 127 会得到 −3.92e-3、填 128 会得到 +3.92e-3，边界帧变成**系统性偏置**而不是随机噪声。所以补零只能在 fp16 域做（缓冲区先 `zeros_`，只写内区）。

**收益（只算 conv_in 这一层）。** patchify 0.012 s（fp32，1.0%）→ fp16 约 0.006 s；conv1 自己的 1 次 Pad + 1 次 Cat ≈ 0.0003 s；conv_in 输入的一次 TransData ≈ 0.001–0.002 s。合计 **0.005–0.010 s/样本（1.5–2.7%，相对 0.374 s 的 fp16 基线）**。附带收益不计入秒数但真实存在：H2D 从 158.6 MB 降到 39.6 MB/样本（4×，按 25 GB/s 约省 5 ms/样本）、`/dev/shm` 压力降到 1/4、patchify 的 7 维 permute 从图里消失（对图捕获友好）。

**它为什么还值得排在 P1。** 秒数不大，但它是唯一能同时改善 H2D、共享内存与图捕获的入口改造，也是 P3 的布局前提（P3 的 down0 若要用 NDC1HWC0 输入，入口就得先写出这种布局）。若只按秒数排序，它应该排在 P4 之后。

**风险与验收。**

1. 归一化必须在 fp32 完成再 cast，否则比现路径多一次舍入；
2. 因果语义逐位对齐：`t0<2` 的 chunk 用零帧，其余 chunk 从源帧重读，不能依赖 `feat_cache` 的快照；
3. 缓冲形状随 chunk 宽度变化要重建并重新 `zeros_`；
4. 若后续全链路改 NDHWC，本 kernel 是唯一需要重写的地方（其余只是 dtype/布局参数）。

验收：与参考实现 `patchify(F.pad(((u.float()/127.5)-1.0).half()))` 逐元素比，`max|Δ| ≤ 1` 个 fp16 ulp（预期 0，因为 fp32 运算顺序一致）；再走 64 个真实 DROID 窗口的 latent 门禁（≤0.64× bf16 量化噪声）；profiler 里 **conv_in 前后的 Pad/Cat 归零**（全链计数只下降 1/227 与 1/272，不要期待总量归零）。

### 3.2 P2：TransData 逐项归因与冻结权重预转（P2）

**归因。** 801 次 TransData ≈ 每个 Conv3D 3 次（输入、权重、输出）+ 逐元素前后的格式转换。三类的可消除性完全不同：

| 来源 | 估计次数 | 能否消除 | 做法 |
| --- | --- | --- | --- |
| 权重 ND→NZ | 约 233 | 能（VAE 冻结） | 加载时一次性预转，随 checkpoint 存 |
| 输入 ND→NC1HWC0 | 约 233 | 部分 | 让激活全链路保持卷积偏好的布局，避免"算完就转回 ND" |
| 输出 NZ→ND | 约 233 | 能（L0C→GM 的 NZ→ND 免费） | 直接按 ND 写出，不让框架再插一次转换 |
| 逐元素/reshape | 约 100 | 部分 | 与 P1/P4 融合 |

**三步推进。**

1. **归因**：用 `op_summary` 把 801 次 TransData 按 shape 与 storage format 分成三类（输入 `ND/NCDHW→NC1HWC0`、权重 `NCDHW→FRACTAL_Z_3D`、输出 `NC1HWC0→ND`），并标出有多少已经在 MTE2 飞行里完成、不算独立 kernel。
2. **权重预转**：VAE 冻结，初始化时对每个 Conv3d 权重做一次 `NCDHW→FRACTAL_Z_3D`（shape `(D*C1*H*W, N1, N0, C0)`，fp16 下 C0=N0=16），运行时直接复用。失效条件必须写成断言：解冻微调、换 dtype 或重载 checkpoint 时缓存作废。
3. **输入直落**：P1 的缓冲本来就按 NDC1HWC0 写出，conv_in 的输入转换因此消失。深层 conv 的输入转换只有把整条链改成 NC1HWC0（norm 要在 C1/C0 两段上归约）才能消掉，作为第三阶段、按归因结果决定，**默认不做**。

**收益（按字节重算，不按次数）。** 801 次 ÷ 233 个卷积 = 3.44 次/卷积，但**次数份额不等于时间份额**：权重张量按字节远小于激活（C=640 的 3×3×3 权重也只有 22 MB，全部 VAE 权重合计约 0.3 GB；激活侧每次编码是 GB 级），而 TransData 的 0.040 s 主要由大张量搬运贡献。权重预转真正省的是 233 次小 TransData 的 launch（每次约 2–5 µs）加约 0.3 GB 的重复搬运（约 0.2 ms），合计 **0.002–0.005 s/样本（0.5–1.3%）**；conv_in 输入直落再贡献 0.001–0.002 s。**0.020–0.025 s 是错的**——它把"次数占 1/3"当成了"时间占 1/3"（红队复核，高估约 5–9×）。

**这项的真正价值是归因，不是收益。** 第 1 步的 `op_summary` 按 shape/format 分类，会直接回答"TransData 到底花在输入、输出还是逐元素"——这个答案决定第 3 步（全链路 NC1HWC0）要不要做。若归因显示激活侧（输入/输出）占大头，那么结论就是"要么全链路改布局，要么不做"，而不是继续抠权重。

**最大不确定性。** `aclnnConvolution` 的 PyTorch 入口可能不认已转格式的 fp16 权重（公开文档只给了 quant conv 的 WeightNz 接口），若算子内部再转一次，收益归零。**必须用 kernel 计数验收，不能用理论推断。**

**为什么它比 P3 优先。** 同样是省时间，P2 是布局与缓存问题，不改数学；P3 要手写 Ascend C 的 im2col 卷积。先做 P2，再决定 P3 是否有必要。

### 3.3 一个被硬件否决的优雅方案：patchify + conv1 的代数融合

**先给结论：不作为生产路径。** 这条恒等式在数学上严格成立，但在 910B2 上把 conv 的 issued K 从 432 抬到 1728（4×）。conv_in 这一层在 fp32 下总共只有 0.015 s，4× 的增量上界约 0.018 s；再减掉省下的 patchify（约 0.005 s），**净亏约 0.005–0.013 s/样本**。保留它的理由有两个：它是 3.1 那条入口 kernel 的数值 oracle，也是"用 issued-K 算术否掉漂亮代数"的方法论样板。下面先给完整推导，再给裁决。

**命题。** 设 patchify 为

```
p[b, c*4 + r*2 + q, t, h, w] = x[b, c, t, 2h + q, 2w + r]        (q, r ∈ {0,1})
```

（通道序取自 `cosmos_framework/model/generator/tokenizers/wan2pt2_vae_4x16x16.py` 的 `permute(0,1,6,4,2,3,5)`，即 `(c r q)`，`q` 是最快的通道位。）编码器首层是 `conv1 = CausalConv3d(12, 160, 3, padding=1)`。把它展开：

```
y[b,o,t,h,w] = Σ_{c,dt,dh,dw,q,r} W[o, c*4+r*2+q, dt, dh, dw] · x[b,c, t+dt−2, 2(h+dh−1)+q, 2(w+dw−1)+r]
```

令 `H' = 2dh − 2 + q ∈ [−2, 3]`、`W' = 2dw − 2 + r ∈ [−2, 3]`，则

```
y[b,o,t,h,w] = Σ_{c,dt,H',W'} Wf[o, c, dt, H', W'] · x[b, c, t+dt−2, 2h+H', 2w+W']
Wf[o, c, dt, H', W'] = W[o, c*4 + (W'%2)*2 + (H'%2), dt, H'//2, W'//2]
```

也就是说：**patchify + conv1 ≡ 一个 `Conv3d(3, 160, kernel=(3,6,6), stride=(1,2,2))`，空间补 2、时间只在左侧补 2 帧。** 权重构造：

```python
# W: [160, 12, 3, 3, 3]（conv1.weight）
# Wf: [160, 3, 3, 6, 6]
Wf = torch.empty(160, 3, 3, 6, 6, dtype=W.dtype)
for hp in range(6):
    for wp in range(6):
        q, r = hp % 2, wp % 2                  # patch 内的高/宽偏移
        dh, dw = hp // 2, wp // 2              # 原 3×3 核的高/宽偏移
        c_src = torch.arange(3) * 4 + r * 2 + q  # patchify 的通道序 c*4 + r*2 + q
        Wf[:, :, :, hp, wp] = W[:, c_src, :, dh, dw]

# 前向（x 为 [-1,1] 的像素，时间维左补 2、空间补 2）
y = F.conv3d(F.pad(x, (2, 2, 2, 2, 2, 0)), Wf, stride=(1, 2, 2))
```

**为什么它在数学上是"免费"的。** 融合后的 im2col K 维是 `3×3×6×6 = 324`，与原来的 `12×3×3×3 = 324` **完全相同**；有用 MAC 数、`baseK ≤ 64 KB/(baseM×2B)` 的预算、L0A/L0B 分块全部不变。变的只有地址映射：原来读 12 个通道的同一像素块，现在按 stride-2 读 3 个通道的 6×6 邻域。已在纯 Python 上做过等价性验证：`max|z − z_fused| = 3.55e-14`。

**如果它在硬件上成立，会省下：**

| 项 | 量 | 来源 |
| --- | --- | --- |
| patchify kernel | 0.012 s（实测） | draft §三 按层 |
| patchify 输出的写 + conv1 的读 | 79.3 MB × 2 = 158.6 MB ≈ 99 µs | derived，1.6 TB/s |
| conv1 输入的 TransData | 一次 12 通道 → 卷积格式 | derived |
| 输入字节数 | 12 通道 patch vs 3 通道原图，读放大 4×→1× | derived |

合计上界约 0.010–0.020 s/样本（fp16 口径；其中 patchify 本身只有约 0.006 s，0.012 s 是 fp32 实测）。

**硬件裁决（关键）。** Ascend 的 im2col K 维不是数学上的 324，而是 `C1 × C0 × kt × kh × kw`，其中 C0 固定为 16（b16），`LoadData3DParamsV2` 的 `kStartPt`/`kExtension` 必须是 16 的倍数：

| 路径 | 有效 C_in | issued K | 有用 MAC/输出 | Cube 浪费 |
| --- | --- | --- | --- | --- |
| 原路径 conv_in | 12（补到 16） | 16 × 27 = 432 | 324 | 1.33× |
| 融合路径 | 3（补到 16） | 16 × 108 = 1728 | 324 | 5.33× |

Conv3D 全族在 910B2 上是 cube-bound（预算表显示它比等价 GEMM 更 cube-favorable），但 conv_in 这一层 C_in 只有 12、C_out=160、M 又大，未必能跑满 Cube——若它实际是 MTE 受限，4× issued MAC 的惩罚会小于 4×（增量下界约 0.005 s，上界约 0.018 s）。无论取哪一端，都没有任何理由指望它变成收益：用上界 0.020 s 的收益去换至少 +0.005 s 的卷积时间，**净亏 0.005–0.013 s**。

**所以正确做法是 3.1 的 P1**：入口 kernel 写出 patchify 后的 12 通道（补到 16 lane），conv_in 保持 C_in=12。恒等式仍然有用——它的源像素反查公式就是 P1 内核的地址映射，它的权重置换可以用来在 CPU 上做 oracle 对拍。

**作为 oracle 保留时仍需注意。**

1. **时间维因果性**：融合核的时间窗仍是 `t−2..t`（左侧补 2 帧、kernel 3），与 `_padding=(W,W,H,H,2T,0)` 一致；分块编码的 cache 语义不变（patchify 逐帧，不跨时间）。
2. **1 帧 prime**：官方 `encode` 先编 1 帧 prime 再按窗口编剩余帧；oracle 不改变这个流程，只改变首层输入布局。
3. **权重预处理的位置**：`Wf` 只有 `160×3×3×6×6×2B ≈ 104 KB`（fp16），CPU 对拍时生成一次即可。
4. **反向**：VAE 冻结，不需要 `Wf` 的反向；若将来解冻，`Wf` 对 `W` 的映射是纯置换，梯度按同一索引反向散射即可。

### 3.4 P3：为 down0/down1 手写因果 Conv3D（P2 研究项）

**现状达成率。** 卷积部分 fp32 实测 0.681 s、约 33 TFLOP，折合 49 TFLOPS；fp16 折算 0.225 s ≈ 147 TFLOPS，是 bf16 峰值 353.9 的 **41%**。仍有约 2.4× 的理论头寸，但拿走它要换掉 aclnn 的 Conv3D 调度。

**预算账（按真实通道 160/320/640，不是通用示例）。**

| 层 | C_in | baseK 分段 | MMAD/tile | L1/级 | 流水级数 | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| down0 | 160 | 单段 160（≤256） | 27 | 57.6 KB 特征 + 40 KB 权重 = 97.6 KB | 4–5 | 值得 |
| down1 | 320 | 2 × 160 | 54 | 115.2 + 40 = 155 KB | 3 | 值得 |
| down2/down3/mid | 640 | 4 × 160 | 108 | 230.4 + 64 = 294 KB | 1 | 不做（M 只有 1.9 万 / 4.7 千） |

- im2col 走 MTE1 `LoadData3DParamsV2`（v1 在 A2 不支持），L1 常驻 NC1HWC0 特征 tile（含 3×3 空间 halo）+ `FRACTAL_Z_3D` 权重 tile；时间维 3 个 tap 用外层循环 + L1 内偏移，因果 pad 由 P1 的缓冲提供，内核不再处理 T 维补零。
- C_in=160/320 都是 16 的整数倍，K 没有浪费（这点与 conv_in 的 12 通道不同）。
- 稳态预算 98.2% 来自 npu-guide 的 C_in=384 / 81-chunk 配置（42,240 cyc/tile）；down0 是 27 个 chunk，按同一模型约 14,600 cyc/tile——**比例不变，但不要拿 42,240 当 down0 的绝对数**。首版手写取 65–75%。

**收益（红队修正）。** 41% 是全族平均；down0/down1 的 C_in=160/320 都是 16 的整数倍、通道又不小，本来就比平均值高效。手写版还有一个结构劣势：baseN=128 的权重 tile 对 C_out=160 只用到 62.5%。把这两点算进去，down0+down1（占 conv 时间约 65% = 0.146 s）的合理区间是 **0.02–0.05 s/样本（5–13%）**，即 0.04–0.10 s/步（1.2–3%）；乐观（75% 且覆盖 down2）也到不了早期草稿写的 0.09 s。**inferred**，必须先按层 msprof 拿到单层 cube 效率再排期。**4–8 周换整步 3%，这是本文性价比最差的一项**——它排在 P2 不是因为它值钱，而是因为它的前置条件（P1/P4 做完）还没满足。

**风险。**

1. 因果语义最易错：首 chunk 的 2 帧零填充、跨 chunk 的 cache 必须与 diffusers 逐位对齐；
2. 性能押在 L2 命中：预算表显示 L2 只给 1.5× 就少 21%，掉到纯 HBM 慢 80%；
3. halo 的空间边界语义（`padList` 只覆盖 H/W，时间维要自己偏移）必须单测；
4. 工作量 4–8 周，且一旦 aclnn 在某层本来就很高效，收益会被高估。

**替代路线。** TileLang-Ascend 目前没有已合并的通用 conv：Conv2D 示例是 host 侧 im2col + GEMM；原生 im2col（PR #1759）与 Conv3D 反传（PR #1738，报告 0.04× 即慢 25 倍）都还 open。所以"用 TileLang 写 Conv3D"在 2026-11 的仓库状态下不是一个可交付选项，真要做得用 Ascend C。**建议先完成 P1/P2/P4 并复测，只有 down0/down1 仍是瓶颈、且整步的其他大头（多流、优化器、host）都已经拿掉时才启动 P3。**

### 3.5 P4/P6：逐元素与 norm 的残余（P1/P2）

RMSNorm 融合已把 52 个 norm 的 6 遍全尺寸读写降到 2 遍（0.927 → 0.748 s，1.24×）。剩下三块（另有一块被硬件否决），按实测 kernel 计数逐项做：

| 项 | 现状 | 做法 | 预期 |
| --- | --- | --- | --- |
| (a) norm + SiLU | 融合 norm 后仍是 `x*scale` → `mul_(gamma)` → `silu` 三遍 | 合成一个 kernel：1 读 1 写，归约留 fp32 | 0.005–0.010 s |
| (b) 残差进 conv epilogue | **910B 上不成立，删除**：FixPipe 没有 tensor residual 输入，L0C 也没有到 UB 的通路，残差只能在 conv 之后单独加 | — | 0 |
| (c) AvgDown3D | `pad→view→permute.contiguous→view→mean`，含一次全尺寸 permute 拷贝 | 换成 strided-mean kernel，直接写 `[C_out,T/2,H/2,W/2]` | 0.002–0.004 s |
| (d) 去掉 `x.clone()` | `WanResidualDownBlock` 的 `x_copy = x.clone()` | AvgDown 只读不写 | 0.001–0.002 s |
| **合计** | | | **0.008–0.016 s（2–4%）** |

**P6：在真正的 NDHWC 布局下用 `npu_rms_norm`。** 数学恒等：`F.normalize(x, dim=1) * sqrt(C) * gamma ≡ x / sqrt(mean_c(x²) + eps) * gamma`，因为 `mean_c(x²) = ||x||²/C` 且 `scale = sqrt(C)`。把 3 个算子压成 1 读 1 写，52 个 norm 每个少 3 次 launch。**但收益要按红队口径重算**：现有的融合 norm 已经是 1 读 1 写（0.748 s），P6 在同一段 affine 链上与 P4(a) 重叠，剩余可省的只有 launch 与一点中间量——**0.002–0.008 s/样本（0.5–2%）**，不是早期草稿的 0.015–0.025 s。

**但这个前提与 P1 的接口有一个必须核对的冲突**：P1 写出的是 `NDC1HWC0`（C1=1、C0=16，12 有效 + 4 零 lane），而对 C=160 的深层 norm，`NDC1HWC0` 的通道被切成 10 个 C0=16 的块，最后一维不是完整的 160——`npu_rms_norm` 无法在最后一维上一次归约。若强行把 16 lane（含 4 个零）当分母，RMS 会被缩小 `sqrt(12/16)`，不是恒等。所以 **P6 只在"整条链路是真正的 NDHWC、且 norm 段通道连续"时成立**；如果 P1 停留在 conv_in 的 NDC1HWC0，P6 只能用于 conv_in 之后的第一个 norm，或者先做一次通道重排（把省下的搬运又加回去）。结论：P6 作为 P1 之后的**布局专项**评估，不作为默认后续。

**其余碎项的去向。** (c) 的 strided-mean 吃掉一部分 Transpose（37 次 / 2.0%，单次 658 µs fp32，是单次最贵的非卷积 kernel）；剩下的 Div 198 次 / 2.7% 要先定位调用点（融合 norm 已替换，说明它在别处），不要盲目全局替换 `div`——`clamp_min(1e-12)` 的语义必须保留。

**验收**：每项独立开关，用 `profile_vae_norm.py` 同口径（batch 2、12 帧、1 预热 + 5 次）比墙钟与峰值；latent 相对误差门禁不变；profiler 确认 Silu/Transpose/Add 的 kernel 计数按预期下降。

### 3.6 可复用的 DeepInfer 内核（P1–P4 的起点）

`deepinfer/model_executor/layers/tilelang_ops/asc/` 里已有三个结构上可直接借用的前向内核，全部是 TileLang + Ascend 后端：

| 内核 | 它做了什么 | 与本文的关系 |
| --- | --- | --- |
| `rgb_pad_to_patches_tl` | 填充 + uint8 读 + 归一化 + bf16 patchify，SIMT 直接读写 GM | **P1 的现成骨架**：把输出从"bf16 patch 张量"改成"NDC1HWC0 padded buffer"，并在同一内核里完成归一化与填充，就是 P1 的实现起点 |
| `norm_tl` | 残差 + RMSNorm + 仿射 + FP8 量化 | VAE 的 RMSNorm 融合已在 PyTorch 层拿到 1.24×；若要把剩余 2 遍读写也收掉，可借它的 tile 结构，把 FP8 量化段换成 bf16/fp16 写回 |
| `batch_downsample_tl` / `gather_tl` | 批量下采样 / 重排 | P2 的布局改造与 P4 的 Transpose 消除可参考其地址映射写法 |

两条限制必须写在前面：这些内核**只有前向**（与 VAE 的 `no_grad` 场景匹配，但不能直接搬到 Transformer 训练路径，见 4.6）；它们面向推理的 KV cache/量化语义，移植时要删掉无关分支，不要为了复用而保留死代码。

npu-guide 的 `templates/08_video_frontend.py` 另外给了"按 W 方向连续行带切 tile"的 Conv3D/patchify 写法与每级 UB 预算，可作为 P1 内核的骨架；它的结论也一致：**Conv3D 是 TileLang 最可能需要手写 Ascend C 的算子族**（不存在 Ascend 原生的 im2col 原语）。

### 3.7 mid-block attention：上限只有约 2 ms，别再做

mid block 总计 0.078 s / 6.2%（**fp32 口径**，分母是 1.229 s；按 fp16 折算约 0.031 s），但**注意力本身只占编码器的约 0.3%（≈0.002 s fp16）**，其余是 2 个 C=640 残差块的 4 次 Conv3D。`npu_fusion_attention` 的约束（D≤896、Nq 1–256）在本处都满足（D=640、单头、1564 token），但换来实现的是同一份 O(n²d) 计算，varlen 还需要 host 列表（`actual_seq_qlen`），反而破坏图捕获。**真正的 mid 瓶颈是 C=640 在 M≈4,692 时的并行度与 L1 只够 1 级流水，不在 attention——而 P3 已经明确不做这一层。结论：不动。**

### 3.8 VAE 的不可行清单（明确不做）

| 方案 | 为什么不做 | 证据 |
| --- | --- | --- |
| FP8 编码 | 910B2 没有 FP8 Cube（Matmul dtype 表 A2/A3 行不含 fp8/hifloat8；`asc_enable_fp8` 仅 950） | documented |
| bf16 编码 | 64 个真实窗口误差是配方 bf16 量化噪声的 4.67–5.01 倍，越过 1.0× 门禁 | measured |
| 直接 `channels_last_3d` | `F.pad(constant_pad_nd)` 报 `Only contiguous_format or preserve_format is supported`；必须先去掉 F.pad（即 P1） | measured |
| torch.compile（当前环境） | inductor 需要 triton-ascend，本容器未安装（**环境缺失，可装**，装上后需重测）；npugraph_ex 稳态慢 18× 是 397 次 `aten::item` 切图，属于图捕获问题，不是 inductor 的锅 | measured（环境）/ measured |
| 24 帧分块 | batch4 下只有 1.06×，显存 1.7–2×；16/28/32 帧还回退 18–32% | measured |
| 预提取 latent 绕过编码 | **已从"不可行"移出**：2026-11-09 起是候选训练配方（见 0.4）；剩下的取舍是增强必须在编码前定型，失去在线增强的多样性 | documented |
| VAE 内部再开多流（chunk 间流水） | 编码器设备占用 99.9%、4381 个 kernel 首尾相接，没有气泡；chunk 之间是波前依赖（chunk i+1 的第 L 层依赖 chunk i 的第 L 层），叠流只改顺序不减工作量 | measured |
| 把 VAE 摊到专门的几张卡（数据流水线并行） | 5:11 拆分时 5 张编码卡约 10 samples/s，11 张训练卡（g=1）约 6.9、g=32 摊薄后约 9.6，与全 16 卡在线编码的约 9.7 持平；没有空闲卡时是零和 | derived |

## 四、Transformer 训练步：融合边界与 two-way kernel

### 4.0 先把"值多少"算清楚

每层前向的 FLOP 构成（按 7388 token、hidden 2048、MLP 9216、16/8 头、head_dim 128）：

| 分量 | GFLOP/层 | 占比 |
| --- | --- | --- |
| MLP（gate/up/down，SwiGLU） | 836.66 | 57.6% |
| attention（QK^T + PV） | 430.28 | 29.6% |
| QKV 投影 | 123.95 | 8.5% |
| 输出投影 | 61.98 | 4.3% |
| 合计 | 1452.87 | 100% |

28 层合计 40.68 TFLOP，实测前向 0.217 s → **187.5 TFLOPS，约 53% 峰值**。这个数字很重要：它说明前向并不是"效率很差"，可拿的头寸只有 47%，而前向只占一步的 6.4%——**即使把前向做到峰值，一步也只快 3%**。这就是为什么本节的所有融合都标 P2。

反向 0.475 s 是前向的 2.19×（含重计算），优化器 0.291 s 单独讨论（4.1）。

### 4.1 T1：优化器是训练侧最大的单项（P0）

**现状。** optimizer + zero_grad 实测 0.291 s（1 卡 msprof 口径；本次 1 卡复测 0.264 s、8 卡 0.332 s，见 6.1.2），设备占用只有 28–43%——典型的**发射受限**，不是带宽受限。按 1.423 B 可训练参数（实测：322 个张量、1,423,383,232 元素）算带宽账：

| 张量 | 全量字节 | ZeRO-2 shard=8 每 rank |
| --- | --- | --- |
| fp32 master（读 + 写） | 11.38 GB | 1.42 GB |
| Adam m（读 + 写） | 11.38 GB | 1.42 GB |
| Adam v（读 + 写） | 11.38 GB | 1.42 GB |
| bf16 梯度（读） | 2.85 GB | 0.36 GB |
| bf16 参数（写） | 2.85 GB | 0.36 GB |
| 合计 | 约 39.8 GB | 约 5.0 GB |

ZeRO-2 shard=8 时每 rank 只更新 1/8 的参数，per-rank 流量约 5 GB → **3.1–3.5 ms**（不切分的单卡下限是 26 ms）。口径必须对齐：**291 ms 是单卡实测，应当与单卡下限 26 ms 比，约 11×**；16 卡上 ZeRO-2 实际是 shard=16（约 2.5 GB/rank、1.6–1.9 ms），这是另一个口径，不能拿 291 ms 直接去除（早期草稿写"90×"就是混了口径）。差距来自"逐张量、逐算子"的发射模式：约 322 个可训练参数张量（实测 317 + 5，草稿曾写 644）× 每个约 10 个逐元素 kernel ≈ 3.2k 次 launch，按实测每次约 50 µs，仅 AdamW 本体就有约 140 ms——**这个次数不随分片数缩小**，所以下面的收益在 shard=8 与 shard=16 下都成立。

**真机修正（2026-11-09，node1，8 卡 ZeRO-2，见 6.1.2）。** 上面几个 derived 数字按实测改写：

- 张量数：644 → **322**（checkpoint 可训练张量 317 + 5 个 5× lr 组；全模型 549 个张量、3.37B 参数）。
- 段时长：291 ms（草稿口径）→ **8 卡实测 332 ms**（HybridAdam，含 `zero_grad` 与 step 内的参数 all-gather）。
- "只换 `NpuFusedAdamW` 到 90–110 ms" 第一轮**未达到**（203 ms），第二轮**达到**：在 AdamW 之外把 grad-norm、grad→fp32、master→bf16 也换成多张量路径后，段 **104–109 ms**（见 6.1.3）。AdamW 本体按算子表 138 → 13 ms（−125 ms 预测 vs −129 ms 实测）；grad-norm 63.8 → 4.1 ms；cast/回写 24.5 → 约 2.3 ms。段内剩余约 100 ms 的构成见 6.1.3 的 event 归因。
- "单卡下限 26 ms" 保留为带宽账的理论值；1 卡实测段为 HybridAdam 264 ms / fused 232 ms——1 卡下张量平均 4.4M 元素，逐张量开销占比反而小，fused 只省 32 ms。

**设计（按收益/成本排序）。**

1. **换 `torch_npu.optim.NpuFusedAdamW`**：`torch.optim.AdamW` 的 drop-in 替换，A2/A3 支持。它把每个分片的 `mul/add/sqrt/div` 链收成一个 kernel（322 个分片 → 322 个 kernel）。**实测 8 卡段 332 → 203 → 104–109 ms**（6.1.2/6.1.3），AdamW 本体 138 → 13 ms；剩下的 190 ms 里有 95 ms 由下面 2–3 项（多张量 cast/grad norm）在第二轮拿掉，残余约 100 ms 见 6.1.3 的 event 归因。
2. **多张量 apply**：`npu_apply_fused_adamw_v2`（torch_npu/MindSpeed 侧的多张量版本）一次处理 7 个张量（param/grad/exp_avg/exp_avg_sq/max_exp_avg_sq 等），把"每张量一次"变成"每 7 个张量一次"，进一步逼近带宽下限。若 ZeRO 把参数切成分片，分片数与张量数的组合要在启动时固定，避免每步重排。
3. **梯度范数与裁剪合并**：~~先在设备上把所有分片的平方和累到一个标量缓冲（一个 kernel），一次 all-reduce，再由 fused apply 读取该标量做 unscale/clip；不要每张量一次 `all_reduce`（那是 N 次小通信）。~~ **已落地**（6.1.3）：322 个分片一次 `torch._foreach_norm`（63.8 → 4.1 ms），块结果在 float64 里求和；unscale/clip 走 `_foreach_mul_`（432×400k 时 10.65 → 1.43 ms）；全局范数仍是一次标量 all-reduce。
4. **`zero_grad(set_to_none=True)`**：省掉 memset；若下一步立刻覆盖梯度，可把清零折进梯度累加的首个 kernel。

**收益。** 实测（8 卡 ZeRO-2）：332 → 203 → **104–109 ms**（两轮合计 −223 ms）；AdamW 本体已只剩 13 ms，残余约 100 ms 主要是 bf16 参数 all-gather 与 fused 本体（见 6.1.3）。这是全文训练侧最大的单项，成本却只是换优化器类、复用两组 cast buffer、换一次 grad-norm 调用，加一轮数值对拍。

**风险与验收。**

- ZeRO/HSDP 的分片参数是 `torch.Tensor` 视图，fused 优化器必须支持非连续/分片视图；先在单卡全参数上验证，再上 ZeRO。**不能把 322 个分片再拉平成一个 buffer**——那会破坏 ZeRO 的 state 布局与 `save_optimizer(shard=True)` 的 checkpoint 契约，而按分片调用已经能拿到绝大部分收益。
- 数值：bf16 参数在 fused kernel 内可能被提升到 fp32 再写回；要求"一步更新后参数最大绝对差 ≤ 1e-6（相对 fp32 AdamW）"，并跑 3 步 loss 曲线。
- action head 的 5× lr 必须留在独立 param group；grad clip 的标量 all-reduce 把各 group 拼成一个张量一次发。
- 验收：`msprof` 的 optimizer 段设备占用从 28–43% 提到 70% 以上，kernel 数从约 3.2k 降到约 322，段时长 ≤ 90–110 ms。**第一轮 203 ms，第二轮 104–109 ms，验收达到**；残余约 100 ms 的构成见 6.1.3。

### 4.2 T2：前向/反向的逐元素融合（P2，收益封顶）

按 2.2 的判据逐条算（bf16，每层 7388 token，激活 30.26 MB）：

| 链 | 现状 | 融合后 | 省/前向 | 整步（含反向/重算） | 结论 |
| --- | --- | --- | --- | --- | --- |
| SwiGLU | gate/up 两个 GEMM + 3 遍逐元素（7 遍 GM 流量） | 一个 N=18432 GEMM + `npu_swiglu`（5 遍） | 7.6 GB = 4.8 ms | 约 0.017 s | P1（顺手） |
| RoPE（5 张量） | 5 条 `q*cos + rotate_half*sin` 手工链，10.3 GB/前向 | 5 次 `npu_rotary_mul`，2.6 GB | 7.7 GB = 4.8 ms | 0.02–0.03 s（含 420 launch/方向） | P2 |
| QK-norm（5 次） | 5×5 个 kernel | 5 个 `npu_rms_norm` | 流量不变，省约 40 launch/层 | — | P2（与 RoPE 同批） |
| norm + residual | add 写 x、norm 再读 x | `npu_add_rms_norm` 只写 x/y | 1.69 GB = 1.06 ms | 约 0.005 s | P2 |
| **合计** | | | | **0.03–0.05 s（0.9–1.5%）** | |

这与 4.0 的结论一致：Transformer 融合是正确的工程，但当前不是杠杆——**整步不到 1.5%**。若要做，按下面四条约束落地：

1. **SwiGLU**：把 `gate_proj` 与 `up_proj` 的权重沿输出维拼成 `[18432, 2048]`（gate 在前），一次 GEMM 得 `[T,18432]`，再调 `npu_swiglu(x, dim=-1)`；反向是 `npu_swiglu_backward(grad, x, dim=-1)`，它需要 pre-activation，SAC 重算路径必须产出同一份 `[T,18432]`。**顺序不能拼错**（`act(gate)*up`），否则静默训坏。
2. **RoPE**：`npu_rotary_mul` 支持 BSND/TND、half 模式、D<896 且偶数（128 ✓）；但 `RotaryMulGrad` 要求 D 为 64 的倍数、**广播轴 ≤1024**，且 cos/sin 必须与 x 同 dtype——当前 processor 的 cos/sin 是 fp32，直接与 bf16 的 q 相乘会触发 type promotion，把整条 prologue 和 FA 输入抬到 fp32（这本身可能就是 4.1 之外的一个隐藏成本）。做法：cos/sin 以 fp32 构造、在 kernel 边界 cast，28 层共用；`k_norm_und_for_gen` 若是纯 L2 normalize（无 gamma）要单独分支。
3. **norm + residual**：`npu_add_rms_norm(x1, x2, gamma, eps) -> (y, rstd, x)`，A2 支持 bf16；但 op_plugin 里**没有独立的 `npu_add_rms_norm_backward`**，要包一层 `autograd.Function`：`dx = npu_rms_norm_backward(dy, x, gamma, rstd)`，再令 `dx1 = dx2 = dx`。`x` 必须显式物化并真的被反向使用（`save_for_backward`），尤其在 full recompute 的段边界——否则重算段与原算段的 bf16 加法顺序不一致。
4. **不要做 GEMM epilogue 融合**：DAV_2201 没有 L0C→UB，朴素 mix 模式必须经 GM。CANN 官方的 4096³ MatMul+LeakyReLU 实测：朴素融合 1.826 ms，比拆分的 1.524 ms 还慢；只有低阶 API 自管流水才做到 1.38 ms。所以 epilogue 融合不是 drop-in 优化，必须先做单层 microbench、证明 ≥10% 再投入。

**一个不要做的事**：不要用 `npu_linear` 给 bf16 投影做 matmul+bias 融合——它在 A2 上不支持 bf16；`npu_fused_matmul` 支持 bf16 但只在 950PR/950DT 上可用。910B 的 bf16 投影只能走 `aclnnAddmm`/`F.linear`。

**一个值得顺手做的小项**：把每路的 `to_q/to_k/to_v` 三个投影沿输出维拼成一个 `[4096,2048]` 的 GEMM（`add_q/add_k/add_v` 同理）。它本身只省约 2.5 ms/步，但产出的 `[T,4096]` 连续缓冲是 CP all-to-all 合并（4.5）与单核 prologue（下面 P2 的 P05b）的前置条件。注意是**同一路内**拼接；AR 与 DM 两路权重不同、M 不同（284 vs 7104），不能合成 grouped GEMM。

### 4.3 T3：two-way attention——结构已经对了，不要为 1 ms 改内核

当前实现（`colossalai/shardformer/layer/two_way_attn.py:635` 的 `npu_two_way_flat_attention`）是两次调用：

- **AR 因果调用**：Q/K/V 为全局 AR 块（284 token），`AttnMaskType.CAUSAL`、`sparse_mode=3`，因果 mask 由后端缓存，不在这里构建；
- **DM 全注意力调用**：Q 为 DM 块（7104 token），K/V 为逐样本 `[AR_i; DM_i]` 交错（`_interleave_two_way_kv`，`two_way_attn.py:594`），用 `actual_seq_qlen`/`actual_seq_kvlen` 表达块对角，**不构建 [S,S] 稠密 mask**。

这是一份很干净的设计：varlen、无稠密 mask、因果 mask 复用、GQA 原生支持。唯一可议的是每层两次 `torch.cat`（K 与 V 各一次）：

| 量 | 值 | 来源 |
| --- | --- | --- |
| 每层交错流量 | 60.52 MB（K/V 各读 15.13 + 写 15.13） | derived |
| 每前向 | 1.69 GB | derived |
| 完全重算时 | 3.39 GB | derived |
| 折算时间 | 1.06 ms/前向；反向的 split 再读一遍同样的字节 → 约 3.2 ms/步 | derived |

**结论：不做。** 为约 3.2 ms/步写一个同时支持 `sparse_mode=3` 与 `sparse_mode=0` 的自定义内核，成本远高于收益；现有 API 也无法在一次调用里表达两种 mask（`npu_fusion_attention` 的 varlen 路径只接受一组 `actual_seq_*` 与一个 sparse mode）。仓库 docstring 里"若拷贝超过预算就把交错折进投影输出"的建议是对的，但预算应该按实测重算——**1.06 ms 远低于预算**，所以维持现状。

同样不做的还有"把 AR 与 DM 合成一次 launch"：`sparse_mode` 是全局单一模式，mode=3 会把 DM 也变成因果（错），mode=0 需要稠密 mask 而 TND 路径在 wrapper 里直接拒绝稠密 mask。若硬改成一次 q=7388/kv=7388 的调用，AR 行要多算 284×7388 对（当前只有 284×284 因果），每样本 q×k 对从约 26.3M 增到 28.3M，attention FLOPs +8%（约 +0.9 TFLOP/前向），远大于省掉交错的 1.06 ms；还要每层读 13.6 MB/样本的稠密 mask。**两次调用是最小分解。**

**真正值得保留的 two-way 相关工作**是它的**输入布局**：如果 T2 的投影 epilogue 将来要重写，可以让 `to_k`/`to_v` 直接写到交错缓冲区，顺手消掉这 1 ms；但这是"重写投影时的附带收益"，不是一个独立项目。

### 4.4 T4：残差、重计算与 SAC 的边界

**先把实测口径钉死（红队复核）**：本 checkout 的 `train.py` 只有 full checkpointing，没有 SAC 分支；3.413 s 那一步的反向/前向 = 0.475/0.217 = 2.19×，正好是**无重算**的比例，说明实测里没有开重算。所以"SAC 只重算两块 MLP"是配方描述，不是这次测量的配置。融合必须尊重两个边界（它们对 full/no-recompute 两档都成立）：

1. **残差必须在重计算段的入口显式物化**。`npu_add_rms_norm` 的输出 `x`（残差和）正是反向需要的中间量；如果把它融进下一层而不落 GM，重算时就得重算整段。做法：在 SAC 边界写一次残差（每层 30.26 MB，28 层 0.85 GB/前向，约 0.5 ms），其余层用融合版。
2. **`npu_mm_all_reduce_add_rms_norm` 在 HSDP/ZeRO 下不可直接用**。它假定 all-reduce 后拿到完整梯度；我们是 reduce-scatter 到分片。要么用 MindSpeed 的 reduce-scatter 变体，要么把 add+norm 单独融合，通信保持 HCCL 原语。

收益已在 4.2 的 norm+residual 行里（约 2.1 ms/前向）；红队复核后这一整节只值 **2–5 ms/步（约 0.1%）**，它是一条"别把重算段的边界写错"的纪律，不是一个优化项。

### 4.5 T5：通信与重叠（P1，与第五节合并执行）

16 卡实测的固定通信约 0.9 s/optimizer step（单步 4.20 s 的 21%），由 `m + c = 4.2`、`4m + c = 14.0` 解出。可动的三处：

1. **梯度 reduce-scatter 分桶**：桶太小 → 启动次数多；桶太大 → 与反向重叠变差。按每层梯度字节数（约 100–200 MB/层，bf16）取 2–4 层一桶，让最后一个桶在反向结束后立刻完成。
2. **参数 all-gather 与下一步的 VAE 重叠**：ZeRO 在 optimizer 后要 all-gather 2.85 GB/rank 给下一步前向。若 VAE 编码在另一个 stream 上（5.3），这段通信可以与 VAE 的 1.0 s 重叠。量级取决于链路：单机 8 卡按 200 GB/s 约 14 ms，双机 16 卡按 25 GB/s 约 114 ms——所以是 **0.01–0.11 s/步**，不是 0.3 s；单机几乎可以忽略，跨节点才值得做。
3. **CP all-to-all 合并**：CP 启用时，每层前向 7 次 + 2 次回换，反向同样 9 次。把同一路的 q/k/v 沿头轴拼成一次交换后，前向 9→5（AR、DM、`k_ar_for_gen` 各一次 + 2 次回换），反向同理；每层少 8 次集合通信，28 层共 224 次/步。另外 `_all_to_all_single` 在 scatter_dim≥2 时会先 reshape+transpose+contiguous、收到后再来一遍，每层每方向约 121 MB 的额外拷贝——让投影输出直接采用发送所需的头顺序可以去掉。保守 **0.02–0.2 s/步**，必须实测；只在 cp>1 时有效，当前 cp 只在单机 sp=2 跑通，优先级低于前两项。

**注意（红队更正）**：0.9 s 是**每次参数更新**的固定成本，其中包含 optimizer 段，不是纯 HCCL；官方配方的 `grad_accum_iter=1`，所以它就是每步成本。早期草稿把"256 个 rank"误记成了"256 个 micro-step"并据此说可以摊薄 256 倍——**这是错的**，不要用那个折扣。这也意味着 5.3 的多流能填的是这 0.9 s 里的设备 idle（整步约 0.32 s），不是 0.9 s 本身。

### 4.6 T6：反向与自定义算子的纪律

训练路径与 VAE 的根本区别是**必须有反向**。这决定了一条硬规则：

- **优先用 op_plugin 自带 backward 的算子**：`npu_rms_norm_backward`、`npu_swiglu_backward`、`npu_rotary_mul` 的 autograd、`npu_fusion_attention_grad`。
- **只做前向的自定义 kernel 在训练里会把收益还回去**：前向省的 GM 流量，反向要么重新读一遍，要么 autograd 保留更多中间量。若确实要写（例如 QKV+norm+RoPE 的 epilogue），必须同时写第二支反向 kernel——norm/rope 的反向是局部的，成本与前向同量级，这是可以接受的；但收益仍然只有 4.2 表中的那几毫秒。
- **不要在反向里引入原子浮点加**：910B 的确定性依赖固定归约顺序，原子加会破坏对拍。VAE 融合（P1–P4）不受影响，因为编码在 `no_grad` 下。

**本节结论：训练侧的算子工作，先做 4.1（优化器）与 4.5（通信重叠），4.2/4.4 作为"顺手做"的清理项，4.3 明确不动。**

## 五、数据链路、精度与调度

这一节的三项都不改变任何算子数学，但它们决定前面那些算子优化能不能变成端到端的步时。

### 5.1 uint8 H2D 与 worker 侧预处理（P1）

每样本 [1,3,33,544,736]：fp32 158.6 MB，uint8 39.6 MB，正好 4×。按 PCIe Gen4 ×16 约 25 GB/s 的名义带宽，单样本拷贝 6.3 ms → 1.6 ms。注意口径：32 样本（16 卡 × 2 样本）一次**迭代**的主机侧总拷贝省约 150 ms，摊到**每 rank 每步只有 2 × 4.7 ≈ 9.4 ms**——即 **0.005–0.015 s/步**，不是早期草稿的 0.03–0.08 s（红队复核，把 16 rank 的累加当成了单步收益）。H2D 的真实阻塞时间仍未实测（inferred）。真正的好处不只是带宽：

- **主机内存**：DataLoader 的 pinned buffer 从 158.6 MB/样本降到 39.6 MB，`/dev/shm` 32 GB 的天花板被抬高 4×（当前 4 样本/卡就报 `unable to allocate shared memory`，已用 `file_system` 策略绕开）；
- **归一化位置**：`u/127.5 − 1` 从主机搬到设备，与 patchify/conv1 的读取融合（3.1），不新增 kernel；
- **精度链**：uint8 → fp16 的转换在设备上做，避免"主机 fp32 归一化 → 再转 fp16"引入一次额外的舍入。

**验收口径**：同一 batch 用 uint8 路径与 fp32 路径各编码一次，latent 的最大绝对差 ≤ 1 ulp of fp16（约 1e-3 量级），且不得越过 fp16 已测的 0.64× 量化噪声门禁。

### 5.2 tokenize 下放 worker（P1）

当前提示词 tokenize 若在主进程，会与 VAE 编码串行。下放到 DataLoader worker 后，它与上一批的 VAE/前反向重叠，主进程只收 token id。收益取决于 tokenizer 的实测耗时：2 条短 prompt 用 fast tokenizer 应是亚毫秒级，**0.001–0.01 s/步**（inferred）；早期草稿的 0.02–0.05 s 高估 ≥10×，而且它和 `aten::to` 的 2.35 s 没有关系。风险是 worker 里的 tokenizer 必须可 pickle（仓库此前已修过 worker seeding 的 picklability，见提交 `0b0ed04e`），以及词表/配置要随 worker 初始化一次而不是每 batch 读。

### 5.3 多流编排：唯一能回收 idle 的手段（挂起：仅在线 VAE 时执行）

> **状态（2026-11-09）**：VAE 改为离线预提取后，训练步里没有 VAE 可重叠，本节的 0.1–0.3 s/步 不再适用于当前主线；保留设计供回到在线编码时使用（见 6.4）。

设备占用率的分段差异是这条设计的全部依据：

| 段 | 设备占用 | 含义 |
| --- | --- | --- |
| VAE 编码 | 97–104% | 计算密集，几乎没有 idle |
| 前向 + 反向 | 60–77% | 有 23–40% 的 idle |
| optimizer | 28–43% | 发射受限，一多半时间设备在等 host |

（分段 idle 相加会得到 0.33–0.49 s，但整步口径是"真在算 3.09 s / 3.413 s"= 约 0.32 s idle；分段数里有重叠与估算误差，**以整步的 0.32 s 为准**，下面的收益也按它封顶。）

因此重叠的正确搭配不是"VAE 与前向"（两个都吃 Cube，抢资源），而是 **VAE 与参数更新窗口**：optimizer 段一半以上 idle，集合通信用 SDMA 而不占 Cube，VAE 的 Cube 计算可以填进这段 idle；前向/反向段的 23–40% 空泡是次要目标（两者都吃 Cube，会互相拖慢）。注意"参数更新窗口 0.9 s"里包含 optimizer 自己的 0.291 s，不是纯通信——能填的是其中的设备空转，不是 0.9 s 全部。设计：

```
stream A（主）: ... fwd(N) → bwd(N) → [event: grads ready] → optimizer(N) → [event: params ready] → fwd(N+1)
stream B（VAE）:      data(N+1) → VAE encode(N+1) ────────────┘（等 grads ready 之前就可以开始）
```

- VAE 只读冻结权重、写 latent，与 optimizer 的参数更新没有数据依赖，唯一约束是**显存**：VAE fp16 峰值 3.11 GB + 主干 12.30 GB，64 GB 卡上安全。
- 同步点只有两个：VAE 必须在 `fwd(N+1)` 之前完成（event B→A）；optimizer 不得在 VAE 读 latent 缓冲之前覆盖该缓冲（双缓冲即可解开）。
- 若 `aten::to` 的 host 派发真的是瓶颈（5.6），多流不会帮忙——host 仍然只有一个。**先修 host，再开多流。**

**收益（红队修正）**：**0.1–0.3 s/步（3–9%）**，而不是 0.3–0.5。上限不是 0.9 s 的通信窗口，而是**整步实测的设备 idle 约 0.32 s**（§1.2：每步真在算 3.09 s，约 91% 占用）——多流只能把 VAE 的工作填进这段 idle，不能凭空造出并行度。0.9 s 里的大部分是 HCCL 在等 host/等对端，设备并不是空的。另外：与 4.5 的参数 all-gather 重叠是**同一笔账**，不能与 0.01–0.11 s 相加。同卡并发还会让 VAE 与模型各慢 5–10%（HBM/L2 争用），净收益可能再被吃掉两成。**验收必须看 profiler 时间线上的真实重叠区间与 step 墙钟，不能只看占用率。**

### 5.4 显存与共享内存

- 每卡显存闭式：峰值 ≈ 27.5 GiB（静态）+ 3.45 GiB × 样本数（开重算）；不开重算 11.33 GiB/样本。**这个闭式偏乐观**：6 样本配置实测 reserved 57–59.5 GiB，比 27.5 + 6×3.45 = 48.2 GiB 高 9–11 GiB（碎片与 allocator 缓存）。所以 64 GB 卡的实际上限是 6 样本（开重算）/ 2 样本（不开），且 6 样本时没有余量再放大块常驻缓冲。
- 3.1 的 padded buffer 本身很小：544×736、12 帧、16 lane、fp16 ≈ `14 × 274 × 370 × 16 × 2 B ≈ 45 MB/样本`，不改变上面的显存闭式。只有把 padded 布局推广到**深层激活**才会带来 `(T+2)/T` 的膨胀（12 帧约 1.17×），而 P1 不这么做。**但 6 样本配置下已经实测 reserved 57–59.5 GiB**，任何新增的常驻缓冲都要按"能否复用"审一遍，不要按 45 MB 就放松警惕。
- `--latents` 旁路（预计算 latent）能把一步 3.41 s 降到 1.05 s（3.2×）。**2026-11-09 起它是候选训练配方**（见 0.4），代价是增强必须在编码前定型（失去在线增强的多样性）；latent 存储约 3.4 MB/样本，17.9k 样本约 61 GB。

### 5.5 精度策略与数值规范

| 组件 | 精度 | 依据 |
| --- | --- | --- |
| VAE 编码器 | **fp16** | bf16 误差是配方量化噪声的 4.67–5.01×，fp16 是 0.58–0.64×；两者同速（2.45× vs 2.46×） |
| Transformer 主干 | bf16 | 官方配方；910B 的 Cube 对 bf16 满速 |
| 优化器状态 | fp32 master + fp32 m/v | 与现有 ZeRO 一致；fused AdamW 内部也必须 fp32 累加 |
| 融合算子 | 不引入新的 fp32 升降 | `npu_rms_norm` 的 rstd 输出本来就是 fp32；`npu_swiglu`/`npu_rotary_mul` 在 bf16/fp16 上原地算 |

两条纪律：

1. **不要为了融合把 bf16 临时升到 fp32 再降回**。VAE 在 bf16 下多出 441 次 Cast、norm 占比从 8% 升到 24%，就是这类"精度升降税"的实证。
2. **确定性**：所有融合都保持固定归约顺序、不用原子浮点加。不同并行模式之间已有约 0.3% 的 loss 漂移（来自 dp 组变化与归约顺序），融合不应放大它；对拍时用"同模式、同种子、融合开/关"两组，而不是跨模式比较。

### 5.6 host 侧 `aten::to`：先定位，再谈优化（P0 调查项）

现象：host 侧 `aten::to` 累计 2.35 s/步，对应设备时间只有 35 ms/步；profiler 没有录调用栈，所以调用点未知。

**必须说清楚的一点**：host 累计时间不等于关键路径时间。异步派发下，2.35 s 的 `aten::to` 可能大部分与设备工作重叠；真正伤步时的是"host 排队导致设备断流"的那部分。所以这一步的目标不是"消灭 2.35 s"，而是回答两个问题：

1. 这 2.35 s 里有多少落在关键路径上（设备在等 host）？
2. 调用点在哪——是 packing 的逐 chunk `.to(device)`/`.float()`，是 `torch.cat` 的 dtype 提升，还是 VAE 编码输入的 cast？

**方法**：`torch.profiler` 开 `with_stack=True` 跑 1 步，按 `aten::to` 聚合调用栈；或用 `msprof` 的 host 调用栈视图。找到后按"uint8 到设备再转、批量 cast、`non_blocking=True`、去掉逐帧循环"四类手法修。**在这项完成之前，不要把 2.35 s 写进任何收益承诺。**

**顺带一个同类的 host 同步点**：`examples/worldmodel/cosmos3_harness/loss.py:39-42` 的 `velocity_mse` 先用 `masked_select` 把监督元素压紧再求和。`masked_select` 的输出形状依赖 mask 内容，框架必须把元素个数读回 host 才能分配输出——这是一次 data-dependent 的 device→host 同步，也是图捕获的常见断点（VAE 的 `npugraph_ex` 慢 18× 就是 397 次 `aten::item` 切图）。改成 `((pred−target)² × mask).sum()` 与 `mask.sum()` 两路标量、不做压紧，语义不变（分子/分母的 token 归一化仍然显式），但去掉了这次同步。量级很小（单样本约 0.6 M 元素），属于顺手清理，不单列优先级。

### 5.7 数据带宽账

DROID success 集 707 GB 在本地 NVMe。uint8 路径下每样本 39.6 MB；每卡每步 2 样本、按 0.3 步/s 计，单卡读约 24 MB/s，8 卡约 190 MB/s——远低于 NVMe 的顺序读带宽。真正的风险是**随机读 + 解码 CPU**，不是盘。把解码放在 worker、并把增强保持在编码之前（官方 lockstep 语义），是这条链路唯一不能妥协的顺序约束。

---

## 六、路线图（按离线 latent 口径重排）

排序依据是 0.4 的新相位账：训练步约 1.05 s，优化器占 27.7%，forward 20.7%、backward 45.2%。VAE 离线后，VAE 侧算子项不再进训练步，单独列为"离线工具链"。

### 6.1 P0（先做，1–2 周；直接进训练步）

| # | 动作 | 改哪里 | 预期 | 验收 |
| --- | --- | --- | --- | --- |
| P0-1 | 优化器去逐 tensor：复用 cast buffer、unscale/clip 走 `_foreach_mul_`、grad norm 走一次 `_foreach_norm`、AdamW 换 fused（A/B 臂） | `colossalai/zero/low_level/low_level_optim.py`、`examples/worldmodel/cosmos3_harness/train.py` 的 `--adamw` 与 `_build_optimizer` | **实测 8 卡 0.332 → 0.104–0.109 s（两轮 −67%）、1 卡 264 → 232 ms（换类）**；0.09–0.11 s 的 derived 目标第二轮达到 | 段时长 ≤ 90–110 ms（第二轮 104–109，达到）；grad norm 相对 CPU fp64 2.6e-08（≤ 1e-5）；fused 臂梯度对象缺陷已修（6.1.3） |
| P0-2 | host 侧审计：`aten::to` 调用栈 + loss/tokenize | `train.py` 的 batch 组装、`loss.py::velocity_mse`、主进程 tokenize | 先回答"是否在关键路径"，再定收益；若命中，上限 0.1–0.2 s/步 | 带调用栈的 profiler 报告；关键路径占比 |
| P0-3 | 统一 profiler 复测（离线口径） | `bench_ops.py` + `train.py --bench-json`（见 6.6） | 确认各项没有互相抵消 | 同口径前后对比表；每次运行的 JSON 记录 |

P0-1 与 P0-2 可以并行：P0-1 是确定的发射开销，P0-2 是未知量。若 P0-2 证明 `aten::to` 在关键路径，它优先于 P0-1 的调优落地（先修 host，再调 kernel）。

#### 6.1.1 P0-1 实施记录（2026-11-09）

已落地，CPU 侧契约测试通过；真机数字见 6.1.2：

| 改动 | 位置 | 状态与说明 |
| --- | --- | --- |
| 梯度回 fp32：两次 `.to()` 合成一次 | `low_level_optim.py:575` | 已落地。旧写法先就地 cast 再搬运，每分片 2 次 launch；新写法 1 次，数值逐位不变 |
| master→bf16 回写：同样合成一次 | `low_level_optim.py:610` | 已落地。这是每参数每步的路径，ZeRO-2 下同样省一半 launch |
| unscale/clip：逐 tensor `mul_` → `_foreach_mul_` | `low_level_optim.py:687` | 已落地。同 dtype/同设备守卫；后端无 foreach 核时回退逐 tensor（dispatch 失败发生在任何 kernel 之前，不会半途缩放） |
| `--adamw {auto,plain,foreach,fused}` | `train.py` 的 `--adamw` / `_build_optimizer` | 已落地。`auto` 在 NPU 上走 `foreach=True`——上游的 foreach 设备表只有 CUDA/XPU，除非 torch_npu 自己 patch，NPU 会退回逐 tensor |
| `--bench-json` / `--bench-warmup` | `train.py::_write_bench_report` | 已落地。复用真实训练循环计时，环境块与 `bench_ops.py` 共用 `collect_env()` |
| 算子基准 `bench_ops.py` | `examples/worldmodel/cosmos3_harness/bench_ops.py` | 已落地。`--op unscale-clip|adamw|all`，各 arm 同进程、同几何、同 warmup/iterations/repeats |
| 契约测试 | `tests/test_zero/test_low_level/test_unscale_and_clip_grads.py`、`tests/test_shardformer/test_model/test_cosmos3_example_helpers.py` | 8 + 11 个用例通过（新增 8 + 4） |

本地冒烟（8 张量 × 4096 元素，CPU，**只证明脚本可用，不作性能结论**）：unscale/clip 逐 tensor 0.0082 ms vs foreach 0.0090 ms——核太小，foreach 的分组开销反超；真机要看 512 个分片、每片 ~1e5 元素的账。AdamW 同规模 plain 0.181 ms vs foreach 0.173 ms。

`NpuFusedAdamW` 暂不作为默认：它在**第一次** `step()` 时把每个 group 的参数、梯度、动量一次性合并成连续 buffer（`npu_combine_tensors` 用 `npu_change_data_ptr` 把原张量重指向合并 buffer），并把 `is_params_grads_combined` 置为 True，此后不再重建。因此**任何在此之后被替换掉的张量对象都对它不可见**。原 ZeRO-2 步每步执行 `master_param.grad = grad.float()`——每个梯度都是新对象——所以 fused 臂从第 2 步起一直在用第 1 步的梯度（实测：每步换新梯度张量，10 步后与 plain AdamW 差 **3.34e-03**；改成复用同一 buffer 后差 **2.38e-07**，见 6.1.2）。原先 10 步对拍用的是同一份梯度值的 `.clone()`，检测不到这一点。本次改动让每个 master 分片的 fp32 梯度 buffer 跨步复用，fused 臂因此**既变快也变对**；但 `param_groups[...]["params"]` 每步换成"本步有梯度的 master 子集"这一路径仍要求在真实 pack 上补一次 10 步对拍后再切默认。

真机待办（按顺序；1–3 已于 2026-11-09 完成，数字见 6.1.2）：

1. ~~`bench_ops.py --op all --device npu --num-tensors <分片数> --numel <每片元素数>`：拿到 unscale/clip 与 AdamW 各 arm 的 median/min。分片几何按 checkpoint 的 `numel / 世界大小` 推，并把 `kernel_details.csv` 一并留档。~~ 已完成，三种几何。
2. ~~`train.py ... --bench-json report.json --adamw plain` 与 `--adamw foreach` 各跑一次：确认整步与优化器段真的下降，且落在"一步参数 `max|Δ| ≤ 1e-6`"的门禁内。~~ 用 `bench_opt_ab.py` 在 1 卡与 8 卡上完成 hybrid/fused 的对拍；foreach 的整步收益留待默认臂切换时补。
3. ~~通过后再测 `--adamw fused`；若对拍不一致，把该臂从脚本里删掉，不留"可能错"的默认。~~ `fused` 数值门通过（10 步 `max|Δ| = 4.47e-08`），仍只作 A/B 臂。
4. ~~grad norm 的 `.double()`（`_compute_grad_norm`）是下一步的首要目标：fused 臂的 optimizer 段仍有 203 ms，而 AdamW 本体只占约 13 ms。~~ 已完成（见 6.1.3）：grad→fp32 与 master→bf16 改为复用 buffer + 一次 `_foreach_copy_`，grad norm 改为一次 `torch._foreach_norm`，段 203 → 104–109 ms，且顺带修掉了 fused 优化器的梯度对象缺陷。

#### 6.1.2 P0-1 真机结果（node1，910B2，1 卡与 8 卡；torch 2.13.0+cpu / torch_npu 2.13.0.rc1）

算子基准 `bench_ops.py --op all --device npu`，fp32、总量 172.8M 元素（≈ 8 卡下 1.423B 可训练参数的分片），每组 30 次 × 3 轮取 median；`--betas 0.9 0.99 --weight-decay 0.05 --learning-rate 2e-4`，与 SFT 配方一致。单位 ms：

| 几何（张量数 × 每张量元素） | unscale/clip 逐 tensor | unscale/clip foreach | AdamW plain | AdamW foreach | NpuFusedAdamW | HybridAdam（现配方） |
| --- | --- | --- | --- | --- | --- | --- |
| 64 × 2.7M | 1.47 | 1.14 | 12.4 | 12.4 | 13.2 | 20.2 |
| 256 × 675k | 6.51 | 1.17 | 14.7 | 14.0 | 13.2 | 84.7 |
| 432 × 400k | 10.65 | 1.43 | 21.6 | 21.4 | 13.2 | 138.4 |

三条结论：

1. **unscale/clip 的 foreach 收益随张量数线性增长**：432 张量时 10.65 → 1.43 ms（7.5×），256 张量时 5.6×，64 张量时 1.3×。foreach 的耗时基本落在带宽下限上（691 MB 读 + 691 MB 写 ≈ 1.38 GB，1.5 TB/s → 0.9 ms），而逐 tensor 版本是纯发射开销。
2. **AdamW 的 `foreach=True` 在 NPU 上几乎无效**（21.6 vs 21.4 ms）：torch_npu 没有把 AdamW 的七个逐元素算子真正合成多张量核，省下的只是 Python 侧调度。**唯一有效的是 `NpuFusedAdamW`**（13.2 ms，1.6×），而且它不随张量数变慢（64 张量时也是 13.2 ms，说明它是按合并 buffer 算的）。
3. **现配方的 `HybridAdam` 是最慢的一臂**：NPU 上它走 `torch_adam_update` 的逐参数 Python 循环（`cpu_adam.py:92`，每参数 7–9 次 launch），432 × 400k 时 138 ms，是 fused 的 10.5×、plain AdamW 的 6.4×，且随张量数线性恶化（64 × 2.7M 时只有 20 ms）。按 64 × 2.7M 那一行线性放大到 1.423B，1 卡下 HybridAdam 本体约 166 ms——这就是 4.1 里"291 ms 优化器段"的主要构成，其余是 grad norm、clip、逐张量 cast 与参数回写。

真实训练步的 1 卡 A/B（`bench_opt_ab.py`，322 个张量 = 317 base + 5 boosted，1.423B 可训练元素，1 样本/步，warmup 1 + 4 步）：

| 内层优化器 | 整步 | pack | forward | backward | optimizer 段 | 峰值 HBM |
| --- | --- | --- | --- | --- | --- | --- |
| HybridAdam（现配方） | 2.139 ± 0.063 s | 1.208 | 0.202 | 0.399 | 0.264 | 38.9 GiB |
| NpuFusedAdamW | 2.159 ± 0.042 s | 1.199 | 0.220 | 0.462 | 0.232 | 53.9 GiB |

两处必须写清楚的事实：

- **1 卡时 fused 只省 32 ms（12%）**，因为 1 卡下 322 个张量平均 4.4M 元素，HybridAdam 的逐参数循环是带宽受限而不是发射受限。分片越小、张量数不变，逐张量开销占比越高——8 卡下每张量降到 ~553k 元素，才是算子表第三行对应的场景。
- **fused 的合并 buffer 让 1 卡峰值 HBM 多 15 GiB**（38.9 → 53.9 GiB）：`NpuFusedAdamW` 把参数、梯度、两个动量都复制进连续 buffer，约 1.423B × 16 B ≈ 22.8 GB，扣掉原先分散的 state 后净增约 15 GiB。8 卡分片后净增约 1/8（~1.9 GiB/卡），单卡 64 GB 下可接受，但它是"换速度的代价"，不是免费的。

8 卡真实 ZeRO-2 分片（`bench_opt_ab.py`，8 rank，1 样本/步，warmup 1 + 4 步，每 rank 分到 1.423B/8 ≈ 178M 元素，即算子表 432 × 400k 那一行）：

| 内层优化器 | optimizer 段（8 rank） | 整步 median（各 rank） | pack | forward | backward | 峰值 HBM |
| --- | --- | --- | --- | --- | --- | --- |
| HybridAdam（现配方） | 0.332 s（0.331–0.336） | 2.431–2.438 s | 1.20–1.57 s | 0.19–0.23 s | 0.42–0.77 s | 22.5 GiB |
| NpuFusedAdamW | 0.203 s（0.203–0.208） | 2.096–2.310 s | 1.18–1.20 s | 0.20–0.24 s | 0.45–0.77 s | 23.0 GiB |

- **optimizer 段 0.332 → 0.203 s（−0.129 s，−39%）**，与算子表预测的 138.4 − 13.2 = 125 ms 只差 4 ms（3%）——按算子级账推整步是成立的，这是本次真机测试最强的结论。
- **段的绝对值不等于 AdamW 本体**：fused 臂剩下的 203 ms 里 AdamW 约 13 ms，其余是 `_compute_grad_norm`（`GRAD_CLIP=1.0`，逐分片 `.double()`）、逐张量 grad cast 与 master→bf16 回写、`zero_grad`，以及 step 末尾把更新后的 bf16 参数 all-gather 回全量（2.85 GB，每 rank 收 ~2.5 GB）的通信。**4.1 里"只换类到 90–110 ms"的 derived 目标没有达到**；要继续压必须先做 msprof 归因（并入 P0-2），而不是继续换优化器类。
- 峰值 HBM 22.5 → 23.0 GiB（+0.5 GiB/卡）：合并 buffer 的代价被 ZeRO 切到 8 卡，远小于 1 卡的 +15 GiB，实测比预测的 ~1.9 GiB 还小。
- 两臂的整步 total **不是同口径**：hybrid 臂用 4 个 dataloader worker，fused 臂因下述卡死改用 `--num-workers 0`；只有 optimizer 段和峰值 HBM 可以横比（它们不经过数据路径）。fused 臂各 rank 的 median 散布（2.096–2.310 s）也比 hybrid（2.431–2.438 s）大，属于 0-worker 下各 rank 同步取数的偏斜。

数值门（`p0_opt_match.py`，node1，32 个参数 × 4096 元素、10 步，对照 plain `torch.optim.AdamW`）：foreach `max|Δ| = 0`，fused `4.47e-08`，hybrid `4.47e-08`，全部 ≤ 1e-6 门限。**这次对拍每步用的是同一份梯度值的 `.clone()`，对 fused 的"梯度张量每步更换即失效"缺陷不敏感**——换成每步新随机梯度后 fused 差 3.34e-03，见 6.1.3；修正后（复用梯度 buffer）为 2.38e-07。这是在真机上用合成几何对拍，未覆盖真实 checkpoint 的分片边界；切默认臂前应在真实 pack 上补一次 10 步对拍。

运维记录：8 卡 fused 首次运行带 4 个 dataloader worker 时卡死约 25 分钟——32 个 spawn worker 停在 `Dl` 状态、CPU 时间只有 ~14 s，宿主 load ~44；`pkill` 后以 `--num-workers 0` 重跑一次完成。这是 NFS home 上的 data path 问题，与优化器无关，但复现时直接用 0 worker。

#### 6.1.3 P0-1 第二轮：残余归因与三项修复（2026-11-09 晚）

第一轮后 8 卡 optimizer 段 203 ms，而 AdamW 本体只占 13 ms，剩余约 190 ms 的构成当时没有分解。这一轮先归因，再按归因改代码，最后同口径复测。

**归因方法与它的坑。** 先用同步插桩（`p0_seg_attrib.py`：在 `_compute_grad_norm`、`_unscale_and_clip_grads`、`TensorBucket.all_gather`、`zero_grad`、`release_param_grad`、内层 `step` 前后各一次 `torch.npu.synchronize()`）跑 8 卡，4 步的结果是 `inner_step` 1122.9 ms、`bucket_all_gather` 643.1 ms / 64 次、`grad_norm` 414.1 ms / 8 次。**这些绝对数不能用**：每次包裹的前置同步会把之前排队的异步工作算进后一次调用，四项相加约 550 ms/步，已经超过整段的 233 ms。能用的只有两条相对信息：(a) `grad_norm` 每次约 52 ms，与单独测的 63.7 ms 同量级，说明第一轮微基准里 `foreach_norm` 2.5 ms、逐分片 63.8 ms 的对比是对的；(b) 内层 step 的首步包含 fused 优化器的一次性合并 buffer 初始化（对 322 个参数与 322 个梯度逐个 `npu_change_data_ptr`），按 4 步均摊会把它摊成每步几百 ms。第二遍改用 NPU event（记录不触发 host 同步）复测，数字见本节末。

**grad-norm 的真机账（1 卡，fp32，真数据；对照是 CPU 上的 float64 范数——NPU 无 float64）：**

| 几何 | 逐分片 `.double().norm()`（原路径） | foreach + 64k 分块 | foreach，整分片 |
| --- | --- | --- | --- |
| 322 × 553k | 63.7–77.0 ms（rel 7.6e-09） | 45.6–51.7 ms | **4.14 ms（rel 2.58e-08）** |
| 64 × 4.42M | 13.0–17.6 ms（rel 8.8e-09） | 64.1–66.6 ms | **1.62–1.72 ms（rel 1.59e-08）** |

两条结论直接写进了代码注释：(a) 分块必须远大于分片，因为每块一次 launch；64k 分块把 322 个分片拆成约 2900 块，在 64×4.42M 几何上反而比原路径慢 4 倍。(b) 分片不超过一块时不要构造 view/split——322 个 view 对象的 Python 开销约 6 ms，比 norm 本身还贵。因此 `_GRAD_NORM_CHUNK = 1 << 24`，只用于保护病态大分片；整分片 fp32 归约相对 CPU fp64 只有 2.6e-08，远在 1e-5 门内。

**三项修复（`colossalai/zero/low_level/low_level_optim.py`）：**

| 改动 | 机制 | 每步收益（8 卡算子账） |
| --- | --- | --- |
| grad→fp32 改用复用 buffer + 一次 `torch._foreach_copy_` | 每个 master 分片一个常驻 fp32 buffer，取代每步每参数一次 `.to()` 新分配 | 12.18 → 约 1.0 ms |
| master→bf16 回写同样复用 buffer + 一次拷贝 | 同上；写回 buffer 同时是参数 all-gather 的输入 | 12.33 → 约 1.3 ms |
| grad norm 改为一次 `torch._foreach_norm` | 322 个分片一次多张量归约，块结果在 float64 里求和 | 63.8 → 4.1 ms |

**fused 的梯度对象缺陷（这一轮最重要的发现）。** `NpuFusedAdamW` 只在第一次 `step()` 时用 `npu_combine_tensors` 把参数、梯度、动量合并成连续 buffer（`npu_change_data_ptr` 把原张量重指向合并 buffer），之后 `is_params_grads_combined` 一直为 True、**不再重建**。原 ZeRO-2 步每步执行 `master_param.grad = grad.float()`，每步都是新张量，于是 fused 臂从第 2 步起一直在用第 1 步的梯度。真机对照（`p0_fused_grad_identity.py`，8 参数 × 4096 元素、10 步、每步换新梯度值，对照 plain AdamW）：**每步换新梯度张量 → `max|Δ| = 3.34e-03`；复用同一 buffer → `2.38e-07`**。第一轮的 10 步数值门（4.47e-08）用的是同一份梯度值的 `.clone()`，所以检测不到。结论有两条：第一轮 fused 臂的 203 ms 在"耗时"上仍可比（kernel 与流量相同），但它的**更新从第 2 步起是错的**；本次的 buffer 复用让 fused 臂既快又对。

**第二轮 A/B（与第一轮同一条命令、同一口径：8 rank、1 样本/步、warmup 1 + 4 步、`--num-workers 0`）：**

| 臂 | optimizer 段 | 峰值 HBM |
| --- | --- | --- |
| HybridAdam（第一轮） | 0.332 s | 22.5 GiB |
| NpuFusedAdamW（第一轮；梯度张量每步更新，数值有缺陷） | 0.203 s | 23.0 GiB |
| NpuFusedAdamW + 本轮三项修复 | **0.104–0.109 s** | 23.4 GiB |

**相对第一轮 −0.095 s（−47%），相对最初 hybrid 臂 −0.223 s（−67%）；4.1 的"段 ≤ 90–110 ms"验收在这一轮达到。** 峰值 HBM 只多 0.4 GiB/卡：复用 buffer 与缓存分配器原本就会复用的块同量级。

**残余的 event 归因（第二轮代码，每步，rank 0–4 一致）：**

| 项 | 每步耗时 | 说明 |
| --- | --- | --- |
| `bucket_all_gather` | **66–76 ms**（16 次/步） | step 末尾把更新后的 bf16 分片 gather 回全量参数（每 rank 收 2.85 GB） |
| `inner_step`（fused AdamW） | 14.5–14.7 ms | 与算子表 13.2 ms 一致 |
| `grad_norm` | 6.9–8.6 ms（2 次/步） | 两个 param group 各一次 |
| `unscale_clip` | 2.6–2.7 ms | `_foreach_mul_` |
| `zero_grad` / `release_grad` / `global_norm_reduce` | 均 < 1 ms | |
| 合计 | 90–103 ms | 与段 104–109 ms 自洽（event 不触发 host 同步，第一遍同步插桩的 550 ms/步不再出现） |

**参数 all-gather 一项就占残余的约 70%**，下一刀在这里（P1-2）。**桶大小这一维已排除**：把 `zero_bucket_size_in_m` 从 12 提到 48（桶数 16 → 4，其余不变、同一条 8 卡命令），optimizer 段是 **0.110 s（7 个 rank 里 6 个）/ 0.126 s（1 个）**，与 12M 桶的 0.104–0.109 s 相比没有收益、反而略高；backward 也没有变快（0.39–0.88 s，12M 桶时为 0.41–0.65 s，含 rank 偏斜）。**结论：残余不是每桶的启动/尾部开销，加大桶不是解法**；P1-2 应直接做 async 重叠，或先查 HCCL all-gather 的有效带宽——2.85 GB / 70 ms ≈ 40 GB/s，而同一体积的独立微基准（50 MB 桶、单次调用）能到 120 GB/s，差 3 倍且与桶数无关，指向 step 内的排队/同步方式。**P1-2 随后已实测：框架现成的 `overlap_allgather` 是净亏，见 6.1.4。**

#### 6.1.4 P1-2：async 参数 all-gather 的实测——钩子开销大于可隐藏的通信（2026-11-09 深夜）

P1-2 的目标是把 step 末尾的参数 all-gather（event 归因 66–76 ms/步，约占第二轮残差的 70%）藏到下一步的 pack/前向后面。框架本身已有这条路径：`HybridParallelPlugin(overlap_allgather=True)` 会把参数转成 `ColoParameter`、装上 `ZeroOpHook`，step 末尾改为逐参数 `all_gather_into_flat_tensor_nd(..., async_op=True)`，handle 挂在参数上，下一步 `pre_forward` 第一次用到该参数时才 wait。不需要改一行代码，因此先用它做 A/B。

**计时口径。** 逐步 `synchronize()` 会把 step 末尾排队的异步工作算进本步，正好掩盖要测的重叠收益，所以改用窗口计时：warmup 3 步之后连续跑 10 步、期间不做任何设备同步，末尾同步一次再除以 10；两臂各跑两轮，按 ABBA 顺序交错抵消机器漂移。

**结果（8 卡，1 样本/步，同一条命令只切 `overlap_allgather`）：**

| 场景 | sync | async | 差 |
| --- | --- | --- | --- |
| 4 个 dataloader worker | **2.206 / 2.262 s** | **2.543 / 2.527 s** | +0.27～0.34 s（+12～15%） |
| 0 worker | 4.344 / 4.331 s | 4.773 / 4.882 s | +0.43～0.55 s（+10～13%） |

方向和量级在两套设置、四轮里都一致：**打开重叠反而更慢**，不是噪声。

**为什么慢。** 相位插桩（0 worker，每段前后同步）把差异定位得很干净：

| 段 | sync | async |
| --- | --- | --- |
| pack | 1.18 s | 1.18 s |
| forward | **0.19–0.22 s** | **0.37–0.41 s**（8/8 rank） |
| backward | 0.40–1.00 s（含 rank 偏斜） | 0.55–0.72 s |
| optimizer | 0.203 s | 0.200 s |

两个结论：(a) **optimizer 段一分钱都没省**——async 只是把 host 的 wait 推迟，段时长 0.203 → 0.200 s，说明通信并没有被移出关键路径（HCCL 在当前 stream 上顺序排队）；(b) 多出来的时间是 `ColoParameter` + `ZeroOpHook` 在每个 aten 算子上的 Python 钩子，forward 一项就 +0.19 s/步，与窗口里的 +0.27～0.55 s 同量级（反向还有同类钩子，未单独插桩）。

**不是通信粒度的问题。** 纯通信微基准（8 卡，322×553k 的真实几何）：逐参数 73.6 ms、分桶含回写 60.6 ms、分桶纯通信 29.4 ms。322 次小集合通信只比 16 个大桶贵约 13 ms，离 0.5 s 差一个数量级。

**旁路 stream 能不能救。** 可以，但收益有上限。8 卡微基准：gather 24.5 ms、compute 85.0 ms、同流串行 104.0 ms、**旁路流重叠 89.4 ms**——约 60% 的 gather 被计算盖住（只暴露 4.4 ms）。所以若保留分桶、把整段 all-gather 连同回写发到独立 stream、只在下一步 forward 前等一次事件（不装逐算子钩子、不转 `ColoParameter`），理论上能拿回 60–74 ms/步；实现上要注意 `TensorBucket` 的临时 buffer 要活到事件完成（`record_stream` 或持有引用）。

**结论与处置。** `overlap_allgather` 在当前 Ascend 栈上是净亏（+0.27～0.55 s/步），默认关闭；P1-2 若继续，唯一有意义的是上述粗粒度版本，收益上限 0.060～0.074 s/步（整步的 2.5～3%），列为 P2 观察项。

**测量环境注记。** 以上数字在 node1 的 `nzf-cai-dev` 容器（8×910B2）上取得。先前几轮把 venv 放在 NFS home（`/home/niuzifan/envs/cai-npu`），单进程 import 整套依赖要 165.8 s，`spawn` 的 dataloader worker（8 rank × 4）会成片卡在 NFS 上，4 worker 对照因此不可用（曾出现 17 s/步和整轮无进展）；把 venv 与仓库复制到本地盘 `/mnt/disk0/jiangzs/niuzifan-local/` 后 import 降到 10.5 s，本节 4 worker 的数据是在本地副本上重测的。

### 6.2 P1（训练步的次要项，1 个月）

| # | 动作 | 预期 | 风险 |
| --- | --- | --- | --- |
| P1-1 | T2 逐元素：SwiGLU 单 GEMM、`npu_rotary_mul`、`npu_add_rms_norm`；QKV 三投影合并 | 0.03–0.05 s/步（整步 3–5%） | 反向可表达性；先单头对拍；`npu_add_rms_norm` 无独立 backward |
| P1-2 | D04 async 参数 all-gather（与下一步前向/数据预取重叠） | **实测为负**：框架 `overlap_allgather`（逐参数 async + `ZeroOpHook`）在 8 卡 4 worker 上把窗口步时从 2.206/2.262 s 推到 2.543/2.527 s（+0.27–0.34 s），0 worker 时 4.344/4.331 → 4.773/4.882 s；钩子开销 ≥0.19 s/步，远大于可隐藏的 0.060–0.074 s（见 6.1.4）。只剩"分桶 + 旁路 stream + 单次 wait"的粗粒度版本，上限即 0.060–0.074 s/步 | 见 6.1.4；粗粒度版本需处理 bucket 临时 buffer 的生命周期 |
| P1-3 | loss 去 `masked_select` 同步 + tokenize 下放 worker | 0.001–0.01 s/步 | data-dependent 同步是图捕获断点；语义不变 |
| P1-4 | D09 CP all-to-all 9→5 合并（含去两次 contiguous） | 0.02–0.2 s/步（随 pack 长度增长） | 仅 cp>1；当前只在单机 sp=2 跑通 |

### 6.3 离线 VAE 工具链（不占训练步，按数据集周转需求排期）

这些项只缩短离线编码的时间（0.5 s/样本 → 目标 0.45），价值是数据集生产与消融的周转，不是每步成本。

| # | 动作 | 预期（每样本） | 风险 |
| --- | --- | --- | --- |
| V-1 | P4 残余逐元素：norm+SiLU 单 kernel、AvgDown strided-mean、去 clone | 0.008–0.016 s | 归约留 fp32；残差 epilogue 在 910B 不成立 |
| V-2 | P1 数据入口一体化（uint8 + 归一化 + patchify + 填充） | 0.005–0.010 s；另省 H2D 4× | 因果语义逐位对齐；补零只能在 fp16 域 |
| V-3 | P2 TransData 归因 + 冻结权重预转 FRACTAL_Z_3D | 0.002–0.005 s | 权重格式可能不被 aclnn 接受；先归因再决定 |
| V-4 | P6 NDHWC 布局专项 | 0.002–0.008 s | 与 P4(a) 重叠；依赖全链路重排 |
| V-5 | P3 手写 Ascend C im2col Conv3D（只做 down0/down1） | 0.02–0.05 s；4–8 周 | 只在 V-1/V-2 做完且 Conv3D 仍 ≥45% 时启动 |

### 6.4 仅在回到在线 VAE 时执行

- **P5 多流（5.3）**：VAE 与 optimizer/HCCL 重叠，0.1–0.3 s/步，上限是整步 idle 0.32 s。离线 latent 后训练步里没有 VAE，本项挂起；若将来回到在线编码（例如需要在线增强），再按 5.3 的设计执行。
- 在线模式下 V-1～V-5 的秒数才会直接进训练步，届时按 0.2 的在线杠杆表重新排序。

### 6.5 明确不做

- patchify + conv_in 代数融合作为卷积替换（3.3，C0=16 导致 4× issued MAC）；
- mid-block attention 换 `npu_fusion_attention`（3.7，上限约 2 ms）；
- VAE 内部再开多流、VAE 专用卡（3.8，零和）；
- two-way K/V 交错的自定义单核（4.3，约 3.2 ms/步，成本过高）；
- FP8 / bf16 VAE / 直接 `channels_last_3d`（3.8）；torch.compile 在当前容器缺 triton-ascend，装上后需重测，暂不列为方向；
- 把 24 帧分块当默认（实测只有 1.04–1.06×，显存翻倍）；
- 用 `npu_linear` 做 bf16 投影融合（A2 不支持 bf16）。

### 6.6 统一测试口径（所有优化项共用）

从本节开始，每个优化项都必须用同一套口径报告，禁止拿不同基线互相比。基准脚本为 `examples/worldmodel/cosmos3_harness/bench_ops.py`，输出 JSON（含 git commit、torch/torch_npu/CANN 版本、卡号、dtype、shape）。

**共同条件**

| 项 | 固定值 |
| --- | --- |
| 硬件 | 910B2 单卡（多卡项单独标注 rank 数与链路） |
| 软件 | CANN 9.x + torch_npu 2.13；记录 `torch.__version__`、`torch_npu.__version__` |
| 预热/测量 | 算子基准：`bench_ops.py` 默认 10 次 warmup + 3 组 × 30 次，组内取"每调用毫秒"，报 median 与 min；训练步：前 `--bench-warmup`（默认 5）步不计。每段前后 `synchronize()` |
| 随机性 | 固定 `--seed`；`bench_ops.py` 在同一进程、同一设备上顺序跑各 arm；训练步用同一 replay pack 与种子 |
| 记录 | 墙钟、kernel 数、设备占用、峰值 HBM；同时存 `kernel_details.csv`。JSON 统一为 `env` / `conditions` / `results` 三段，两个工具共用 `collect_env()`（git commit、torch、torch_npu、CANN、device） |

**重叠类优化（P1-2、P5）另用窗口计时**：warmup 之后连续跑 N 步、期间不做任何设备同步，末尾同步一次再除以 N。原因是逐步同步会把 step 末尾排队的异步工作算进本步，正好掩盖要测的重叠收益；窗口计时报的是稳态吞吐，两种口径不能混比。

**测量环境**：真机数字均在 node1 的 `nzf-cai-dev` 容器（8×910B2，CANN 9.1，torch_npu 2.13.0.rc1）上取得。2026-11-09 起 venv 与仓库副本放在本地盘 `/mnt/disk0/jiangzs/niuzifan-local/`：NFS 上的 venv 单进程 import 要 165.8 s，`spawn` 的 dataloader worker 会成片卡在 NFS RPC 上，4 worker 的对照因此不可用；本地副本为 10.5 s。

**两类基准**

1. **训练步（离线 latent）**：2 样本/rank、bf16、pack 后 7388 token（AR 284 + DM 7104）、grad-accum=1、ZeRO-2；分别计时 forward / backward / optimizer+zero_grad，并给整步。
2. **VAE 离线编码**：batch 2、12 帧、544×736、fp16；分别计时整段与 Conv3D/TransData/Pad/Cat/逐元素分族；latent 与参考实现对拍。该模式随 6.3 的 V-1 一起加入 `bench_ops.py`（`--op vae`）；在此之前用独立脚本按本节条件测，并附同样三段式的 JSON。

对应的两条命令（只换一个量做 A/B，其余原样复制）：

```bash
python examples/worldmodel/cosmos3_harness/bench_ops.py --op all --device npu --num-tensors 512 --numel 131072 --output bench-optimizer.json
```

```bash
torchrun --nproc_per_node=<N> examples/worldmodel/cosmos3_harness/train.py --model <ckpt> --replay-pack <固定 bench pack> --save-dir <tmp> --save-interval 1000 --backend npu --precision bf16 --zero-stage 2 --steps 40 --bench-warmup 10 --bench-json bench-step.json
```

`--numel`、`--steps`、`<N>` 按真实分片与 pack 填；`train.py` 的 `--bench-json` 只由 rank 0 写出。

**验收门禁**

- 数值：VAE latent `max|Δ| ≤ 1` 个 fp16 ulp 且相对误差 ≤ 0.64× bf16 量化噪声；优化器一步后参数 `max|Δ| ≤ 1e-6`（相对 fp32 AdamW）；loss 连跑 3 步不漂。
- 性能：只有 median 与 min 同时改善才算通过；若 median 改善但 min 恶化，视为噪声，重新测。
- 归因：任何"融合"必须给出融合前后的 kernel 计数，不能只报墙钟。

---

## 七、证据缺口与待验证清单

按"会影响结论的程度"排序：

1. **参数量不自洽**：28 层 × 2048/9216 与 3.369 B 总数、1.423 B 可训练对不上。本文全部用相对量；若要做绝对工期推算，必须先核对 checkpoint 的真实参数量。
2. **`aten::to` 的调用点与关键路径占比未知**。这是 P0-1 的全部内容，也是本文唯一可能被显著低估的项。
3. **fp16 的分项占比是从 bf16 剖面折算的**，不是 fp16 逐 kernel 实测。P1/P2/P4 的秒数因此是 derived 而非 measured；P0-4 复测时要重新采一次 fp16 的 `kernel_details.csv`。
4. **3.3 的代数等价只在纯 Python 上验证过**（max|Δ| = 3.55e-14）。它作为生产路径已被 C0=16 的 issued-K 算术否决；剩下的用途是当 P1 的数值 oracle——需要在 NPU 上确认 fp16 下 max|Δ| 仍 ≤ 1 ulp，以及 `F.pad` 的左填充语义在 5 维下与 `WanCausalConv3d` 完全一致。
5. **P2 的"权重预转 FRACTAL_Z_3D"收益是上界估算**：权重按次数占 801 次的约 1/3，但按字节与时间远小于激活侧；0.002–0.005 s/样本 可能被归因结果进一步下调甚至归零。必须在 P1-3 的 `op_summary` 归因后看实际下降；此外要确认预转后的权重能否被 `aclnnConvolution` 直接接受，而不在每次前向重新做 NZ 转换。
6. **多流收益是 inferred**：optimizer 段 28–43% 的占用率是实测，但"VAE 能填进去多少"取决于两者对 Cube/MTE 的争用，需真机时间线。
7. **`npu_rotary_mul` 在 TND 布局下的 `B,N<1000` 约束语义未核实**：当前序列 7388 token，若该约束按 token 数解释则不可用，只能走自定义 kernel。替换前先在目标 torch_npu 版本上做最小用例。
8. **`npu_add_rms_norm` 没有独立 backward 符号**：反向组合方式（`npu_rms_norm_backward` + add）未在目标版本验证。
9. **Conv3D 的 41% 达成率来自"33 TFLOP"的推算**（由 233 次卷积的核尺寸反推），不是逐层 FLOP 实测；而且 41% 是全族均值，down0/down1 本就更高效、手写版又受 baseN=128 对 C_out=160 的浪费拖累，所以 P3 的 0.02–0.05 s 仍可能偏高。
10. **uint8 H2D 的收益未实测**：PCIe 带宽、pinned 内存、worker 解码耗时都要在目标容器里量。
11. **DVPP/AIPP 在 910B2 容器内的可用性未验证**：npu-guide 的 §4.4 讲的是能力，不是本容器的实际设备节点。
12. **16 卡没有分相位拆解**：fp16 VAE 约占 4.2 s 中 1.0 s 是按单卡 profiler 推算的，不是 16 卡实测；多流在 16 卡上的收益可能被跨节点通信掩盖。
13. **VAE 分块只扫到 32 帧、只测 480p**：256p/720p 的最优窗口可能不同，P1 的 padded buffer 尺寸与 tile 切分要按分辨率参数化。
14. **CP 只在单机 sp=2 跑通**：all-to-all 合并与 VAE 重叠在多机下都没有验证。
15. **Pad/Cat 的逐层归属没有逐条核实**：P1 只消 conv_in 这一层的 1/227 Pad 与 1/272 Cat，其余 226/271 次必须由手写卷积消掉。P1 的 0.005–0.010 s 依赖这个归因，必须在 `op_summary` 里按 shape 核实。
16. **优化器段的残余已归因到约 100 ms，但还不是逐 kernel 的 msprof 分解**：2026-11-09 在 node1 量到 unscale/clip（10.65 → 1.43 ms @ 432×400k）、AdamW 各臂（HybridAdam 138 / plain 21.6 / foreach 21.4 / fused 13.2 ms）、grad-norm 三种实现（63.8 / 45.6 / 4.1 ms）与整步 optimizer 段（8 卡 332 → 203 → 104–109 ms，1 卡 264 → 232 ms），见 6.1.2/6.1.3。同步插桩的绝对数不可用（前置同步把异步队列算进后一次调用，四项相加 550 ms/步 > 整段 233 ms），第二遍改用 NPU event 归因；残余部分的逐 kernel 分解仍留给 msprof（P0-2）。另外，fused 优化器"合并 buffer 只建一次"的语义曾让第一轮 A/B 的 fused 臂从第 2 步起用旧梯度（3.34e-03），已由复用梯度 buffer 修复（2.38e-07），见 6.1.3。
17. **P4(b)（残差进 conv epilogue）在 910B 上不成立**：FixPipe 没有 tensor residual 输入，L0C 也没有到 UB 的通路。已从设计里删除；若有人坚持，先在单层 microbench 上证伪。
18. **conv_in 这一层的 bound 类型未定**（cube 还是 MTE）：它决定 §3.3 的净亏取 0.005 还是 0.013 s/样本，也决定 P3 的收益上限。
19. **P1-2 的 async all-gather 实测为负，原因不是通信粒度**：2026-11-09 在 node1 用 8 卡窗口计时，`overlap_allgather` 让 4 worker 的步时从 2.206/2.262 s 涨到 2.543/2.527 s（0 worker 4.344/4.331 → 4.773/4.882 s）；相位插桩显示 optimizer 段两臂相同（0.203/0.200 s），多出来的是 `ColoParameter` + `ZeroOpHook` 的逐算子钩子（forward 0.19–0.22 → 0.37–0.41 s，8/8 rank）。纯通信微基准只差 13 ms（逐参数 73.6 vs 分桶含回写 60.6 ms）。旁路 stream 重叠可行（104.0 → 89.4 ms，约 60% 被计算盖住），但收益上限 60–74 ms/步，见 6.1.4。

---

## 附：本文与已有草稿的关系

| 草稿 | 本文新增 | 本文不重复 |
| --- | --- | --- |
| `draft_cosmos3_vae_optimization_zh.md` | P1 的数据入口、P2 的 TransData 归因、3.3 的代数融合裁决、P3 的逐层判据、P4/P6 的残余定位 | fp16/分块/RMSNorm 融合的实测 |
| `draft_cosmos3_training_pipeline_zh.md` | 优化器的带宽下限与融合路径、host `aten::to` 的调查方法、多流的占用率依据 | S1–S15 分步与提案清单 |
| `draft_cosmos3_profile_zh.md` | 用相位占比做 Amdahl 裁决、每层 FLOP 复核（53% 峰值） | 相位/kernel/显存实测 |
| `draft_cosmos3_16card_zh.md` | 固定通信 0.9 s 的三处可动点 | 16 卡 A/B 与工期推算 |

<!-- SECTION_ROADMAP -->
