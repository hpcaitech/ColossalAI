# Cosmos 3 的 VAE 编码：逐层 profiling 与底层优化方案

对象是 `train_sft.py` 每个 micro-batch 都要跑的冻结 VAE 编码（`Wan2.2-TI2V-5B-Diffusers/AutoencoderKLWan`）。
在 544×736×33 这个真实配方尺寸下，fp32 单样本编码 **1.229 s**，而同一张卡上 DiT 的一个 micro-batch 只有几百毫秒 ——
数据侧的开销几乎全在这一步。本文件回答三件事：时间具体花在哪、每个低效点的根因、以及按收益排序的改法。

> **当前状态（十三、十四节之后）**：fp32 原生 1.229 s/样本 → fp16 + 8 帧窗口 + fused RMSNorm + fast encoder
> **0.354 s/样本（3.08×）**，latent 与 fp32 的差是 bf16 转换误差的 0.615×。官方的 AOT 编译与
> `channels_last_3d` 在本机都不可用，原因分别见十二、六节。

## 一、测法与口径

| 项 | 值 |
| --- | --- |
| 设备 | 单卡 910B2（8 卡机器上的一卡，其余空闲） |
| 输入 | `[1, 3, 33, 544, 736]`，即 DROID 三路相机拼成 640×540 后映射到 480p 表的 `"4,3"` 档（736×544） |
| 权重 | 官方 fp32，`allow_hf32=False`、`float32_matmul_precision="highest"`（与训练完全一致） |
| 逐层计时 | `forward_pre_hook` 与 `forward_hook`，**每个钩子出口都 `npu.synchronize()`** |
| 逐 kernel 计时 | `torch_npu.profiler`（CPU+NPU），导出 `kernel_details.csv` 与 `trace_view.json` |

逐层计时必须同步，这一点值得单独说明：不同步时钩子测到的是 **CPU 派发**时间，而不是设备时间。
同一份代码不同步只测到 0.411 s，其中 0.817 s 落到钩子之外，看起来像"框架开销"，实际是设备在跑而 CPU 已经跑完。
同步后逐层之和 1.251 s，与墙钟 1.229 s 只差 1.8%（同步本身的开销），这个数才是可以拿来做决策的。

## 二、编码到底在算什么

1. **patchify**：`[B,3,T,544,736] → [B,12,T,272,368]`（2×2 空间折叠，通道 3→12）。
2. **因果 3D 卷积 + 特征缓存**：`_encode` 按 `1 + (T-1)//4` 切成 **9 个 chunk**，首个 chunk 只有 1 帧，其余 8 个各 4 帧；
   每个卷积把前一个 chunk 的末 2 帧 `cat` 进输入，保证时间因果。
3. **4 级下采样**（`WanResidualDownBlock`，通道 160/320/640/640）：每级 2 个残差块，每个残差块 2 个 3×3×3 因果卷积；
   级间还有一条 `AvgDown3D` 捷径（先 `x.clone()`，再 view/permute/mean）。
4. **mid block**：2 个残差块 + 1 个注意力（时间维已降到 1，只有 34×46=1564 个 token，占 0.3%）。
5. **head**：RMSNorm → SiLU → `conv_out`（→ 96 通道，即 mean/logvar）。

单样本约 **33 TFLOP**（fp32，按 233 次 conv3d 的卷积核尺寸推算）。实测卷积部分 0.681 s，折合约 **49 TFLOPS**。
这个数字很关键：它说明 **卷积本身已经跑到了 fp32 峰值的相当比例**，不是"写错了"的低效，而是**数学模式选错了**。

## 三、设备侧时间分布（fp32 基线）

设备 1.228 s 忙 / 1.230 s 墙钟 = **99.9% 占用**，4381 个 kernel 首尾相接。**单卡没有任何调度空隙可挖**，
所有收益只能来自让 kernel 本身变便宜。

### 按层（同步钩子，9 个 chunk 合计）

| 阶段 | 耗时 | 占比 | 说明 |
| --- | --- | --- | --- |
| down_blocks.0 | 0.558 s | 44.6% | 160 通道，T=4，272×368 —— 空间最大，卷积量占绝对多数 |
| down_blocks.1 | 0.330 s | 26.3% | 320 通道，T=2，136×184 |
| down_blocks.2 | 0.188 s | 15.0% | 640 通道，T=1，68×92 |
| mid_block | 0.078 s | 6.2% | 640 通道，34×46 |
| down_blocks.3 | 0.073 s | 5.8% | 640 通道，34×46 |
| conv_in | 0.015 s | 1.2% | 12→160 |
| conv_out + norm_out + SiLU | 0.010 s | 0.8% | |
| patchify / quant_conv / cat | 0.012 s | 1.0% | 可忽略 |

前两级吃掉 **71%**：优化必须盯着 level 0/1，其余是尾数。

### 按 kernel（`kernel_details.csv`，单次编码）

| kernel | 次数 | 总时间 | 占比 | 单次 |
| --- | --- | --- | --- | --- |
| `aclnnConvolution_Conv3dv2WithFlag_Conv3DV2` | 233 | 0.681 s | **55.5%** | 2.92 ms |
| `aclnnConvolution_TransData_TransData` | 801 | 0.213 s | **17.4%** | 266 µs |
| `aclnnConstantPadNd_PadV3AiCore_PadV3` | 227 | 0.063 s | 5.1% | 276 µs |
| `aclnnCat_ConcatD_ConcatD` | 272 | 0.034 s | 2.8% | 125 µs |
| `aclnnDiv_RealDivAiCore_RealDiv` | 198 | 0.034 s | 2.7% | 169 µs |
| `aclnnInplaceCopy_TransposeAiCore_Transpose` | 37 | 0.024 s | 2.0% | 658 µs |
| Adds / Silu / Mul / Muls / LpNorm / BroadcastTo | 1240 | 0.099 s | 8.0% | ~80 µs |
| Conv2D（Resample 里的 2D 卷积） | 45 | 0.015 s | 1.2% | 337 µs |

`TransData` 有 801 次、平均每次 266 µs —— 基本是**每个卷积 3 次**（输入、权重、输出各一次）。
这是 Ascend 的 Conv3D 在内部 fractal/NHWC 布局与 NCDHW 之间来回搬运张量，纯数据移动，不做任何计算。

## 四、四个具体低效点

1. **矩阵/卷积跑在 fp32 上（55.5%）**。配方为了数值精度关掉了 HF32，Conv3D 因此走非 Cube 的 fp32 路径。
   Ascend 的 Cube 单元在 bf16/fp16 上的吞吐是 fp32 的数倍，这是最大的一块。
2. **布局转换（17.4% + 2.0%）**。每个卷积前后都要 TransData/Transpose，因为 PyTorch 侧是 NCDHW，
   而 kernel 想要 NHWC(fractal)。
3. **RMSNorm 的仿射与精度往返（fp32 下 ~8%，bf16 下升到 24%）**。`WanRMS_norm` 是
   `F.normalize(x, dim=1)` 再 `* scale * gamma + bias`：一次 LpNorm、一次 Div、一次 BroadcastTo（expand）、
   两次逐元素乘、一次加；bf16 下还多出 441 次 Cast（`x.float()` 上采再 `to(x.dtype)` 下采）。
   注意 `gamma` 是逐通道缩放，**但折不进紧随其后的 conv**：每个 norm 后面都跟 SiLU，不是齐次函数
   （`SiLU(gamma*y) != gamma*SiLU(y)`）；`bias` 在本模型是 0，`scale` 是标量——收益只能从减少整张量读写来（见十三）。
   另外逐层数据显示：每个 norm 的标量乘/加都会触发一次 **device→host 同步**（单次编码 397 次，
   `aten::item` → `_local_scalar_dense`），这让 CPU 派发与设备执行完全串行 —— 设备占用 99.9% 时它不额外收费，
   但一旦 dtype 优化把设备时间压下去，CPU 派发（138 ms）就会成为新的下限。
4. **因果卷积的 pad 与 cache cat（7.9%）**。每个卷积前 `F.pad(x, (1,1,1,1,2,0))` 会把张量**放大成 (T+2)×(H+2)×(W+2) 再整块复制**；
   从第二个 chunk 起还要把 2 帧 cache `cat` 到输入上，又是一次整块复制。

## 五、已验证：#1 一个改动就够一半 —— 半精度

只把权重和激活换成半精度（其余结构不动），单样本编码：

| | fp32 | bf16 | fp16 |
| --- | --- | --- | --- |
| 编码墙钟 | 1.229 s | 0.501 s | **0.499 s** |
| 加速比 | 1× | 2.45× | **2.46×** |
| Conv3D kernel | 0.681 s | 0.225 s | — |
| TransData | 0.213 s | 0.040 s | — |
| 峰值显存 | 6.11 GiB | 3.11 GiB | 3.11 GiB |

`down_blocks.0` 从 0.558 s 降到 0.259 s（2.15×，最差），`down_blocks.1/2` 约 2.7×。
level 0 提升最小，因为它的逐元素/pad 比例最高 —— 这正好指出 dtype 之后该做什么。

### 精度：fp16 是那个"免费"的选择

latent 在 `packing.py` 里**本来就会被 `.to(torch.bfloat16)`**，所以门槛不是"零误差"，而是"不劣于这次已有的 bf16 量化"。
同一进程、同一颗固定种子、同一份权重，先 fp32 测一遍，再原地转 dtype 测第二遍：

| | 与 fp32 参考的平均差 | 相对 | 与 bf16 量化误差之比 |
| --- | --- | --- | --- |
| bf16 编码 | 2.19e-3 | 4.34e-3 | **3.07×**（比配方自带的量化噪声还大 3 倍） |
| fp16 编码 | 2.84e-4 | 5.64e-4 | **0.40×**（严格小于已有量化噪声） |
| 参考：fp32 latent 做 bf16 舍入 | 7.13e-4 | 1.41e-3 | 1.00× |

也就是说：**bf16 快，但它的误差是配方本来就要吃的量化噪声的 3 倍，不能算免费；fp16 一样快（2.46×），
误差只有那个噪声的 0.4 倍，等于没引入可观测的偏差。** 结论是编码跑 fp16，而不是 bf16。
（bf16 仍然可用作更激进的选项：3 倍量化噪声对条件 latent 通常仍可接受，但不该默认为它开绿灯。）

### 多卡多样本复核：8 卡 × 8 样本 = 64 个真实窗口

上面那组是随机噪声输入、单卡。真实数据未必一样：纯色桌面和快速运动的手腕在量化上表现不同，
而训练时每一张卡、每一个 micro-batch 都在编码不同的画面。所以补了一轮真实数据的全节点测量——
每卡一个独立进程（不建进程组、不走 HCCL）、各取自己的一段数据，每卡 2 个 batch × 4 样本，
共 **64 个互不相同的 DROID 窗口**（取自 8 个 episode，共 1491 个窗口的数据集），全部是配方尺寸 `[4,3,33,544,736]`。

| | 逐样本相对误差（64 个） | 排间中位数 | 卡间离散度 | 与 bf16 量化误差之比（64 个） |
| --- | --- | --- | --- | --- |
| fp16 编码 | 7.89e-4 – 9.00e-4 | **8.53e-4** | ±2.9% | **0.580 – 0.636** |
| bf16 编码 | 6.33e-3 – 7.31e-3 | 6.85e-3 | ±2.5% | 4.670 – 5.011 |

三点结论：

1. **卡与卡、样本与样本之间的差异很小**（变异系数 3% 以内）。误差是 dtype 的属性，不是画面内容的属性，
   这比"抽一个样本碰运气"要可信得多：64 个样本里 **没有一个** fp16 越过 1.0× 门槛，**没有一个** bf16 低于 4.6×。
2. **真实数据比随机噪声苛刻约 1.5 倍**：fp16 从 0.40× 升到 0.61×，bf16 从 3.07× 升到 4.87×。
   随机输入会让人低估风险——好在结论方向没变，fp16 依然严格在门槛内，bf16 依然不行。
   bf16 的误差本身是 fp16 的 **8.0 倍**。
3. 绝对误差随样本变化很大（fp16 `max|Δ|` 从 1.4e-2 到 1.9e-2，bf16 从 7.6e-2 到 1.1e-1），
   而相对误差几乎不动 —— 说明差的只是信号幅度不同的场景，模型行为没有分叉。
   这也是为什么验收用**相对误差**而不是绝对误差。

复现：`scripts/profile_vae_accuracy.py`（每卡一行，八卡并发），口径与上表一致。
覆盖面受限于 8 个 episode；若要再扩，把 `--max-episodes` 放大重跑同一脚本即可，口径不变。

## 六、被平台限制挡住的：#2 通道后置（channels_last_3d）

直接把模型和输入转成 `channels_last_3d` 会失败：

```
RuntimeError: Only contiguous_format or preserve_format is supported.
```

后来确认这个结论比最初判断的更彻底：**不是哪一个算子不接受 `channels_last_3d`，而是这个 torch_npu 版本根本
不支持这种内存格式**——连 `torch.randn(...).to(memory_format=torch.channels_last_3d)` 这一步就报同一句错
（`probe_channels_last.py`，fp32/fp16 都一样，先于任何模型代码）。所以"预分配 NHWC padded buffer"、"5D cat"
这类绕法都不可能成立，`channels_last_3d` 这条路在本机是死的；kernel 内部的 fractal 布局是唯一的"通道后置"，
用户侧够不到。

能做的只剩把**手工 pad 的那一半**去掉：H/W 的对称 padding 本来就该由 conv 自己的 `padding=` 表达，
只有因果 T 维的两帧前置 padding 是非对称的（而稳态下这两帧来自 feature cache，`pad_t` 直接归零）。
这就是第十四节实现的 `vae_fast_encoder`：稳态下不再有任何 `F.pad`。TransData（bf16 下 8%、fp32 下 17.4%）
是 conv 内部布局转换，仍拿不回来。

## 七、其余算子融合清单（按收益排序，半精度基线 0.499 s，占比取自 bf16 的 kernel 剖面）

| # | 改动 | 现在占比 | 预期收益 | 风险 |
| --- | --- | --- | --- | --- |
| 0 | **加大 encode 每次调用的帧数**（官方 480p 一次 24 帧；见十一：**本机实测只有 1.04–1.12×，显存线性增长，最优是 12 帧**） | 9 次调用 → 4 次 | 本机实测 1.04×（24 帧也只有 1.05×） | 低（因果 cache 跨调用传递，已用 latent 对比验证等价） |
| 1 | **RMSNorm 融合**（已实现，见十三）：`scale/norm` 先在 `[B,1,T,H,W]` 上算，去掉 fp32 升降与死的 `+0.0` | ~24% | **fp16 实测 0.927→0.748 s，1.24×**；fp32 仅 1.06× | 低（误差 0.615× bf16 转换，与原生的 0.604× 同级） |
| 2 | **H/W padding 交给 conv 自己做**（已实现，见十四）：对称方向交给 `conv3d(padding=)`，稳态下无 `F.pad` | 5.1%（Pad） | **fp16 实测 0.926→0.890 s，1.04×**；逐位不变 | 无（`torch.equal` 验证） |
| 3 | **去掉 `WanResidualDownBlock` 的 `x.clone()`**（已实现，见十四）：短路分支先算 | 含在 copy 里 | 时间 0.6%，**峰值 −0.53 GiB（fp16）/ −1.07 GiB（fp32）** | 无（`torch.equal` 验证） |
| 4 | `torch.compile` 兜底：把零散逐元素合成一个大 kernel | 15.1% | 0.02–0.05 s | 首次编译按 shape 计，分钟级；本机两条后端都走不通（见十二） |
| 5 | 多流重叠：VAE 与 optimizer step 放不同 stream | — | 取决于 DiT 时长 | 中（显存上升） |

0–3 做完后 fp16 的编码从 0.926 s 降到 **0.707 s**（见十四）；剩下的主要是 conv 内部的 TransData 与
`torch.compile` 才能拿到的逐元素融合，本机都够不到。相对 fp32 原生的 2.175 s 是 **3.08×**。

## 八、官方到底预不预提取 latent

之前的版本在这里留了一个待确认的问题（"增强发生在编码之前还是之后"）。现在有答案了：**官方不预提取，每个 batch 都在线编码**。
依据有两处：

- 技术报告 5.1（Training loop）：*"Also incorporating on-the-fly variational encoders (e.g., the Wan2.2 VAE), the pipeline
  operates end-to-end on raw multimodal inputs. This design eliminates offline latent-extraction phases and ensures that
  augmentation, encoding, and training remain in lockstep across runs."*
- 官方代码 `cosmos_framework/model/generator/tokenizers/wan2pt2_vae_4x16x16.py`：每次 `encode()` 都新建一份 `enc_cache`
  并只返回 latent；落盘的只有 AOT 编译出来的 chunk 函数（`.pt2`，用完即删），没有任何按 batch/step/sample 键的 latent 缓存。
  `use_cached_encoder` 是"一次请求内跨 chunk 的因果 cache"，不是跨步缓存。

为什么不预提取，理由就是上面那句 "remain in lockstep"：**增强作用在原始像素上，编码在增强之后**。
DROID 后训练明确用了 random image augmentation（报告 4.x），一旦把 latent 提前抽好，增强就被冻死在抽取那一刻；
而且改分辨率、改 aspect、改增强策略都要重抽 350 小时的数据集，缓存还会和 dataloader 的采样产生版本漂移。

官方对"VAE 占关键路径"这件事的回答不是把它拿掉，而是把它做快（报告 5.2.6）：

1. **加大每次调用的帧数**。Wan2.2 的默认粒度是"1 个 latent chunk = 首帧 + 4 帧"，报告说这个默认在训练分辨率下
   "each kernel launch operates on too little work to saturate the tensor cores"。官方改为每次调用编码一批**像素帧**：
   256p 68 帧、480p 24 帧、720p 12 帧（代码默认 `{"256": 68, "480": 24, "720": 8}`）。这一项贡献了"the bulk of the per-step speedup"。
2. **AOT 编译**。`torch.compile` 再降 52% 编码延迟；静态 shape 要求按 3 分辨率 × 5 长宽比 × 3 种 cache 状态 = **45 张图**分别编译。
   串行编译让启动多了约 15 分钟，于是他们把编译**按 data-parallel rank 分摊**（AOTInductor，每个 rank 编一张），
   产物写共享盘再各自加载，启动开销降到 1 分钟以内。
3. **已知帧数特化**。像 DROID 这种每条 clip 帧数固定的机器人数据集，直接按真实 shape 编译，
   绕开"补到 chunk 整数倍 + 再裁掉"的通用路径。

**对我们的直接含义**：我们现在走的 diffusers `AutoencoderKLWan._encode` 正是报告批评的那个默认粒度
（33 帧 → 9 次调用：1 + 8×4）。但第十一节的实测表明，第 1 条的前提（kernel 太小打不满）在 910B2 上不存在：
把窗口加到 24 帧只快 1.05–1.12×，显存却涨 3–4 倍，本机的最优值是 12 帧而不是 24
（这是对 diffusers 原生编码器而言；pad 下沉之后膝点消失，默认值已改为 8 帧，见十四）。
`--compile-vae` 对应第 2 条但没做 rank 分摊；DROID 固定 33 帧正好吃第 3 条。
这三条都不改变数值语义（因果 cache 跨调用传递），是纯粹的调度优化。

顺带保留两条单卡调度的结论：**单卡内部没有调度空间**（设备 99.9% 占用、kernel 首尾相接，多流只会让串行依赖图更复杂）；
真正的杠杆只有"多 rank 天然并行"和"VAE 与 optimizer step 放不同 stream"两处，后者收益受限于 DiT 只有几百毫秒。


## 九、实施顺序与验收

1. **把 fp16 编码接进训练**（`build_vae` 加 dtype 参数 + `packing.py` 不再强转 fp32），跑通 3 步；
   已做过门禁：随机输入 0.40×、真实数据 64 个窗口最差 0.64× 量化噪声。
2. **分块编码**（第十一节）。原结论是 12 帧；十四节在 fast encoder 之后重测发现膝点消失，默认值改为
   **8 帧**（5 次调用、0.700 s，比 12 帧省 1.7 GiB 峰值），`0` 回到 diffusers 的 4 帧循环。
   接线已在单卡上跑通一步：`--vae-dtype float16 --samples-per-rank 2`，
   `[pack] samples=2 AR=284 DM=7104`、`[step 1] loss 2.534409`、峰值 36.81 GiB（单卡不分片，含 bf16 主干与优化器）。
3. ~~七.1 RMSNorm 融合~~ **已完成**（十三）：1.24×，误差 0.615× bf16 转换，峰值 −1.05 GiB。
4. ~~六.1 padded buffer~~ 改为十四的"padding 下沉 + 去 clone"：`channels_last_3d` 本机根本不支持（六），
   而稳态下 `F.pad` 可以完全消失；逐位不变，1.04×，峰值再降 0.5 GiB（fp16）。
5. `torch.compile`（官方第 2 条）已在十二节试过两个后端，本机都不可用，不再作为路线。
6. 每一步都用同一套工具复测：`profile_vae_deep.py`（逐层，同步）+ `analyze_vae_kernels.py`（逐 kernel）
   + `analyze_vae_trace.py` + `profile_vae_chunk.py` / `profile_vae_norm.py` / `profile_vae_conv.py` /
   `profile_vae_recipe.py`（窗口与各优化的 A/B）。

验收口径：**单样本 VAE 编码 ≤ 0.60 s（相对 fp32 基线 ≥ 2×）**，且 latent 与 fp32 参考的差异不超过
bf16 量化误差的 1.0 倍。当前生产配置（fp16 + 8 帧 + fused norm + fast encoder）实测 **0.354 s/样本
（0.707 s / batch 2，相对 fp32 原生 3.08×），0.615× 量化噪声**，两条都已满足。

## 十、对启动方案的影响（顺带修的另一个坑）

VAE 显存减半直接抬高"每卡能塞多少样本"的天花板，但真正卡住大 batch 的不是显存，是 `/dev/shm`：

| 每卡样本数 | 模型+优化器阶段峰值 | 结果 |
| --- | --- | --- |
| 2 | 18.01 GiB | 通过 |
| 4 | 24.93 GiB | 通过 |
| 8 | 38.4 GiB（模型推算） | 旧代码在 shm 上崩，与显存无关 |
| 12 | 需 ~51 GiB | **显存 OOM**（43.83 GiB 已分配 + 需要 7.23 GiB） |
| 16 | 需 ~65 GiB | 必然 OOM（已停掉，未跑完） |

单个视频张量 `[3,33,544,736]` fp32 就是 158 MB；4 个样本一个 batch、8 个 rank、4 个 worker、prefetch 2，
共享内存里同时压着几十 GB 的 tensor，容器只给了 32 GB 的 `/dev/shm`，于是 `samples_per_rank=4` 起就会
`unable to allocate shared memory ... No space left on device`。修法是把共享策略切到 `file_system`
（落到 `/tmp`，那里有几个 TB），已在 `train_sft.py` 里加好，可用 `COSMOS3_SHM_STRATEGY` 覆盖：
`torch.multiprocessing.set_sharing_strategy(os.environ.get("COSMOS3_SHM_STRATEGY", "file_system"))`。

结论：**每卡 8 个样本是这个配方在 8 卡单机上的实际上限**（显存 38 GiB 左右，留出余量），12 已经越界。

`file_system` 只是把问题挪到磁盘：每卡每轮要在 `/tmp` 里写十几 GB 的临时文件。如果实测发现取数变慢，
更好的修法是**让 worker 直接发 uint8**（原始视频本来就是 uint8，`[3,33,544,736]` 从 158 MB 降到 40 MB），
在 H2D 之后再转 float —— 一次把共享内存、临时文件、H2D 带宽三样同时降 4 倍。那属于数据链路的一次改动，
不在这次 profiling 的范围内。

## 十一、分块编码在 910B2 上的实测：官方 24 帧不是本机最优

第八节把官方的三条优化列了出来，其中第一条（加大每次编码调用的帧数）官方称贡献了 "the bulk of the per-step
speedup"，配置是 256p/480p/720p = 68/24/12 帧（代码默认 `{"256":68,"480":24,"720":8}`）。
**但那是 NVIDIA 训练硬件上测出来的数字**：它解决的前提是"每次 kernel 的工作量太小，打不满 tensor core"。
910B2 上这个前提不成立——第三节已经测到编码器设备占用 99.9%、4381 个 kernel 首尾相接。
所以这一条不能照抄，要按本机重扫。这一节就是重扫的结果。

### 测法

`scripts/profile_vae_chunk.py`，单卡（编码是每 rank 独立，其余七张卡只会加噪声），真实 DROID 窗口
（同一 canvas 的样本组成一个 batch，取数据集的前 229 个窗口），每档 = 1 次预热 + 5 次计时 + 1 次取 latent。
窗口宽度扫描 `native, 4, 8, 12, 16, 20, 24, 28, 32`：`native` 是 diffusers 自带的 `_encode`（1 首帧 + 8×4，
共 9 次调用），其余是 `vae_chunked.enable_chunked_encode` 换上的窗口宽度；33 帧在 12 帧窗口下是 1+12+12+8 = 4 次调用，
24 帧下是 1+24+8 = 3 次。

误差口径与第五节一致：`mean|Δ| / mean|latent|`，再除以同一个 latent 的 bf16 量化误差
（`packing.py` 本来就要做的那次舍入），小于 1.0 就算"没有引入可观测的偏差"。

### 结果

**batch 1、fp32**（单样本，基线 1.229 s）

| 窗口 | 每次编码的调用数 | 墙钟 | 加速 | 相对误差 | / bf16 量化 | 峰值 GiB |
| --- | --- | --- | --- | --- | --- | --- |
| native | 9 | 1.229 s | 1.00× | 0 | — | 6.11 |
| 4 | 9 | 1.228 s | 1.00× | **0.000e+00** | 0.000 | 6.11 |
| 8 | 5 | 1.145 s | 1.07× | 2.92e-07 | 0.000 | 8.03 |
| 12 | 4 | 1.119 s | 1.10× | 2.95e-07 | 0.000 | 9.94 |
| 16 | 3 | 1.344 s | **0.91×** | 2.97e-07 | 0.000 | 11.86 |
| 20 | 3 | 1.180 s | 1.04× | 2.96e-07 | 0.000 | 13.52 |
| 24 | 3 | **1.094 s** | **1.12×** | 2.95e-07 | 0.000 | 15.43 |
| 28 | 3 | 1.481 s | **0.83×** | 2.93e-07 | 0.000 | 17.35 |
| 32 | 2 | 1.380 s | 0.89× | 2.96e-07 | 0.000 | 19.27 |

**batch 1、fp16**（基线 0.496 s）

| 窗口 | 每次编码的调用数 | 墙钟 | 加速 | 相对误差 | / bf16 量化 | 峰值 GiB |
| --- | --- | --- | --- | --- | --- | --- |
| native | 9 | 0.496 s | 1.00× | 0 | — | 3.25 |
| 4 | 9 | 0.497 s | 1.00× | **0.000e+00** | 0.000 | 3.25 |
| 8 | 5 | 0.477 s | 1.04× | 1.71e-04 | 0.121 | 4.45 |
| 12 | 4 | 0.477 s | 1.04× | 1.74e-04 | 0.124 | 5.64 |
| 16 | 3 | 0.551 s | 0.90× | 1.74e-04 | 0.124 | 6.84 |
| 20 | 3 | 0.501 s | 0.99× | 1.75e-04 | 0.124 | 7.90 |
| 24 | 3 | **0.465 s** | **1.07×** | 1.74e-04 | 0.124 | 9.10 |
| 28 | 3 | 0.596 s | 0.83× | 1.74e-04 | 0.123 | 10.29 |
| 32 | 2 | 0.567 s | 0.88× | 1.75e-04 | 0.124 | 11.49 |

**batch 2、fp16**（16 卡那两次实测的每卡形状，基线 0.962 s）

| 窗口 | 每次编码的调用数 | 墙钟 | 加速 | 相对误差 | / bf16 量化 | 峰值 GiB |
| --- | --- | --- | --- | --- | --- | --- |
| native | 9 | 0.962 s | 1.00× | 0 | — | 5.18 |
| 8 | 5 | 0.923 s | 1.04× | 3.67e-05 | 0.026 | 7.57 |
| 12 | 4 | 0.926 s | 1.04× | 8.81e-06 | 0.006 | 9.96 |
| 16 | 3 | 1.089 s | **0.88×** | 0.000e+00 | 0.000 | 12.35 |
| 20 | 3 | 0.975 s | 0.99× | 0.000e+00 | 0.000 | 14.49 |
| 24 | 3 | **0.914 s** | **1.05×** | 8.81e-06 | 0.006 | 16.87 |

**batch 4、fp16**（一次训练 step 的真实形状，基线 1.981 s）

| 窗口 | 每次编码的调用数 | 墙钟 | 加速 | 相对误差 | / bf16 量化 | 峰值 GiB |
| --- | --- | --- | --- | --- | --- | --- |
| native | 9 | 1.981 s | 1.00× | 0 | — | 9.04 |
| 4 | 9 | 1.981 s | 1.00× | **0.000e+00** | 0.000 | 9.04 |
| 8 | 5 | 1.961 s | 1.01× | 3.66e-05 | 0.026 | 13.83 |
| 12 | 4 | 1.878 s | 1.05× | 3.66e-05 | 0.026 | 18.60 |
| 16 | 3 | 1.939 s | 1.02× | 3.69e-05 | 0.026 | 23.38 |
| 20 | 3 | 1.881 s | 1.05× | 3.66e-05 | 0.026 | 27.65 |
| 24 | 3 | **1.862 s** | **1.06×** | 3.64e-05 | 0.026 | 32.43 |
| 28 | 3 | 1.910 s | 1.04× | 3.67e-05 | 0.026 | 37.21 |
| 32 | 2 | 1.933 s | 1.02× | 3.67e-05 | 0.026 | 41.99 |

### 三点结论

1. **实现是对的，且数值等价**。chunk=4 走我的实现时，latent 与 diffusers 原生路径**逐位相同**
   （相对误差 0.000e+00，墙钟与峰值也一模一样），说明因果 cache 的跨调用传递接对了；
   8–32 帧的差异在 3e-7（fp32）/ 3.6e-5（fp16），只有配方自带 bf16 量化噪声的 0.03–0.12 倍。
   也就是说，分块只改变浮点求和顺序，不是一次数值取舍。
2. **收益远小于官方**。最大 1.12×（fp32、单样本、24 帧）；训练形状下 12/20/24 帧都只有 1.04–1.06×。
   原因很直接：本机编码器不是"launch 打不满"。batch 从 1 加到 4，耗时严格线性（0.496 → 0.962 → 1.981 s），
   说明 **4 帧 × 1 个样本这一档就已经把机器喂饱了**，加大窗口省下的只是每次调用的 pad/cat/launch 残差。
3. **显存线性增长**：每多 4 帧约 +0.6 GiB（fp32、batch 1）；训练形状下 9.04 → 32.43 GiB（24 帧）→ 42.0 GiB（32 帧）。
   而显存正是每卡样本数的天花板（第十节），所以官方那个 24 帧在本机等于"用 23 GiB 换 0.12 s"。

### 本机该取哪个值

三档 batch 放在一起看（相对原生的加速）：

| 窗口 | batch 1 | batch 2 | batch 4 | 峰值（batch 4） |
| --- | --- | --- | --- | --- |
| 8 | 1.04× | 1.04× | 1.01× | 13.8 GiB |
| 12 | 1.04× | 1.04× | 1.05× | 18.6 GiB |
| 24 | 1.07× | 1.05× | 1.06× | 32.4 GiB |

8 帧和 12 帧在 batch 1/2 打平，但 batch 4 上 8 帧掉到 1.01×；24 帧始终只比 12 帧快 0.5–2%，
显存却是它的 1.7–2 倍。所以本机的膝点是 **12 帧**：相对原生 1.04–1.05×，显存只多 5–10 GiB。
`--vae-chunk-frames` 的默认值取 12，`0` 表示回到 diffusers 自己的循环（代价只有 4–5%）。

> 十四节补测：把 pad 下沉、clone 去掉之后，这个膝点**消失了**——4/8/12 帧的耗时都在 0.70–0.72 s，
> 而显存仍随窗口线性增长（4.9 / 6.6 / 8.4 GiB）。默认值已改为 **8 帧**。本节"12 帧最优"的结论只对
> diffusers 原生编码器成立，原因是宽窗口的收益本来就是摊薄每次调用的 pad+cat。

另外注意 16/28/32 三档的**回退**（batch 2 下 16 帧比 12 帧慢 18%，batch 1 下慢 20%，28 帧慢 32%，
两种 dtype 都重现，不是噪声）；batch 4 时这个差距缩到 3%。这是某些时间长度上 conv3d 选了更差的切分造成的，
说明**分块宽度不是单调收益，必须在本机实测**，不能按"越大越好"外推。
（顺带一个反直觉的细节：batch 2 下 16 帧和 20 帧的 latent 与原生路径**逐位相同**，而 8/12/24 帧反而有 1e-5 量级的差——
同一份权重、同一个输入，只有窗口宽度不同。可见差异来自 kernel 在不同 T 下选的切分，而不是分块本身。）

### 对工期推算的影响

16 卡 fp16 实测里每个 rank 一个 micro-step 编码 2 个样本，batch 2 下原生 0.962 s、12 帧 0.926 s，
1.04× 只省 0.036 s，相对 3.3 s 的 micro-step 是 1%——低于日志 1 s 的分辨率，端到端测不出来。
所以上一份文档里"分块编码是最大一块可挖收益"的说法**只对 NVIDIA 硬件成立**，在 910B2 上不成立：
本机的 VAE 时间主要在 fp32→fp16（2.46×，已做），而不是在调用粒度上。

### 复现

```bash
# 单卡窗口扫描（batch 2, fp16）
dev-container.sh /home/niuzifan/scripts/run_vae_chunk_canonical.sh

# 分块语义的单元测试（CPU，无需权重）
dev-container.sh /home/niuzifan/scripts/run_chunk_tests.sh
```

代码：`examples/worldmodel/cosmos3/vae_chunked.py`（`enable_chunked_encode`，按窗口宽度替换 `_encode`）、
`scripts/profile_vae_chunk.py`（扫描脚本）、`tests/test_vae_chunked.py`（窗口切分、补齐裁剪、宽度校验）、
`train_sft.py --vae-chunk-frames`（十四节之后默认 8；本节测量时是 12，0 = diffusers 原循环）。

## 十二、官方第 2 条（AOT 编译）在本机两条路都走不通

官方用 `torch.compile`（AOTInductor）把编码再降 52%。在本机把 `vae._encode` 交给 `torch.compile`
（`--compile-vae`，batch 2 / fp16 / 12 帧窗口），两个后端各试了一次，结论都是不可用，但原因不同。

### 后端 A：`inductor` —— 编不出来

```
LoweringException: AssertionError:
  target: aten.sum.dim_IntList
...
torch_npu/_inductor/__init__.py:77: UserWarning: triton-ascend is not installed, install it first.
```

`aten.sum.dim_IntList` 来自 `WanRMS_norm` 的 `F.normalize(x, dim=1)`（沿通道维求范数）。
这不是 VAE 特有的问题：直接 `torch.compile(lambda x: torch.sum(x, dim=1), backend="inductor")`
跑一个裸的 `sum`，报的是同一个断言。torch_npu 的 inductor 后端整条路依赖 Triton 代码生成
（`has_triton()` 不满足就只剩警告），而 Ascend 的 Triton 就是 `triton-ascend`，本镜像没有装。

### 后端 B：`npugraph_ex`（torchair，Ascend 原生图引擎）—— 跑得通，但慢 18 倍

这个后端不需要 Triton，走 ACL/GE 图。先用最小算子验证它可用：

| 算子 | `inductor` | `npugraph_ex` |
| --- | --- | --- |
| `sum(x, dim=1)` | 失败（同一个断言） | 通过，与 eager **逐位相同** |
| `rsqrt(sum(x*x, dim=1))` | 失败 | 通过，逐位相同 |

换到 VAE 上（batch 2 / fp16 / 12 帧）：

| 路径 | 首次调用（含建图） | 稳态每次编码 | 峰值 |
| --- | --- | --- | --- |
| eager | 0.9 s | **0.926 s** | 9.96 GiB |
| `npugraph_ex` | 21.0 s | **16.629 s（慢 18×）** | 7.54 GiB |

**慢的原因（推断，未逐层验证）**：这个编码里有 397 次 `aten::item`——`WanRMS_norm` 的标量乘/加引起
device→host 同步（见第四节）。图捕获必须在每次 host 同步处断开，于是整张图被切成几百个碎片，
每片都要付一次捕获/下发/布局转换的代价；日志里反复出现的
`Cannot create tensor with internal format ... will be created with base format` 也指向同一件事：
图边界处张量被迫回到 base format，等于把省下来的 TransData 又加回去。18× 的回退与这个机制自洽。

另一个佐证：稳态单次耗时（16.6 s）与首次建图（21.0 s）是同一量级，看起来更像**每次调用都在重建图**，
而不是"建好一次、之后回放"。如果要继续追这条线，该调的是 torchair 的缓存/模式开关
（`torchair.CompilerConfig` 的 `mode`、`cache_compile`、`enable_cache` 一类），而不是继续调窗口宽度。

### 结论与替代路线

官方第 2 条优化在 910B2 上**目前拿不到**：inductor 需要 `triton-ascend`（装了也未必能过低规约下降），
`npugraph_ex` 能跑但慢一个数量级。`--compile-vae` 因此保持默认关闭，后端默认 `npugraph_ex`
（它至少是正确的），实测数字写在帮助文字里。

替代路线还是**手工融合**，按收益排序：

1. **RMSNorm 融合**（七.1，已实现并实测）：不去折 `gamma`——每个 norm 后面都跟 SiLU，`gamma` 折不进卷积——
   而是把除法从激活挪到标量：`scale/norm` 在 `[B,1,T,H,W]` 上算好再乘，去掉 fp32 升降两次整张量的搬运和
   死的 `+0.0`，每个 norm 从 6 遍激活读写降到 2 遍。数学上只多一次输入精度的舍入，实测 1.24×（fp16），
   误差 0.615× bf16 转换，峰值显存还低了 1 GiB。详见十三。
2. padded buffer 替掉 pad+cat（六.1），把 TransData 吃回去。
3. 若前两步做完还想再要 compile，再考虑装 `triton-ascend`；那时 RMSNorm 已不再是障碍。

所以 910B2 上 VAE 提速的路线是：**dtype（已完成，2.46×）→ 分块编码 12 帧（已完成，1.04×）
→ RMSNorm 手工融合（已完成为 1.24×，见十三）→ padding 下沉 + 去 clone（已完成为 1.05× 且逐位不变，见十四）**，
合计 3.08×。`torch.compile` 那条路本机走不通（十二），`channels_last_3d` 本机根本不支持（六）。




## 十三、RMSNorm 融合：1.24×（fp16），误差不动，峰值反降

十二节说明官方第 2 条（AOT 编译）在 910B2 上拿不到，替代路线是手工做 compile 会做的融合。第一个目标是
RMSNorm——它占编码时间的 ~24%，每个 residual block 里有 2 个，整个编码器一共 52 个模块（每次编码按窗口调用数十次）。

### 为什么不能把 `gamma` 折进后面的卷积

最初的设想是把仿射折进后一个 conv（conv 是线性的，`conv(gamma * x) = conv(gamma * w)` 只有在
`gamma` 逐通道且 conv 紧跟其后时成立）。实际结构不允许：diffusers 的 `WanResidualBlock` 里每个
`WanRMS_norm` 后面都跟一个 SiLU，而 SiLU 不是齐次函数，`SiLU(gamma * y) != gamma * SiLU(y)`。
逐通道的 `gamma` 也折不进逐元素的非线性。所以收益只能从**减少整张量读写次数**来。

### diffusers 原版在做什么

`WanRMS_norm.forward`（fp16 输入）：

```python
normalized = F.normalize(x.float(), dim=1).to(x.dtype)   # 升 fp32、除法、降回 fp16
return normalized * self.scale * self.gamma + self.bias   # 两次乘、一次加
```

三件事值得注意：

1. `x.float()` 把整张激活复制成 fp32，再 `to(x.dtype)` 复制回来——两次全尺寸搬运，只是为了让除法在 fp32 里算；
2. `self.bias` 在这个 checkpoint 里是 Python 浮点 `0.0`（模块以 `bias=False` 构造，52 个 norm 里 0 个有 bias 张量），
   那个 `+ 0.0` 是一次纯属浪费的全尺寸 kernel；
3. `* self.scale * self.gamma` 是两次全尺寸乘，而 `scale` 是标量——可以先把它并进除法。

即：6 遍激活读写（升、除、降、乘 scale、乘 gamma、加 0）。

### 替换版

```python
norm = torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True, dtype=torch.float32).clamp_min(1e-12)
scale = (self.scale / norm).clamp_max(torch.finfo(dtype).max).to(dtype)  # [B,1,T,H,W]
out = x * scale          # 一次全尺寸乘
out.mul_(self.gamma)     # 一次全尺寸乘，原地
```

- 归约用 `dtype=fp32` 在 kernel 内部完成，**不再物化 fp32 副本**——精度该在哪儿还在哪儿；
- `scale / norm` 只作用在 `[B,1,T,H,W]` 的小张量上（比激活小 `C` 倍），除完一次转成输入精度；
- `+ 0.0` 直接消失；
- 只剩 2 遍全尺寸读写。

两个数值上的注意点：`clamp_min(1e-12)` 与原版一致（原版 `F.normalize` 的 eps 就是 1e-12）；
`clamp_max(finfo.max)` 防的是"全零窗口 → `scale/1e-12` 溢出成 inf，再乘 0 得 nan"——原版先除张量
（`0/1e-12 = 0`）不会溢出，融合后的顺序必须先饱和。单测 `tests/test_vae_fused_norm.py` 覆盖了这两种情况，
以及 fp32/fp16、`channel_first` 的 5-D 与 `images` 的 4-D 两种布局。

### 实测

`scripts/profile_vae_norm.py`，单卡，真实 DROID 窗口，batch 2（与第五节同口径），12 帧窗口，
每档 1 次预热 + 5 次计时 + 1 次取 latent；两臂唯一的差别是 norm 的实现。参考 latent 是 fp32 原生版。

| dtype | norm | wall s | vs 原生 | rel err | vs bf16 转换 | 峰值 GiB |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | 原生 | 2.174 | 1.00× | 0 | 0.000 | 17.22 |
| fp32 | 融合 | 2.044 | **1.06×** | 4.5e-07 | 0.000 | 17.22 |
| fp16 | 原生 | 0.927 | 1.00× | 8.5e-04 | 0.604 | 9.96 |
| fp16 | 融合 | 0.748 | **1.24×** | 8.7e-04 | 0.615 | **8.91** |

四点：

1. **fp16 才是收益所在，fp32 只有 1.06×**。fp32 下本来就没有升降精度，省掉的只是死的加法与一次除法；
   fp16 下那两次 `TransData` 是实打实的搬运，去掉后 0.927 → 0.748 s，省 0.18 s（原耗时的 19%）。
2. **误差没有变差**：融合版 0.615× bf16 转换，原生 0.604×，两者都在"比 `packing.py` 本来就要做的
   bf16 舍入还小"的同一侧，远低于 1.0 的门槛。多出来的那一次舍入（先舍 `scale/norm` 再乘）在 fp32 下
   只有 4.5e-07 的相对差。
3. **峰值显存反而低了 1.05 GiB**（9.96 → 8.91 GiB）：不再物化 fp32 的激活副本。
4. 与 dtype 那一步合起来：fp32 原生 2.174 s → fp16 融合 0.748 s，**2.91×**。

### 接线

`vae_fused_norm.enable_fused_rmsnorm(vae)` 把 52 个 `WanRMS_norm` 的 `forward` 换成融合版；
`train_sft.py` 的 `--fuse-vae-norm`（默认**开**，`--no-fuse-vae-norm` 可显式关闭）在 `build_vae` 里调用它，
启动日志会打印 `norm=fused`。它只替换前向、不新增参数，权重仍是原 checkpoint 的。

### 端到端

同一个单卡 1 步（fp16、12 帧窗口、batch 2）：`[vae] dtype=float16 window=12 frames per encoder call, norm=fused`，
`[pack] samples=2 AR=284 DM=7104`，`[step 1] loss 2.533458`，峰值 36.81 GiB。
基线（同一配方、diffusers 原生 norm）是 `[step 1] loss 2.534409`、峰值 36.81 GiB——loss 差 9.5e-4（0.04%），
与 latent 的 8.7e-04 相对差同量级，显存完全相同。收尾时 DataLoader worker 在解释器退出阶段被 abort
（`terminate called without an active exception`），发生在 `done` 之后，基线那次没有触发，属于退出竞态，不影响结果。

### 复现

```bash
python scripts/profile_vae_norm.py --checkpoint /mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-Policy-DROID \
    --data /mnt/disk0/jiangzs/datasets/Cosmos3-DROID/success --batch 2 --iters 5
```

至此 910B2 上的 VAE 在线编码从 fp32 原生的 2.174 s 降到 0.748 s。下一步按七节清单是 padded buffer
替掉 pad+cat（省 0.03–0.04 s），然后是 `WanResidualDownBlock` 的 `clone`。

## 十四、把 padding 交给卷积、去掉无效 clone：再快 1.05×，逐位不变，峰值再降 0.5 GiB

十二节的结论是官方第 2 条（AOT 编译）在本机走不通，替代路线是手工做 compile 会做的那类变换。十三节做了
RMSNorm（去掉多余的整张量读写），这一节做卷积路径上的两处——它们比 RMSNorm 更"免费"，因为结果与 diffusers
**逐位相同**，不引入任何舍入。

### 改动一：对称 padding 交给卷积自己做

`WanCausalConv3d.forward` 在每次 3×3×3 卷积前调用 `F.pad(x, (1,1,1,1,2,0))`，把激活复制成一个
T+2、H+2、W+2 的张量。bf16 剖面里这是 `aclnnConstantPadNd` 227 次、占 5.1%。

三个方向里只有 T 维是**非对称**的（前补两帧、后不补——因果性的来源），H/W 都是对称的，正是 conv 自带
`padding=` 参数的含义。而 T 维的前两帧在稳态下由 feature cache 提供，`pad_t` 直接减到 0。于是：

- 稳态（第 2 次 encoder 调用起）：**没有任何 `F.pad`**，H/W 交给 `F.conv3d(..., padding=(0, ph, pw))`；
- 首帧那次（无 cache）：只补 T 维两帧，张量从 `(T+2)(H+2)(W+2)` 降到 `(T+2)HW`。

### 改动二：`WanResidualDownBlock` 的 clone 是无效的

diffusers 的 `forward` 第一行是 `x_copy = x.clone()`，供残差栈跑完后 `avg_shortcut(x_copy)` 使用。但残差栈
只是把 `x` 重新绑定到新张量，从不原地写输入，所以短路分支完全可以先算：`shortcut = avg_shortcut(x)`，
循环结束后 `x + shortcut`。最宽那级 block 的激活是 `160×4×272×368`（fp16 下 128 MB），每次编码 4 个 block
各复制一遍。

### 实测

`scripts/profile_vae_conv.py`，与十三节同口径（batch 2、12 帧窗口、真实 DROID 窗口、5 次计时）。四臂：
`native`、`fast_conv`（只改卷积）、`fast_conv+norm`（再加十三节的融合）、`fast_enc+norm`（再去掉 clone）。

| dtype | arm | wall s | vs 原生 | rel err | vs bf16 转换 | 峰值 GiB |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | native | 2.175 | 1.00× | 0 | 0.000 | 17.22 |
| fp32 | fast_conv | 2.102 | 1.03× | 0 | 0.000 | 17.17 |
| fp32 | fast_conv+norm | 1.972 | 1.10× | 4.5e-07 | 0.000 | 17.17 |
| fp32 | fast_enc+norm | 1.964 | 1.11× | 4.5e-07 | 0.000 | **16.10** |
| fp16 | native | 0.926 | 1.00× | 8.5e-04 | 0.604 | 9.96 |
| fp16 | fast_conv | 0.890 | 1.04× | 8.5e-04 | 0.604 | 9.96 |
| fp16 | fast_conv+norm | 0.711 | 1.30× | 8.7e-04 | 0.615 | 8.89 |
| fp16 | fast_enc+norm | **0.707** | **1.31×** | 8.7e-04 | 0.615 | **8.36** |

三点：

1. **`fast_conv` 的 rel err 与 native 完全相同**（fp16 都是 8.514e-04，fp32 都是 0）——不是"接近"，是逐位相同；
   单测 `tests/test_vae_fast_encoder.py` 用 `torch.equal` 断言了这一点，覆盖了有无 cache、1~2 帧 cache、
   3×3×3 与 (3,1,1) 两种核、stride 1 与 2、两种精度。
2. **只省下 0.036 s（1.04×），而不是 Pad 在剖面上的 5.1%**：卷积自己做 padding 也不是零成本，
   一部分省下的时间被 conv 内部的边界处理吃掉。但它是免费的，没有理由不做。
3. **clone 那一步几乎不省时间（0.711 → 0.707），省的是显存：fp16 峰值 8.89 → 8.36 GiB，
   fp32 17.17 → 16.10 GiB**。原因是 clone 的峰值与激活同寿命；去掉后 `avg_shortcut` 的输出成为常驻的那一份。

与前面几步合起来：fp32 原生 2.175 s → fp16 + 12 帧窗口 + fused norm + fast encoder **0.707 s，3.08×**。

### 窗口宽度：优化做完以后，十一节那个膝点消失了

十一节测出 12 帧是 910B2 的膝点，依据是"每次 encoder 调用要付一次 pad+cat 的复制"。`fast_encoder` 把 pad
去掉之后，这个前提不存在了，窗宽必须重测（`scripts/profile_vae_recipe.py`，生产配置：fused norm +
fast encoder，fp16，参考 latent 用 fp32 编码器）：

| 窗口 | 调用数 | wall s | vs 4 帧 | 峰值 GiB |
| --- | --- | --- | --- | --- |
| 4 | 9 | 0.719 | 1.00× | 4.91 |
| 8 | 5 | 0.700 | 1.03× | 6.64 |
| 12 | 4 | 0.706 | 1.02× | 8.37 |
| 16 | 3 | 0.872 | 0.82× | 10.10 |
| 20 | 3 | 0.765 | 0.94× | 11.57 |
| 24 | 3 | 0.708 | 1.02× | 13.30 |

（4/8/12 三行用 5 次与 10 次计时各跑一遍，两轮数字完全相同；16/20/24 取自 5 次那轮。）

读法：4/8/12 帧的耗时都在 0.70–0.72 s，差别在噪声量级，而峰值显存随窗口近似线性增长
（4.9 → 6.6 → 8.4 GiB）。也就是说**宽窗口不再用时间换时间**，只在用显存换更少的调用次数。16 帧那档依旧
回退（kernel 在该 T 下选了更差的切分），20/24 也没回到 8 帧的水平。所以默认窗口从 12 改成 **8**：
5 次调用、比 4 帧快 1.03×、比 12 帧省 1.7 GiB 峰值。十一节"12 帧最优"只对 diffusers 原生编码器成立——
那 4–5% 的收益本来就是摊薄每次调用的 pad+cat，pad 没了，收益也就没了。

8 帧还有一个附带的好处：33 帧正好拆成 1 + 8×4，**没有末窗补齐、也没有裁剪**（12 帧是 1 + 12 + 12 + 8，
末窗短一截）。所有调用形状一致，这对官方推崇的"按精确 shape 特化"是更干净的前提——将来若 `torch.compile`
可用，需要特化的图更少。

### 接线

`vae_fast_encoder.enable_fast_encoder(vae)` 同时打两个补丁（`enable_fast_conv` / `enable_fast_downblock`
也可单独调用），`train_sft.py` 的 `--fast-vae-encoder`（默认开）在 `build_vae` 里调用；启动日志打印
`conv=fast`。`--vae-chunk-frames` 的默认值随之改为 8（见上一小节）。它不新增参数，权重仍是原 checkpoint 的。

### 端到端

默认配方（不再显式传任何 VAE 选项，除 dtype）单卡 1 步：

```
[vae] dtype=float16 window=8 frames per encoder call, norm=fused, conv=fast
[pack] samples=2 AR=284 DM=7104 backend=npu
[step 1] loss 2.534625
[mem rank 0] peak allocated 36.81 GiB
```

对照基线（diffusers 原版编码器、12 帧）：`loss 2.534409`、峰值 36.81 GiB。loss 差 2.2e-4（0.009%），
在 bf16 latent 的舍入量级内；峰值不变是因为这一步的峰值由 bf16 主干与优化器决定，VAE 的 2 GiB 级差别被淹掉。

### 复现

```bash
python scripts/profile_vae_conv.py --checkpoint /mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-Policy-DROID \
    --data /mnt/disk0/jiangzs/datasets/Cosmos3-DROID/success --batch 2 --iters 5
python scripts/profile_vae_recipe.py --checkpoint /mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-Policy-DROID \
    --data /mnt/disk0/jiangzs/datasets/Cosmos3-DROID/success --batch 2 --iters 10 --chunks 4,8,12
```
