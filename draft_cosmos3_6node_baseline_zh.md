# 六机 48 卡上的 Cosmos 3 SFT 基线：能测到的、被否掉的、还剩什么

本篇记录 2026-11-12 在 node1/5/6/7/8/9 上做的一轮系统测量：环境统一、基线标定、算子级剖面、
以及一组单变量 A/B。结论先行：**PyTorch 层能拿的收益基本拿完了（−3% 步时），剩下的只有加卡和重写算子**。

## 一、环境

| 项 | 值 |
| --- | --- |
| 机器 | node1(46)/node5(6)/node6(98)/node7(138)/node8(114)/node9(88)，各 8×910B2 |
| 镜像 | `ascendai/torch-npu:2.13.0rc1-cann9.1.0-910b-openeuler24.03-py3.12`（ID `23112526d56c`） |
| venv | 本地盘 `/mnt/disk0/jiangzs/nzf/cai-npu`，`torch 2.13.0+cpu / torch_npu 2.13.0rc1 / torchvision 0.28.0+cpu / diffusers 0.40.0 / transformers 5.17.0` |
| 仓库 | 本地盘 `/mnt/disk0/jiangzs/nzf/ColossalAI`，`PYTHONPATH` 覆盖 NFS 上的 editable 安装 |
| TileLang | `0.1.4+ubuntu.24.3.cann910`（wheel 从 release 资产下载后手工安装，`--no-deps`） |
| 容器 | `--privileged --network host --shm-size 512g --user 0:0` |

六个节点的 `env_fingerprint.sh` 输出逐字段一致，`train_sft.py` 的 md5 相同。

## 二、三个会伪装成「训练卡死」的运行时坑

1. **共享 NFS 上的 JIT 编译锁。** ColossalAI 的 JIT 扩展缓存在 `~/.cache/colossalai/torch_extensions/<hash>/`，
   `torch.utils.cpp_extension.load` 用其中的 `lock` 文件（`FileBaton`）做跨进程互斥。一次在编译中途的强杀会留下这个文件，
   之后**所有节点、所有 rank** 的 `HybridAdam` 初始化都会无限等待，日志停在
   `[extension] Loading the JIT-built cpu_adam_arm kernel` 且无任何报错。
   修复：本分支给缓存根加了 `COLOSSALAI_EXTENSION_CACHE`，六台指到本地盘；重建容器后仍是 `load`（0.028 s），不再串行编译（55 s vs 91 s）。
2. **ssh 会话会带走任务。** `nohup ... &` 只挡宿主 shell 的 SIGHUP，外层 `docker exec` 客户端仍挂在通道上，
   断连后容器内进程一起消失。必须 `setsid nohup ... < /dev/null`。
3. **`/dev/shm` 与 worker 数。** 单卡一批 32 样本 ≈ 1.28 GB，在途量 ≈ `rank × worker × prefetch(2) × 1.28 GB`；
   4 worker 实测水位 190 G，256 G 会在数步内打满并报 `unable to allocate shared memory(shm)`。512 G 下 12 步无压力。

排查手段：`py-spy dump --pid <rank>` 一次即可区分「卡在锁」「卡在集合通信」「卡在取数」。

## 三、基线（32 样本/卡、full 重算、bf16 VAE、无增强）

| 规模 | 配置 | 稳态步时 | 全局样本/步 | 吞吐 | 8192 万样本 |
| --- | --- | --- | --- | --- | --- |
| 2×8 | 2 worker, enc-batch 4 | 33.3 s | 512 | 15.4 样本/s | 61.6 天 |
| 4×8 | 2 worker, enc-batch 4 | 35.6 s | 1024 | 28.8 | 33.0 天 |
| 4×8 | 4 worker, enc-batch 8, 512 G | **34.7 s** | 1024 | 29.5 | 32.1 天 |
| 4×8 | 4 worker, enc-batch 8 + fast-encoder | **33.7 s** | 1024 | 30.4 | 31.2 天 |
| 5×8 | 4 worker, enc-batch 4 | 35.1 s | 1280 | 36.5 | 26.0 天 |
| **6×8（48 卡）** | 4 worker, enc-batch 8 + fast-encoder | **35.1 s** | 1536 | **43.8** | **21.7 天** |

- 步时对规模几乎不敏感（33–35 s），**工期只由卡数决定**；扩展效率 32→48 卡为 96%。
- 步内构成：**VAE 15.0–16.0 s、fwd 2.8 s、bwd 12.5–16.1 s、optim 0.3 s**。
- 4 机 4 worker 版波动 <1.5 s，作为后续对比的标准基线。
- 首步冷启动 136–197 s，其中 `wait` 占 104–167 s，是纯数据侧（mp4 解码 + 磁盘），与模型/VAE 无关。

> **关于 node1 的一次误判。** 本轮早期曾据 `npu-smi` 的 AICore% 判断 node1 被其他任务占用（读到 93–99%），
> 因而只用了 5 机。后来发现这个读数在这套驱动上不可信：4 台机器只剩僵尸进程时它仍显示 89–95%。
> 六机 48 卡的实际启动失败原因是 JIT 编译锁（§2.1），清掉后 node1 可以正常加入。
> 结论：判断节点是否被占用不能只看 AICore%，要看有无持有设备的真实进程（`npu-smi info -t proc-mem -i N`），
> 或用 `py-spy dump` 看训练进程卡在哪。

## 四、单变量 A/B：什么有效、什么无效

| 改动 | VAE | 步时 | 判定 |
| --- | --- | --- | --- |
| `--fast-vae-encoder`（H/W pad 交给 conv、去掉残差块死 clone） | 16.06 → **14.96 s（−6.9%）** | 34.67 → 33.70（−2.8%） | **有效，精确等价，进基线** |
| `--vae-encode-batch 8`（每步 8 次调用 → 4 次） | 16.49 → 15.96（−3.2%） | — | 有效，小 |
| `--vae-encode-batch 16` | 未测（显存风险） | — | 未验证 |
| `--fuse-vae-norm` | 更慢 | +5% | 无效 |
| `--vae-chunk-frames 12` | 更慢 | — | 无效 |
| `--vae-dtype float16`（bf16 下） | 更慢 | — | 无效 |
| `--checkpoint-mode selective`（SAC） | — | 反向不变慢、显存 +9.8 GiB，32 样本/卡 OOM | 无效 |
| `channels_last_3d` 布局 | 平台不支持 | — | 不可行 |

**判据**：只有**减少实际搬运字节**的改动有效（pad kernel、死 clone、调用次数）。只减少指令数、或改数值格式的改动，
在 batch 4、544×736×33 这个形状上都会被访存成本吃掉。

## 五、算子级剖面（bf16、batch 4、1 次 encode）

一次 `vae.encode` 发出 **5146 个 kernel**，设备侧 3.42 s：

| 类别 | 调用数 | 累计 ms | 每次 ms |
| --- | --- | --- | --- |
| Conv3D(v2) | 233 | 898.0 | 3.85 |
| TransData | 801 | 161.7 | 0.20 |
| Div/RealDiv | 198 | 140.0 | 0.71 |
| PadV3 | 227 | 137.0 | 0.60 |
| Cast | 441 | 122.3 | 0.28 |
| Concat | 272 | 74.1 | 0.27 |
| LpNorm/Mul/Silu/Adds/BroadcastTo/ClampMin 等 | ~1300 | ~300 | ~0.23 |

- **Conv3D 只占 26%**，其余是逐元素与布局类小算子；宿主侧光 launch 就 1.13 s。
- 但把调用次数减半只省 3.2%，说明**耗时随样本线性**，per-call 固定开销很小——「融合小算子省 launch」的收益上限被高估了。
- `channels_last_3d` 验证：`conv3d` 直接报 `NPU contiguous operator only supported contiguous memory format`，
  整个 encode 报 `Only contiguous_format or preserve_format is supported`。801 次 TransData 是 CANN conv3d 内部的
  ND↔NC1HWC0 转换，**PyTorch 层无接口消除**，只能整条链自己写。

## 六、被否掉的路线（连带理由）

- **预计算 latent（已实测，见 §8）**：`packing.py` 原先给出两条反对理由——增强会被冻死（同窗口 6 次增强之间 relL2 0.614，
  比相邻两个不同窗口的 0.522 还远），以及全量无损缓存要 100 TiB。实测后第一条仍然成立（缓存路线必须关增强）；
  第二条按真实 latent 形状 `(1,48,9,33,40)` 修正为 **21.9 TB**（`keep_ranges` 下 16.2 TB），比原估小 4 倍，
  六台本地盘恰好放得下。修订后净收益约 **7 天**（旧估 3 天是解码串行导致的虚高），见 §8。
- **按阶段拆机**：VAE 与模型都吃同一批 NPU，同构集群上拆机不改变吞吐上界（N 卡 = N / 1.065 样本/s）。
  外部证据一致：NVIDIA EPD 明确写「disaggregated encoder 只在有更低成本 GPU 层时才有意义」。
- **SAC / VAE 四开关组合**：整体较基线慢 4–5%、显存 +9.8 GiB。

## 七、还剩什么

| 手段 | 预期 | 前提 |
| --- | --- | --- |
| **48 卡（已实测）** | 31.2 → **20.8–21.7 天（−30%）** | 已完成，见 §3 |
| 加大梯度桶（`--zero-bucket-size-m 48`） | **无变化**（33.6 vs 33.6 s） | 已实测，见下 |
| 反向 reduce-scatter 与计算重叠 | 上限 −6~8%（48 卡 bwd 12.8–13.0 s vs 单机 8.1 s，通信约 4.8 s） | 需要动插件的梯度桶/hook |
| TileLang 重写 conv3d 链 + NZ 布局 | −3~5%（吃掉 TransData/Pad/Cast 约 12% 设备时间） | 能编译但**比 CANN 慢 2.8×**，见下 |

**conv3d 的 TileLang spike（两种结构都比 CANN 慢 3×，且都没算对）。** 前三次失败的真因是模块顶部的
`from __future__ import annotations`：它把 `T.Tensor(...)` 注解变成字符串，`@T.prim_func` 在 `def main(` 处报
`expected Object but got str`。去掉之后 kernel 能编译、能跑，但结果不支持继续移植。

两种结构都试了，**输入/权重/输出统一用 bfloat16**（VAE 真实推理就是 bf16，CANN 的 bf16 卷积路径和 fp16 不同）：

| 结构 | 每次耗时 | 等效算力 | 数值（vs fp32 参考） |
| --- | --- | --- | --- |
| 27 个 tap 各发一次 `T.gemm_v0`（K=64） | 0.48 ms | 0.66 TFLOPS | `max_abs=254.8`，错 |
| 27 个 tap 拼进同一 L1 面板、一次 K=27·32 GEMM | 0.44 ms | 0.72 TFLOPS | `relL2=9.95e-1`，仍错 |
| torch `conv3d`（CANN，bf16） | **0.15 ms** | **2.19 TFLOPS** | `relL2=1.65e-3`（bf16 自身舍入） |

两个问题：

1. **性能**：即使把 K 从 64 折到 864，也只从 0.48 降到 0.44 ms，仍是 CANN 的 2.9× 慢。说明瓶颈不在 K 的
   大小，而在搬运与流水——CANN 在 MTE1 里做 im2col、L1 双缓冲、按核分块，直白的 `T.copy` 序列达不到。
2. **正确性**：向 L1 缓冲区的**子区间**拷贝（`a_l1[0:ow, tap*k : (tap+1)*k]`）在这个版本里没有按预期落到
   偏移上，27 个 tap 互相覆盖，`relL2≈1` 等价于结果与参考基本无关。

结论：**在拿到官方卷积模板之前，这条线不值得继续**。补充两个后续验证：

- 仓库里最接近的模板是 `examples/causal_conv1d`（带 `design.md`），但它是**纯 Vector 的 1D 因果卷积、没有 Cube/GEMM**，
  教不了 im2col + GEMM 的结构；它用的 `pass_configs`（`TL_ASCEND_AUTO_CV_COMBINE`、`TL_ASCEND_MEMORY_PLANNING`）
  套到我们的 kernel 上，耗时 0.44 → 0.48 ms，**没有改善**；
- `tilelang-ascend` 的 `examples/convolution/` 在 `ascendc_pto` 分支（commit `4805777`）上**并不存在**，
  README 里的链接指向一个未发布的目录；所以"照抄官方卷积骨架"这条降本路径也不成立。

即：正确性卡在「向 L1 子区间拷贝」的 DSL 语义上，性能卡在缺少 MTE1 im2col 与双缓冲流水上，而端到端上限只有 −3~5%。
TileLang 当前的价值是「环境可用、能写 elementwise/GEMM 类算子」（六台 vec_add + 1024³ GEMM 已验证），
不是「替代 conv3d」。

**梯度桶大小的负结果。** 曾假设 48 卡的 backward 变慢（12.8–16.1 s vs 单机 8.1 s）是因为桶太小、同步点太多，
把 `zero_bucket_size_in_m` 从 12 加到 48 后实测 33.62 / 33.55 s，与桶 12 的 33.33 / 36.84 s 同量级——
两次的差异完全来自桶 12 那一次 16.14 s 的反向抖动。**同步点数量不是瓶颈**，那约 4.8 s 是集合通信本身的成本，
只能靠真正的重叠（改动 plugin 的梯度桶调度）或换更快的互联来消，加桶参数做不到。

优先级：**通信重叠 > 内核重写**（加卡这一项已经落地）。48 卡下 bwd 的绝对值比 32 卡高 3–4 s，
说明复制组从 4 节点扩到 6 节点后跨节点归约更贵，而这一点不能靠调桶解决。

## 八、latent 预提取的实测（2026-11-13，修订版）

按「离线把 VAE 编码结果缓存下来、训练只读 latent」的路线做了提取与验证。**旧版本这里的数字（1.85 s/样本、
4.3 样本/s/节点、六台 9.1 天）作废**，原因有两个，都必须记下来：

1. **那个 4.3 样本/s 是「零解码并行」的结果**。`precompute_latents.py` 当时 `--num-workers 0`，解码在编码进程里
   串行执行；每样本 1.9 s 里只有 0.47 s 是 VAE，其余是解码——**3 个相机各 33 帧，共 99 帧**，不是 33 帧。
   训练循环里解码是藏在 dataloader worker 后面的，
   所以提取端不并行解码就等于把两者相加。
2. **测量期间 node5 上有一个上次会话遗留的训练进程**（`scale-s1x8-32`，`--max-iter 4`）。它的 rank 处于半死状态，
   HCCL 一直失败，于是一直重启 4 个/卡 的解码 worker，把 CPU 与磁盘占满。这既降低了提取吞吐，也是
   「32 个解码器反而更慢」这个结论的来源——**那条结论是错的**。

清理后重测。**共同条件**：node5 单机 8×910B2 全用，每卡 1 个 rank 进程（`ASCEND_RT_VISIBLE_DEVICES=$r` +
`--device 0`，8 卡 8 进程，无集合通信）；`--data .../success --max-episodes 8`，`chunk_length=32`、`fps=15`、
`resolution=480`，`--limit 512` 即 **512 窗口 / 每 rank 64 个**；`--shard-size 256`（每 rank 结束时 1 个 shard）；
bf16 latent；VAE 取 `Cosmos3-Edge-Policy-DROID/vae`。

| 配置 | 分片方式 | 每卡 worker | 节点解码器 | encode-batch | 每 rank 批数 |
| --- | --- | --- | --- | --- | --- |
| `str_nw0` | stride | 0（解码在编码进程内串行） | 8 | 8 | 8 |
| `blk_nw0` | block | 0 | 8 | 8 | 8 |
| `w2_eb4` | block | 2 | 16 | 4 | 16 |
| `w4_eb4` | block | 4 | 32 | 4 | 16 |
| `w8_eb4` | block | 8 | 64 | 4 | 16 |

用日志里 `batches=/samples=/at=` 的计数器取**首尾斜率**作为稳态值（排除 kernel 首次编译的一次性预热，
实测预热 17–130 s 不等，短跑里它会主导 `per_sample`）。

| 配置 | 首 batch 到达 | 稳态 s/样本 | 节点吞吐 | 相对 0 worker | 六台合计 | 20.11 M 全量 | keep_ranges(74%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `str_nw0` | 21–26 s | 2.62 | 3.1 /s | 0.72× | 18.3 /s | 12.7 天 | 9.4 天 |
| `blk_nw0` | 14–17 s | 1.89 | 4.2 /s | 1.00× | 25.4 /s | 9.2 天 | 6.8 天 |
| `w2_eb4` | 37–41 s | 1.06 | 7.5 /s | 1.78× | 45.2 /s | 5.2 天 | 3.8 天 |
| `w4_eb4` | 64–69 s | 0.66 | 12.1 /s | 2.86× | 72.7 /s | 3.2 天 | 2.4 天 |
| **`w8_eb4`** | **130 s** | **0.58** | **13.7 /s** | **3.25×** | **82.5 /s** | **2.8 天** | **2.1 天** |

两处口径必须说明：`str_nw0` 只有 6 个 rank 的结果（8 个里 2 个在 eb8 下 OOM，报在 VAE `norm2` 的 fp32
归一化），不影响 block/stride 的对比方向；`encode-batch` 在两组间不一致（0 worker 组 8、并行组 4，因为发现
eb8 显存贴边后把并行组降到了 4），eb4 vs eb8 本身只差约 3%，但严格单变量应该统一，这一项**没有补测**。

几点结论：

- **分片方式（块 vs 步长）只值 28%**，不是我先前猜的 3 倍。窗口列表本身按 episode、episode 内 start 递增排列，
  连续块确实免掉了每样本的跨文件 seek，但「解码 3 视角 × 33 帧」的成本本身不变。**默认用 block，收益真实但不大。**
- **解码并行才是主变量**：0 → 8 worker 是从 1.89 到 0.58 s/样本（3.25×），且 4 → 8 worker 已接近饱和。
- **VAE 下限**：0.47 s/样本 × 20.11 M / 48 卡 = **2.28 天**。8 worker 的 2.82 天是该下限的 1.24 倍，
  差的那 0.11 s/样本是主循环里的搬运/裁剪/存盘，不是解码。
- 与「在线 VAE 占 47%」对账：在线 20.8 天里 VAE 占 9.8 天，那是 **81.92 M** 样本；离线只需 **20.11 M**（1/4.07），
  即 9.8 / 4.07 = **2.4 天**，与实测 2.8 天、下限 2.28 天三者自洽。**之前的 9.1 天纯属解码串行造成的虚高。**
- **「六台训练 + 一台提取」仍然不成立**：一台提取机 13.7 样本/s，六台训练（48 卡、步时约 18.6 s、1536 样本/步）
  需要 82.6 样本/s，覆盖率 **17%**。要并发就得 6 台提取对 6 台训练（1:1），而总工作量守恒，没有收益。
  可行的排期仍是**先提后训**。

**样本体积与全量规模**：latent 形状 `(1, 48, 9, 33, 40)` bf16 = **1.09 MiB/样本**。
success 16.85 M + failure 3.26 M = **20.11 M 窗口** → **21.9 TB**；套用官方 `keep_ranges`（约 74%）为 **16.2 TB**，
六台本地盘（合计约 18.2 TB）只有后者放得下。

**净账（修订）**：提取 2.4–2.8 天 + 训练（48 卡、步时 33.6 → 约 18.6 s、11.0 天）= **13.4–13.8 天**，
对比全在线 VAE 的 20.8 天，**净赚约 7 天**（旧版本写 3 天，低估了）。

**两个必须记住的操作坑**：

- `--encode-batch 8` + 8 个 rank 同时跑时显存贴边：一次 512 样本的测试里 8 个 rank 有 2 个 OOM
  （`NPU out of memory. Tried to allocate 1.91 GiB`，发生在 VAE 的 `norm2` fp32 归一化）。
  提取侧用 **batch 4**，吞吐损失很小（4→8 worker 是并行度，不是 batch）。
- **不给 `--max-episodes` 或窗口区间就不要跑全量提取**：数据集构造会为**全部 20.11 M 窗口**各建一个 `Window` 对象，
  8 个进程各来一遍，几十 GB/进程，还没开始编码就被 OOM 杀掉（实测 21 s 内静默消失、日志只有 import 警告）。
  全量提取需要先用 `--max-episodes` 分片，或给 `precompute_latents.py` 加窗口区间参数。

**数值门禁**（8 episode / 1,024 样本子集，缓存 vs 在线 encode）：形状与几何完全一致（canvas 544×736、裁剪 33×40），
`max_abs` 0.031–0.063、`relL2` **4.1e-3**，三个样本高度一致。这个量级正是 **bf16 存盘多一次舍入**的误差
（bf16 尾数 8 位），**换成 fp16 存盘体积不变、relL2 应降到 1e-3 量级**。

只在实验子集上做链路验证时（64 episodes / 16,799 窗口），按 13.7 样本/s/节点算约 **20 分钟**、占 18 GB。

## 九、图像增强（官方配方默认项）已跑通（2026-11-13）

此前所有基线都带 `--image-augmentation none`，原因是 **torchvision 装不上能用的一份**：某次安装的 wheel
能 import 但没注册算子，第一处 `torchvision::nms` 就崩，看起来像数据 bug 而不是依赖问题。当时的处置是在所有
对比里显式传 `none` 绕过它——那批数字是**偏离配方**的（`train_sft.py` 的默认一直是 `torchvision`）。

现在（node1/910B2，`torch 2.13.0+cpu` / `torch_npu 2.13.0rc1` / `torchvision 0.28.0+cpu`）增强可用，
`import torchvision.transforms.v2` 无错。实现本来就是照抄配方的（`T.RandomCrop(0.95)` → `T.Resize(antialias)`
→ `T.ColorJitter(0.3/0.4/0.5/0.08)`，**在 `torch.cat([wrist,left,right])` 上一次调用**，三视角共享同一组
裁剪偏移与颜色因子），这次只做了验证，没改逻辑。

**验证不是「跑完没报错」，而是三条**（`scripts/probe_augmentation.py`，同一窗口跑两遍增强）：

| 判据 | 实测 | 含义 |
| --- | --- | --- |
| 增强 vs 不增强 `relL2` | 0.52 / 0.56 | 像素真的被改了 |
| 两次增强之间 `relL2` | 0.58 / 0.67 | 是随机抽样，不是固定裁剪 |
| canvas / content 几何 | 完全一致（544×736 / 540×640） | 只动像素，不动布局 |

增强后取值范围仍在 `[-1, 1]`，说明 ColorJitter 在 fp32 上按 [0,1] 裁剪，不会把数值推出量化区间。

**代价**：`_augment` 在 99×3×360×640 上 **226–487 ms**（浮动来自 ColorJitter 抽到的因子，hue 走 RGB→HSV→RGB
最贵；参考脚本 `/mnt/disk0/jiangzs/share/bench_augmentation.py` 把这一点写清楚了：**必须固定 seed 才能比较**，
否则两次测量不可比）。同一脚本测得 CPU fp32 整条 Compose 221 ms，其中 ColorJitter 208 ms；**NPU 上 664 ms，
慢 3 倍**，所以增强只能留在 CPU worker 里做。

**端到端步时**（单机 8 卡、4 worker、`--vae-encode-batch 8`、`--fast-vae-encoder`）：

| 每卡样本 | 增强 | step 1（冷启动） | step 2–4 稳态 | 稳态 `wait` |
| --- | --- | --- | --- | --- |
| 8 | torchvision | 143.3 s（wait 134.0） | 8.41 / 7.93 / 8.08 → **8.14 s** | 1.15 / 0.50 / 0.81 |
| 8 | none | 89.2 s（wait 80.0） | 8.80 / 8.08 / 7.83 → **8.24 s** | 0.52 / 0.68 / 0.70 |
| 32 | torchvision | 376.6 s（wait 346.7） | 31.91 / 32.33 / 31.07 → **31.77 s** | 2.00 / 3.59 / 2.20 |
| 32 | none | 173.3 s（wait 143.4） | 32.18 / 31.08 / 31.35 → **31.54 s** | 3.27 / 2.92 / 2.59 |

结论：

- **稳态步时没有可测差异**（32 样本/卡：31.77 vs 31.54 s，0.7%，在噪声内）。步时由 VAE（15.0 s）+ 反向（11 s）
  决定，增强被 4 个 worker 的预取吸收掉了。
- **冷启动翻倍**（32 样本/卡：wait 346.7 vs 143.4 s）。一万步里这一项只出现一次，可以忽略，但做 4 步短测时
  它会污染第一行数据，比较步时必须丢掉 step 1。
- 因此 §3 里那些 `--image-augmentation none` 的基线**仍然成立**，不需要因为打开增强而重测；
  正式复现应当用默认的 `--image-augmentation torchvision`。
- 增强和 latent 预提取**互斥**：预提取把增强结果冻进缓存（同窗口不同增强之间 relL2 0.6，比两个不同窗口
  之间的 0.52 还远，等于把一条样本变成多条矛盾样本），所以 §8 那条路线必须关增强。

## 十、复现命令

```bash
# 六机统一环境（node6 从 node5 引导）
bootstrap_node.sh http://192.168.0.6:<port> /mnt/disk0/jiangzs/nzf

# 六机 48 卡标准基线（master 那台必须是 node_rank 0；把 NODES 改成 4/5 即其它规模）
# §3 的基线当年显式加了 --image-augmentation none；§9 之后正式复现用默认的 torchvision。
for rank in 0 1 2 3 4 5; do
  setsid nohup dev-container-node.sh train_scale_inner.sh <tag> $rank 6 192.168.0.6 <port> \
    --samples-per-rank 32 --grad-accum 1 --hsdp-shard-size 8 \
    --max-iter 4 --metrics-every 1 --log-every 1 --num-workers 4 \
    --image-augmentation torchvision --vae-encode-batch 8 --fast-vae-encoder
done
```

`--fast-vae-encoder --vae-encode-batch 8` 应作为后续所有对比的基线配置。

```bash
# 图像增强的验证与代价（§9 的数据来源）：先证明它真的改了像素，再测步时
dev-container-node.sh env PYTHONPATH=/mnt/disk0/jiangzs/nzf/ColossalAI \
  python /home/niuzifan/scripts/probe_augmentation.py 2 1     # relL2 / 几何 / 单次耗时

setsid nohup dev-container-node.sh ab_augmentation_train.sh 32 29521 8 \
  > /home/niuzifan/logs/ab_aug32.log 2>&1 < /dev/null &       # torchvision vs none，各 4 步

grep -E "^\S+ \[step" /home/niuzifan/logs/ab_aug32.log         # 比较时必须丢掉 step 1（冷启动）
```

```bash
# latent 预提取的解码并行度扫描（§8 的数据来源）：8 卡各一进程，8 episode / 512 窗口
setsid nohup dev-container-node.sh ab_latent_workers.sh 2,4,8 4 512 8 \
  > /home/niuzifan/logs/ab_workers.log 2>&1 < /dev/null &

# 分片方式 A/B（block vs stride，§8 的第一行与第二行）
setsid nohup dev-container-node.sh ab_latent_shard.sh 512 8 \
  > /home/niuzifan/logs/ab_shard.log 2>&1 < /dev/null &

# 从日志的批次计数器算稳态吞吐（排除 kernel 预热），并外推到 20.11 M
ssh node5 'for t in w4_eb4 w8_eb4; do for f in /mnt/disk0/jiangzs/nzf/latents/$t/rank-*.log; do \
  echo "$t|$(basename $f .log)|$(grep -h batches= $f | tr "\n" "|")"; done; done' > batch_lines.txt
python scripts/summarize_extraction.py batch_lines.txt
```

**跑提取前先确认节点是干净的**，否则会得到 §8 里那种被污染的数字：

```bash
docker exec nzf-cai-dev pgrep -af "train_sft.py|precompute_latents.py"   # 应为空
npu-smi info -t proc-mem -i 0                                          # 应为 "No process in device."
```

参数里带空格时**不要**指望 `dev-container-node.sh` 保留引号：它把参数拼成一个 `bash -c` 字符串，
`"2 4 8"` 会散成三个位置参数并把后面所有参数错位（表现为 `--max-episodes 4 --limit 8`）。用逗号分隔。
