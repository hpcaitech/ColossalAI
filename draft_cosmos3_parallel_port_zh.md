# Cosmos 3 并行训练：把官方训练基础架构的并行能力落进 ColossalAI

对象：`examples/worldmodel/cosmos3` 的 SFT（Ascend 910B2 ×8 单机，×16 双机）。
对照：官方技术报告 Sec. 5（Cosmos 3: Omnimodal World Models for Physical AI）与官方 `cosmos-framework` recipe。
配套：`draft_cosmos3_parallel_zh.md`（八卡实测矩阵与调优结论）。

提交：`f293e8e1`（HSDP + Ulysses CP）→ `b6ef5782`（SAC）→ `1c469ae3`（policy 测试）→ `c75202b3`（VAE 编译）
→ `4b3e521e`（CP 集合通信可微）→ `dadf3b64`（峰值显存上报）。

## 一、官方 Sec. 5.2 的优化点，逐条对照

| # | 官方做法 | 在我们这套里的落点 | 状态 |
| --- | --- | --- | --- |
| 5.2.1 | JointDataLoader：token 预算打包、rank 同步流选择（+54%）、look-ahead 打包（+8%）、冷启动预热、可观测性 | 示例的 `droid_dataset`/`pack_batch` 是固定 33 帧窗口，单流；而且 `collate` 要求同一批样本 canvas 一致（不同画幅堆不起来），token 预算打包要先改这条约束 | 不适用（需先重构数据侧） |
| 5.2.2 | 双向扁平注意力：两路 varlen SDPA，[R₀,G₀,R₁,G₁,…] 交织，+22% | 已在 `ColossalTwoWayAttnProcessor` / `two_way_flat_attention`（naive 参考 + NPU 融合两套后端） | 已有 |
| 5.2.3 | HSDP（组内切分、组间复制）+ Ulysses CP，两维正交 | 本次：`HybridParallelPlugin(hsdp_shard_size=…)` 与 `sp_size>1` 的 CP | **本次实现** |
| 5.2.4 | 选择性激活重算：按 FLOPs/占用比排序，优先保留 attention 输出，+13% | policy 参数 `Cosmos3OmniPolicy(selective_checkpointing=True)` / 示例 `--checkpoint-mode selective`：整层重算改成只重算两块 MLP，attention 的输出保留不重算 | **本次实现**，CPU 上梯度等价已验证 |
| 5.2.5 | `torch.compile(fullgraph=True, dynamic=True)`，+41% | 示例加了 `--compile-blocks`：`booster.boost` 之后逐层 `torch.compile`（在 policy 装好 processor 之后，编译第一次调用时看到的是最终图） | 已接线，默认关；NPU 上未验证 |
| 5.2.6 | VAE 分块编码（256p 68 帧 / 480p 24 帧 / 720p 12 帧一次）+ AOTInductor 把 45 张静态图分片编译 | 分块不用做：diffusers 的 `AutoencoderKLWan._encode` 本来就在 4 帧一块地循环、用 `feat_cache` 跨块传状态（源码 1142–1156 行）。编译这一半接成了 `--compile-vae`：静态 shape、按 canvas 一张图 | 分块已内置；编译已接线（默认关，实测无加速） |
| 5.2.7 | 异步 ckpt：独立 Gloo 进程组、save plan 复用、`dedup_to_lowest_rank` | 示例加了 `--async-save`：`save_model(use_async=True)`，下一步 `optimizer.step()` 前 `checkpoint_io.synchronize()`（只等固定内存拷贝，不等磁盘），收尾 `_sync_io()` 后再修文件权限 | 已接线，默认关 |

一句话：**并行维度按官方补齐了（HSDP + CP），激活重算做成了官方的选择性版本；torch.compile（逐层与 VAE）、异步 ckpt
都接成了开关。剩下两项：VAE 分块编码 diffusers 早就内置了，JointDataLoader 要先放开我们"一批样本必须同 canvas"
的约束。** 理由写在第六节。

旋钮一览（都作用在同一个 `HybridParallelPlugin` 上，不冲突）：

| 目标 | 命令 |
| --- | --- |
| 单机 8 卡，纯 ZeRO-2（默认） | `torchrun --nproc_per_node=8 train_sft.py …` |
| 单机 8 卡，长序列：2 路 Ulysses CP | `… --sp-size 2` |
| 单机 8 卡，CP 与 HSDP 同时开 | `… --sp-size 2 --hsdp-shard-size 4`（dp×sp=8，切成 2 组各 4 张，组间复制） |
| 双机 16 卡，节点内切分、跨节点复制 | `… --nnodes=2 --nproc_per_node=8 --hsdp-shard-size 8`（切 8 张=一个节点，2 个副本） |
| 双机 16 卡，CP 与 HSDP 同时开 | `… --nnodes=2 --nproc_per_node=8 --sp-size 2 --hsdp-shard-size 4` |

注意：单机上开 HSDP 没有收益（通信本来就都在节点内），它的作用是 16 卡以上把参数级集合通信留在节点里；
单机那两行只是用来验证代码路径。

## 二、HSDP：组内切分、组间复制

官方在 256 卡上用 `data_parallel_shard_degree=8 × replicate_degree=32`。ColossalAI 原来只有一个 dp 组，
跨 16 卡时 ZeRO 的 reduce-scatter 和参数 all-gather 会整段走 RoCE。现在：

```python
HybridParallelPlugin(..., zero_stage=2, hsdp_shard_size=8)
```

- dp 组（`dp × sp`）按网格顺序切成连续的 `hsdp_shard_size` 大小的 **shard 组**；ZeRO 的 reduce-scatter、
  参数 all-gather、优化器状态切分都只在 shard 组内做。
- 每个 shard 组里同一个偏移上的 rank 组成 **replica 组**；反向里在 reduce-scatter 之后对**已切分的梯度**
  再做一次 `all_reduce`（大小只有参数的 1/shard_size），这就是 HSDP 的第二级。
- 两级平均复合起来正好等于原来在整个 dp 组上的一次平均，所以数学上等价，只是把参数大小的通信留在了节点内。
- checkpoint：`HybridParallelCheckpointIO` 按 shard 组汇聚（组内已覆盖完整状态），只有 replica 0 落盘，
  不会出现多个副本抢写同一路径。
- 与官方差异：官方的 HSDP 连**参数**也切（FSDP 语义），ColossalAI 的 ZeRO 只到 stage 2，参数在组内是复制的。
  所以显存上我们等价于"ZeRO-2 切 8 份"，通信上等价于官方 HSDP。要再往下走需要 ZeRO-3/Gemini 与 HSDP 组合，
  那是另一个工程。

启动：

```bash
torchrun --nnodes=2 --nproc_per_node=8 ... train_sft.py --hsdp-shard-size 8 ...
```

`hsdp_shard_size` 必须整除 `dp × sp` 组的大小（开了 CP 时 dp 指的是这个混合组）；不传就是原来的纯 ZeRO。
`zero_stage=0` 时该参数无效（DDP 本来就在整个 dp 组上）。

## 三、Ulysses 上下文并行（CP）

官方 CP 用 Ulysses：序列维切给 CP 组，每层 attention 前一次 all-to-all 把"序列"换成"头"，算完再一次换回来。
两个 all-to-all 之间，每张卡持有**完整序列的一个头子集**，所以逐样本的因果/双向掩码不需要改。

### 为什么能直接接上

diffusers 的 `Cosmos3OmniTransformer.forward` 有两个显式的 CP 缝：

- `_cp_shard_fn(und_seq, gen_seq, rotary_emb)`：embedding 之后、decoder 之前，按需切分；
- `_cp_gather_fn(und_out, gen_out)`：decoder 之后、逐样本解码之前，按需汇合。

我们的 policy 在 `postprocess` 里把这两个缝接到 `Cosmos3ContextParallel`，并把每层 attention 的 processor
换成 `ColossalTwoWayCPAttnProcessor`。模型本身不需要改。

### 数据流

1. 每张卡都在**整包**上算 embedding 和 mRoPE（token 级、便宜），然后 CP 缝把 AR/DM 两路各自 pad 到 cp 的倍数、
   按连续块切分，每卡拿自己那一段。
2. decoder 里 norm / MLP / 残差都是 token 独立的，在本地分片上算。
3. attention：本地 q/k/v 先做投影、qk-norm、RoPE（都是 token 级，分片上结果与全量一致），然后
   **一次 all-to-all 把序列汇齐、把头散开**；切掉 padding 后，用**全局的每样本长度**调用原来的双向 kernel；
   输出补回 padding，再一次 all-to-all 换回"序列分片、头完整"。
4. CP 缝汇合两路输出并丢掉 padding，之后逐样本解码、loss 与单卡完全一样。

关键点是**精确性**：padding 只在 all-to-all 的缓冲区里存在，进 kernel 之前已经裁掉，所以不会有 pad token
进入 attention 分数；每样本边界用的是打包器装在 processor 上的全局 meta，不是从分片反推的。

### 约束

- Ulysses 按头切分，所以 `num_attention_heads=16`、`num_key_value_heads=8` 都要能被 cp 整除 → cp ∈ {1,2,4,8}。
- 暂不支持 tp 与 cp 组合（两者都切头，组合后 kernel 需要知道复合后的头布局），policy 里显式拒绝。
- 仍不支持 PP（和原来一样，官方也不用 PP）。

### 启动

```bash
torchrun --nproc_per_node=8 ... train_sft.py --sp-size 2 ...
```

`--sp-size 2` 时 dp 组变成 4，采样器只按 dp 切，所以同一个 CP 组的两张卡拿到**同一批样本**（各算一半序列），
这正是 CP 要的语义。ZeRO 仍在 `dp × sp` 上切分。

### 梯度与 loss 为什么不用改

CP 缝在解码前把所有分片汇合回完整序列，所以每张卡上的 loss 与单卡一致。反向时 all-gather 的反传是
reduce-scatter（把 cp 份相同的上游梯度求和），于是每张卡本地分片的梯度被放大 cp 倍；而 ZeRO 的平均是在
`dp × sp` 上做的，分母也多了 cp，两者正好抵消，最终等于"每个 dp 副本的完整梯度求平均"。
所以 `compute_loss` 与 `grad_accum` 的写法都不用动。

前提是这几处集合通信**对 autograd 可见**：`torch.distributed` 的 `all_gather` 和 `all_to_all_single` 都写进新分配的
缓冲区，默认不带梯度——前向正确，反向直接断掉。所以两次 Ulysses 交换和 CP 缝的汇合各包了一层
`torch.autograd.Function`：all-to-all 是"自己是自己的逆"的置换，反向就是同一个集合通信；all-gather 的反向是
reduce-scatter。少了这层包装，`--sp-size 2` 会在第一次反向上报 "element 0 of tensors does not require grad"，或者
更糟——attention 投影拿到零梯度、残差路径照常训练，**静默训坏**。

### 收益预期（要诚实）

CP 摊薄的是**每张卡的激活和 attention 计算**：pack 越长，收益越接近线性；但它摊不薄 VAE（每卡仍各自编码整段
视频），也摊不薄 embedding 和逐样本解码（这两段在 CP 缝之外，仍是全量）。我们现在的负载里 VAE 占单步 70%、
窗口固定 33 帧，所以 8 卡上端到端提升有限；CP 真正的用途是把"能训的序列长度"抬高，以及为多机长上下文做准备。

## 四、选择性激活重算（SAC）

官方 5.2.4 不是"整层重算"，而是按"FLOPs / 占用字节"排序：重算代价高、占内存相对小的部分留着，
占内存大的部分才重算。落到 Cosmos 3 的 decoder layer 上，分界线正好落在 attention 与 MLP 之间：

```python
und_norm = input_layernorm(und_seq); gen_norm = input_layernorm_moe_gen(gen_seq)
und_attn, gen_attn = self_attn(und_norm, gen_norm, rotary_emb)   # 保留：输出就是残差流，下一层本来也要用
r_und = und_seq + und_attn; r_gen = gen_seq + gen_attn
mlp_und = mlp(post_attention_layernorm(r_und))                   # 重算：4x 中间层，整层最大的激活
mlp_gen = mlp_moe_gen(post_attention_layernorm_moe_gen(r_gen))
return r_und + mlp_und, r_gen + mlp_gen
```

- attention 的 saved tensors 留着，反向不重算 attention —— 省下的是一次全序列注意力，也就是整层里最贵的 FLOPs；
- 两块 MLP（`mlp` 与 `mlp_moe_gen`）包进 `torch.utils.checkpoint`（`use_reentrant=False`），反向重算，代价是两次
  up-projection；
- 与 `model.enable_gradient_checkpointing()` **互斥**：那个是整层重算，两者叠加会让 MLP 每步算三遍。示例用
  `--checkpoint-mode {full,selective,none}` 三选一，默认仍是 recipe 的 `full`。

实现放在 policy 的 `postprocess()` 里（`enable_cosmos3_selective_checkpointing`），和 processor 的安装同一个时机，
所以补丁落在 ShardFormer 处理完之后的那批层上，不会被模块替换冲掉。

**实测**（2026-11-06，node1，8×910B2，每卡 2 样本，3 步，`Cosmos3-Edge-Policy-DROID` + 8 个 episode）：

| 模式 | 每卡峰值分配 | 单步耗时 |
| --- | --- | --- |
| `full` | 18.01 GiB | ~4 s |
| `selective` | 23.97 GiB | ~4 s |
| `none` | 33.85 GiB | ~4 s |

`none - full = 15.84 GiB / 2 样本 ≈ 7.9 GiB/样本`，与 3.45 vs 11.3 GiB/样本的差（7.85）吻合；SAC 落在两者之间，
比 `full` 多 5.96 GiB（+33%），只有 `none` 超额的一半左右。

**吞吐没有差别**：三档单步都是 ~4 s。这个规模下单步被 VAE 编码和数据解码占住（冻结的 VAE 在反向之外），
重算省下的那点反向 FLOPs 显不出来。官方的 +13% 是在 VAE 占比远低于我们的负载上测的；要在我们这里看到差别，
得把每卡样本数抬上去，让反向占比变大。所以**选哪一档是显存换样本数的问题**，不是直接换吞吐：
64 GB 卡上 `none` 还能再塞一个样本，`full` 能塞更多。默认仍是 recipe 的 `full`（示例 `--checkpoint-mode full`）。

## 五、验证

CPU/gloo 测试：`tests/test_shardformer/test_model/test_hsdp_and_context_parallel.py`（无需 NPU、无需 diffusers）。

- **HSDP 梯度**：4 rank、两层 MLP、真实 `LowLevelZeroOptimizer`。对比"参数更新 == 四个 rank 梯度的普通平均"
  （即单级 ZeRO 的结果）。覆盖 shard=4（纯 ZeRO-2）、shard=2（2 组切分 × 2 组复制）、shard=1（只复制不切分），
  以及 ZeRO-1 的全量梯度路径。最大参数误差 1.5e-8。
- **Ulysses CP（前向）**：2 rank、假 attention 模块、真实双向 kernel。包长取 AR=9 / DM=7（都是奇数，强制触发
  padding），对比"CP 处理器逐卡分片 → 交换 → kernel → 汇合"与"单卡处理器在整包上的结果"。
  AR/DM 输出误差 0，shard/gather 往返误差 0（逐位一致）。
- **Ulysses CP（反向）**：同一个包跑一次带梯度的前反向，把各 rank 的参数梯度按 CP 组求和再除以组大小（这正是
  ZeRO 在 `dp × sp` 上做的平均），与单卡算出的梯度比对，误差 < 1e-5。这一项覆盖两次 all-to-all 交换和 gather
  三处集合通信的反向；没有它，前向逐位一致的 CP 到了训练时是断的。

SAC 另有一个测试文件：`tests/test_shardformer/test_model/test_cosmos3_selective_checkpointing.py`（同样无需 NPU/diffusers）。

- **SAC 梯度与重算范围**：两层假 decoder layer（两路交叉 attention + 两个 4× 升维 MLP），分别用原样 forward 与
  打补丁后的 forward 跑同一次前反向，比对全部参数梯度（rtol 1e-5 / atol 1e-6）。同时计数：attention 调用 2 次
  （两层各一次，反向没有重算），MLP 调用 8 次（前向 4 + 反向 4），证明确实只重算了 MLP。
- 两个文件一起跑：`4 passed`。

### NPU 端到端（2026-11-06）

node1 的 8 张卡清出来后，用 `Cosmos3-Edge-Policy-DROID` + DROID success 的 8 个 episode 跑通了三条路径。每条都是
8 rank × 每卡 2 样本 × 3 步，走示例真实的 `train_sft.py`，不是单独的 benchmark：

| 配置 | 结果 |
| --- | --- |
| `--sp-size 2`（Ulysses CP） | `done`，loss 2.622345 / 12.552932 / 3.441761 |
| `--hsdp-shard-size 4`（2 组各 4 卡切分） | `done`，loss 2.775528 / 12.689603 / 4.814145 |
| `--checkpoint-mode selective`（SAC） | `done`，loss 2.775528 / 12.663039 / 4.810678 |

`--sp-size 2` 跑通本身就是这轮最重要的验证：两处 `all_to_all_single` 交换和 gather 的 autograd 包装第一次在 HCCL 上
走完整的前反向 + optimizer，说明第三节那三处集合通信在 NPU 上不是断的。

两点说明：

- **CP 的 loss 不能和 sp=1 直接比**。`sp_size=2` 时 dp 组变成 4，同样的 `--samples-per-rank 2` 对应的全局 batch
  从 16 变成 8，采样器给出的样本也不同，所以第一步 loss（2.622 vs 2.776）本来就该不同。验证 CP 的数值等价靠的是
  上面那个 CPU 梯度测试——同一份输入、逐 rank 梯度求和再除以组大小——不是这里的 loss。
- **不同模式从第 2 步起有 ~0.3% 的 loss 漂移**（12.663–12.696），但同一个 `full` 在两个不同 dp 拓扑下
  （hsdp4 与纯 ZeRO）也差 0.0065，`none` 与 `full` 数学上完全等价却也差 0.011。四组的第一步 loss 完全相同，
  说明前向一致，差别来自反向里的浮点非确定性（重算与集合通信的归约顺序），不是 SAC 改变了梯度。严格的
  确定性对比还没做。

**还没做的验证**：两机 16 卡 `--hsdp-shard-size 8`。node5 的 8 张卡是空的，`/home` 与 node1 是同一个 NFS
（剩 359 GB），但数据在 node1 的本地盘 `/mnt/disk0/jiangzs`（数据集 585 GB）——要先挑几个 episode 和 8.6 GB 的
权重拷到共享盘，再在 node5 起容器，才能跑跨节点那一组。

`--compile-vae` 做过一次小规模实测（`examples/worldmodel/cosmos3/scripts/smoke_compile_vae.py`：随机初始化的小 Wan
VAE，输入 2×3×9×16×16，跑在 CPU 上）：能编译，数值一致（最大误差 6.6e-7），但**没有加速**——公平计时下 eager 0.56 s，
静态编译 1.14 s（0.49×），动态 1.24 s（0.45×，首次编译 50 s）。小张量 + 大量小卷积在 CPU inductor 上就是这个结果，
说明不了 480p 真实 VAE 在 NPU 上的情况，所以开关保留、默认关，结论等有卡再下。

`--compile-blocks` 与 `--async-save` 只做了语法检查，一次都没有真跑过（本地没有 torch，节点上的卡被占）。
三个开关都默认关，不影响默认路径；开启前按第六节列的前置条件确认。

## 六、和官方的差距，以及下一步

1. **参数切分（ZeRO-3 / FSDP 语义）**：官方 HSDP 会把参数也切到 shard 组。ColossalAI 的 ZeRO 只到 stage 2，
   所以我们现在是"参数复制 + 梯度/优化器状态切分"。3.4B 参数下每卡多 6.3 GB 静态显存；要压缩这一段，
   需要把 Gemini/ZeRO-3 与 HSDP 组合，工作量远大于本次。
2. **SAC 的调参**：显存三档已经量出来了（18.01 / 23.97 / 33.85 GiB，每卡 2 样本），但单步都是 ~4 s——这个
   batch 下 VAE 占大头，重算的差别被淹掉。要决定 recipe 用哪一档，得在"最大可塞样本数"那个规模上再测一次吞吐。
3. **torch.compile**：已接成 `--compile-blocks`。报告是 +41%，但 NPU 上冷编译 18 分钟、热缓存首调仍 95–140 s，
   单步 13.8 s 的负载承担不起，所以默认关；开启前要先确认两件事：融合 attention kernel 是否以自定义算子注册
   （否则 `fullgraph=True` 会直接报 graph break），以及激活重算用的是非 reentrant 版本。
4. **VAE 已经量清楚了**：单样本编码 1.229 s（fp32、544×736×33），设备占用 99.9%——不是调度问题，
   卷积 55.5%、布局转换 17.4%。改 fp16 后 **0.499 s（2.46×）**，latent 与 fp32 的差异只有配方自带
   bf16 量化的 **0.61 倍**（8 卡 × 8 样本共 64 个真实窗口实测，最差 0.64 倍；bf16 编码则是 4.7–5.0 倍，不能用），
   峰值显存 6.11 → 3.11 GiB。分块编码确实不需要做（`_encode` 本来就按 4 帧分块）；
   逐层/逐 kernel 拆解、channels_last 的平台限制和后续融合清单见 `draft_cosmos3_vae_optimization_zh.md`。
   `--compile-vae` 之前"小规模没有加速"的结论要在 fp16 之后复测一次，那时逐元素占比会从 8% 升到 24%，值得再试。
   顺带一个启动侧的坑：每卡 4 样本起，dataloader 的 `/dev/shm` 就会打满（32 GB），已把共享策略切到 `file_system`；
   实测每卡样本上限是 **8**（模型阶段 38 GiB），12 已经显存 OOM。
5. **JointDataLoader**：不是实现难度问题，是我们这条数据链的前提不同——`collate` 直接 `torch.stack` 视频，
   所以一批样本必须落在同一个 canvas 上；官方的 token 预算打包恰恰要打散分辨率。要做就得先给 `collate` 加补边/分组，
   再写一个按 token 预算切批的 sampler，属于数据侧的一次重构，不是 policy 的事。
6. **异步 checkpoint**：已接成 `--async-save`。与官方还差两点：ColossalAI 的异步写是线程池 + 固定内存，
   不是独立的 Gloo 进程组；保存计划（index 文件）每次重算，没有官方那样的复用与 `dedup_to_lowest_rank`。

优先级建议：先接 VAE 的 fp16 编码（2.46×，精度已过门禁，几乎零风险），再补双机 16 卡的 HSDP 验证
（单机三组已经跑通），然后按"最大可塞样本数"（每卡 8）重测 SAC 三档的吞吐，最后是 VAE 的算子融合与
torch.compile（后者要等离线编译缓存）。
