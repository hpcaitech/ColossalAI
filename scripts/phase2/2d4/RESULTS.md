# 2D4 Ouro 收尾验收（2026-11-18）

**结论：2D4 的开发档 L1/L2 验收已完成。** 两份 Ouro tiny 的数据并行、
共享参数重复调用、梯度检查点和续训检查通过；两份官方完整权重的单卡前向对齐及一次
BF16 ZeRO-2 训练更新也通过。这里不包含原尺寸模型的完整梯度 L2、长序列、收敛或性能验收。

分支：codex/ouro-2d4，基于 feat/support_llada_and_ouro 的
9a9bbc34dc32752631e4e668590769eb7ea59698。原有 #6507 不包含本次新增提交。
[11 月 4 日的原始失败报告](RESULTS-20261104.md) 保留不改。

## 实际交付

- Ouro 支持 HF 非 reentrant、逐层梯度检查点；测试确认循环每层调用 2/4 次，
  backward 确实重算，并保持同一组 Parameters。
- 默认 adapter-class HF reload 自动选择 eager。
- 官方配置没有 pad_token_id 时按 None 构造 embedding，恢复旧 HF 默认语义；
  不把 token 0 错设为 padding。两份无 padding 配置的加载/梯度/HF 回读有回归测试。
- 修复 Gemini CPU offload 在 optimizer.step() 后立即导出时读到旧 BF16 分片的问题：
  待同步时使用最新配对主权重，再转为原导出精度；不移动或同步 live chunk。
- 两份 Ouro 注册到 model_zoo，工厂支持真正的 LazyTensor 延迟构造。
- 使用共享数值工具及独立官方计算参考；只增加严格限定的 Gemini 静态完整 CPU
  offload precision recipe，没有放宽数值阈值。
- 新测试纳入 CI batch 5；固定官方源码/配置以 hash 校验的离线 fixture 提供，无模型权重。
  可复用脚本、机器可读结果、源码指纹和失败日志均保留。

## 环境

- Torch 2.13.0+cu130，Transformers 5.17.0；2 张 H20，物理卡 5/6。
- 容器：wzj-colossalai-2d4-20261118；复用现有镜像、挂载代码、home、缓存和结果目录。
- 仓库：/mnt/beegfs/ColossalAI/wangzhijian/src/ColossalAI，origin 为 hpcaitech/ColossalAI 主仓。
- 容器代码：/workspace/ColossalAI。
- 以下日志相对路径均位于容器 /results/2d4/。
- 宿主机日志根目录：/mnt/beegfs/ColossalAI/wangzhijian/containers/dev-torch213-tf517-peft0212/results/2d4/。
- 未改宿主机代理、网络、驱动配置；未在服务器提交、推送或保存 GitHub 凭据。

## Tiny L1/L2

两份固定快照分别为：
- Ouro-1.4B：574fa66cb8bf5abdc979642d01cf2b79b16bfab1
- Ouro-2.6B：1ed04250da1a9936042725d302e81c8fa2ab5abd

Tiny 保留官方 decoder、循环、gate 和 loss，使用 2 层、hidden 64、词表 128、
batch 2、sequence 16。参考方独立构造官方模型，只在构造期间桥接 HF5 RoPE，
并独立构造三角 attention mask；参考前向不调用生产 adapter。
这不是在单独的 HF4 环境运行的参考。

| 检查 | 结果 |
| --- | --- |
| 2 份快照 × 12 种插件配置，4 次循环、checkpoint、3 步 | 24 用例通过；48 份 rank 报告，每份 59/59 |
| 单卡参考空白对照 | 2 用例通过，每份 59/59 |
| 2 GPU 对 1 GPU、全局 4 个 micro-batch、真实 token 加权 causal CE | 10 配置通过，每份 27/27 梯度 |
| 2+2 步续训对连续 4 步，训练状态及 HF 重载 | 10 配置通过；逐位比较 |
| Lazy Init + checkpoint + 真实 CE，2 快照 × 6 插件 × 1/2 GPU | 24 项均覆盖通过；端口冲突复测见下 |
| Gemini CPU offload step 后立即导出，累积开/关 × 分片/非分片 | 4/4 |
| 原有 Gemini master/resume 回归 | 16/16 |
| 最终 CPU：Ouro、原适配层、D4、数值工具基础回归 | 141/141 |
| 完全空 HF 缓存的最终 Ouro CPU 用例 | 14/14 |
| CI 批次脚本单测 | 84/84 |
| pre-commit | 全部适用检查通过 |

12 种数值配置：DDP FP32/BF16、ZeRO-1 FP32/BF16、ZeRO-2 FP32/BF16、
ZeRO-2 BF16 CPU offload、Gemini 默认 BF16、Gemini 显式分片 BF16、
Gemini 静态完整 CPU offload、FSDP FP32/BF16。

检查包括 logits、实际 LM loss、全部 27 个参数的梯度、更新后的权重及 3 步工具投影 loss。
该短轨迹是工具的合成投影目标，不是语料训练收敛曲线。
两项 early-exit gate 梯度必须有限且非零。Gemini/FSDP 的梯度使用工具既有的
Adam beta1=0 一阶矩提取法。BF16 误差按 FP64 truth 和相同 precision recipe 的参考误差判定。

### 原始 GPU 运行与复测

正式完整矩阵原始结果是 **79 passed、1 failed**，不是单次全绿：
pytest-20261118-041428-97kUjc。

唯一失败为 test_ouro_lazy_checkpoint_step[1-ddp_bf16-ouro14]：
TCPStore 端口 35706 被占用，EADDRINUSE，发生在模型构造前。
采用仓库已有的 rerun_if_address_is_in_use；不重试数值断言或任意模型异常。
之后两份模型 × 1/2 GPU 的 DDP Lazy Init 定向检查 **4/4 通过**：
pytest-20261118-043433-tJ9G71。

因此 80 个不同 GPU 检查项均有通过记录。该原始批次的失败仍保留；
[evidence/20261118/gpu-coverage.json](evidence/20261118/gpu-coverage.json)
记录原始失败、复测项和最终未解决项（为空），不把原始日志改为通过。
70 份独立对照/空白对照/累积/续训数值报告全部通过。

随后补充的 pad_token_id 缺省处理不改变 tiny 显式 pad_token_id=0 的路径。
最终源码再次经过 141 项 CPU、14 项离线 CPU、上述 4 项 GPU 定向检查和两份原尺寸检查。

最终 CPU 日志：pytest-20261118-043302-v5ZPx9；
空缓存日志：pytest-20261118-043251-qW7Ksc；
CI 脚本日志：ci-checks-k79ww3；
格式检查日志：lint-20261118-o2Ej35。

## 官方完整权重

未缩小模型结构。两份模型均使用原始完整权重，严格检查没有 missing/unexpected/mismatched keys。
BF16、单卡、batch 1、sequence 32、4 次循环，随机 token 输入。
参考前向复用官方 decoder/loop 和相同权重，使用独立 mask；tiny 的独立构造对照另行完成。

| 模型 | 参数数 | 原 decoder 层数 | 前向 logits/loss | 反向与更新 |
| --- | ---: | ---: | --- | --- |
| Ouro-1.4B | 1,434,652,673 | 24 | 逐位一致 | 269 个梯度张量全部有限，checkpoint 重算，ZeRO-2 更新通过 |
| Ouro-2.6B | 2,667,974,657 | 48 | 逐位一致 | 533 个梯度张量全部有限，checkpoint 重算，ZeRO-2 更新通过 |

结果目录：
- run-20261118-043249-GcymfW（Ouro-1.4B）
- run-20261118-043544-o9nR6i（Ouro-2.6B）

首次完整加载发现官方配置没有 pad_token_id，HF5 不再提供原来的 None 默认值。
修复前失败记录仍在 run-20261118-043049-1iBZAD 和 run-20261118-043101-pcXbP9。
这也是只用 tiny 配置不足以覆盖真实模型加载的一个实际例子。

## 原有阻塞的处理

11 月 4 日的 Gemini 默认 chunk 多卡累积、ZeRO BF16 master checkpoint、
Gemini named_parameters(remove_duplicate=...) 问题，当前基础分支已包含修复；
本次矩阵全部回归通过。本次新增修复是 HF 默认加载/缺省 padding、
Ouro checkpoint 接入和 Gemini offload 导出读取旧分片。

Gemini 导出问题在修复前的独立回归中稳定复现：
pytest-20261118-040202-3TXWvY。最终 4 项专项回归与 16 项既有 master/resume 回归通过。
没有用修改判断阈值、增加一次刷新前向或关闭 offload 来绕过问题。

## 验收边界与复用

- 接受 eager 全注意力、use_cache=False、默认 full-head RoPE、连续位置。
- Checkpoint 接受逐层 non-reentrant，不含 activation offload 或部分层选择。
- Gemini offload 只验收 static、显式全部分片、完整 CPU optimizer/parameter offload。
- 续训 fixture 是确定性的；应用自身的数据位置和随机数恢复仍由训练入口负责。
- 原尺寸结果是完整权重加载、前向对齐和一次真实训练的 smoke，不是完整原尺寸 L2。
- 不宣称 TP/PP/SP/EP、融合内核、FP8、generation、长序列、收敛或性能验收通过。
- LLaDA 的完整扩散目标、2D2/2D3 不因本次 Ouro 验收而自动完成。

命令见 [README.md](README.md)。
机器可读结果见 [results.json](evidence/20261118/results.json)。
源码指纹见 [sources.json](evidence/20261118/sources.json)，用于核对服务器已验证版本与 Mac 提交文件。
张量 traces 和完整日志保留在服务器持久目录。
本轮是开发容器内验证；新增 CI 条目已配置，尚未在新分支上运行 GitHub Actions。
