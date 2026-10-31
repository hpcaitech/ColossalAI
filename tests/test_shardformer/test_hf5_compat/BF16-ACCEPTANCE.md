# HF5 BF16 验收约定

目标：PyTorch 2.11.0+cu128、Transformers 5.17.0。以下约定不代表全部组合已经通过。

## 两种结果分开报告

1. **原生 HF 基准**：同权重、同 BF16、同任务和优化器初始化；保留原断言结果，失败不删除、不用 skip/放宽阈值改写为通过。
2. **独立计算路径控制**：单设备原生 PyTorch/HF 分块线性、分组 backward、明确 attention 后端。用于解释 TP/内核/微批舍入差异，不使用生产适配 forward 作为唯一参考，不自动替代原生基准验收。

## 每次执行必须记录

- Python/Torch/HF/FlashAttention 版本、源码哈希、GPU 编号、命令、退出码和逐 rank PASS/FAIL。
- seed、初始化权重、精度、输入/labels/padding、cache、dropout、attention 实际 backend（不能仅看 config 名）、mask、scale/stride、微批、TP/PP/SP/ZeRO 拓扑。
- 优化器超参数及 FP32 master 初值：BF16-rounded 初始权重的 FP32 副本，不混用未舍入的 reference master。

## 必检内容

- 输出形状、有限值、任务 logits/reconstruction/hidden state、loss dtype 和数值。
- 全部可训练逻辑参数梯度及更新权重。缺失参数、requires_grad 不一致、单侧 grad=None、形状错误均失败。
- 只有存在明确 partition inventory 才 gather 分片；词表 padding/tied aliases 单独记录与验证，不能靠自动截断掩盖形状错误。ZeRO 检查实际 working gradient buffer，不把清空 .grad 当成无需比较。
- 同组复制参数（包括 Qwen3 q/k norm）梯度和更新一致。q/k norm 使用 SUM，不用平均；TP+SP 不重复归约。
- `bf16_checks.py` 是非 ZeRO 全参数比较的公共小工具；当前不自动支持 padding/PP/ZeRO，调用方必须给出对应映射或明确标注未覆盖。

## 数值标准

- 默认 BF16 保持 `atol=rtol=0.005`；FP32 保持原测试阈值。
- 同计算路径输出可额外记录逐位一致，但不要求所有不同内核/累积路径逐位一致。
- 任何新增全参数检查超阈值，专项结论必须失败；原采样断言 PASS 另列，不能混成全参数通过。
- 不仅报告最大误差，也报告超差元素数、均值/relative L2、参考幅度以及首次分歧层。更改阈值必须有独立证据和明确评审，不以测试变绿为理由。

## 最小回归与当前边界

- 固定小模型，至少两个 seed、两个更新步的关键专项；更广拓扑允许单步定向回归。
- CPU/Gloo q/k norm 专项用于同步/累积机制；真实两卡 CUDA/NCCL 验证 TP、split/ring 和可执行 PP/all_to_all 路径。
- TP×PP 或 TP×独立 SP 的四 rank 组合、八卡矩阵仍待最后验收，不能由 CPU/Gloo 模拟替代。
- Whisper ConditionalGeneration 原独立控制按未补齐词表均分，生产 head 则按 51968 个物理行分片（逻辑词表 51866，TP2 每片25984）。固定真实 padding/分片边界及 grouped backward 后，seed42/123 的 logits/loss、全部 decoder 梯度边界精确一致，88 个参数梯度均通过；证据 `logs/2026-10-09-whisper-causal-193000/`。
- `bf16_reference.grouped_column` 提供独立 zero-padding/F.linear/分组反向控制；必须显式传入真实 padding_multiple，不自动从候选层复制实现。完整 native-HF 默认矩阵仍不能据此改写为全部通过；Whisper 多步收敛/四八卡及更广输入范围尚未验收。
