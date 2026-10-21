# HF5 长尾模型 policy 门禁说明（审阅稿）

环境版本： PyTorch 2.11.0 / Transformers 5.17.0

## 1. 名单与计数

当前注册 **20 个模型族**。扣除八核心 BERT、Llama、Qwen2、Qwen3、Mixtral、DeepSeek-V3、ViT、Whisper，长尾包含是 **12 个**：

T5、GPT2、GPTJ、OPT、Bloom、SAM、BLIP2、ChatGLM2、Deepseek（旧版、非 V3）、Falcon、Mistral、Command/Cohere。

## 2. 状态矩阵

| 模型族 | 当前状态 | 门禁/后续边界 |
|---|---|---|
| OPT | 修复 HF5 共享 attention 类选择；eager/SDPA 非并行 FP32 输出、loss、全参数梯度、SGD 更新通过 | TP/PP/SP、Flash/JIT/fused/fp8 等仍门禁；基础通过不代表生成、Cache、任意 dropout 或 BF16 通过 |
| Command/Cohere | StaticCache 稳定导入；HF5 基础路径保留原生 attention；eager/SDPA 非并行训练对照通过 | 旧自定义 attention/Cache 与优化/并行路径未放行 |
| BLIP2 | 修正三个已消失 dropout 目标；有效 image placeholder 的 eager/SDPA 非并行训练通过 | TP/Flash/ZBV、parallel dropout 数值/RNG 未验收，继续门禁 |
| ChatGLM2 | policy 可导入，remote-code revision/行为未验证 | HF5 内置 policy 在导入前拒绝；需要固定模型代码版本后单独验收 |
| 旧 Deepseek（非 V3） | 旧 remote-code 注册，不等于 HF5 原生 DeepseekV2 | HF5 内置 policy 门禁；原生 V2 未注册 |
| T5/GPT2/GPTJ/Bloom/Falcon/Mistral | 导入及小配置描述/必需目标路径探针通过 | 不 blanket 禁用或 skip；尚未完成 HF5 数值/并行验收，后续失败必须显露 |
| SAM | CPU get_rng_state 已修；随后被 CPU manual_seed 不支持阻断 | CPU 平台阻断，尚不能归因于 HF5 模型不兼容 |

## 3. 当前门禁逻辑

### 3.1 长尾模型

- 当前门禁只针对 Transformers 5，依据适配测试结果制定。Transformers 4 保留原有运行路径，兼容性尚未重新测试。
- ▎ OPT、Command、BLIP2 的基础适配目前只在 Transformers 5.17.0 上验证通过。低于 5.17.0 的 HF5 版本暂不放行；更高版本仍需测试，不能保证兼容。
- 上述三族有以下请求时拒绝：`enable_tensor_parallelism`、`enable_sequence_parallelism`、`enable_flash_attention`、`enable_jit_fused`、`enable_fused_normalization`、`enable_all_optimization`、`fp8_communication` 或 pipeline stage manager。

### 3.2 入口

- `colossalai/shardformer/_utils.py`：`hf5_policy_error` / `check_hf5_policy`。
- `policies/auto_policy.py`：remote-code 在 policy 懒导入前检查。
- `shard/sharder.py`：自动/显式内置 policy 在模型修改前按配置检查；已覆盖公开入口及 `_tf4`/`_tf5` 内置实现路径。
- 用户自定义 policy / 外部子类不误拦，正确性由调用者负责；未知模型仍保留原“未实现”错误。
- `get_autopolicy` 只选择 policy 时没有完整 shard_config，配置限制最终以实际 ShardFormer 接入检查为准。

## 4. 基础可用配置与报错指引

```python
from colossalai.shardformer import ShardConfig, ShardFormer

config = ShardConfig(enable_tensor_parallelism=False)
model, shared_params = ShardFormer(config).optimize(model)
```

OPT/Command/BLIP2 新基础测试为小配置 FP32、eager/SDPA、dropout 关闭、use_cache=False。BLIP2 需要正确 image_token_id 及 num_query_tokens 个占位 token，并正确 mask label。

报错应包含：模型族、实际 HF 版本、被拒设置、限制原因和文档路径。例如：

```text
ShardFormer opt policy does not support this configuration with transformers 5.17.0:
... requested enable_tensor_parallelism. See docs/HF5-COMPATIBILITY.md.
```

旧版指引：仓库当前声明 `transformers==4.51.3`。若需要未支持 HF5 路径，使用**独立旧环境**并针对目标配置验证，不在共享 HF5 环境原地降级。旧声明不是所有模型/配置都通过的保证。

## 5. 测试、skip 与证据

- D4 正式专项：`tests/test_shardformer/test_hf5_compat/test_d4_compat.py`，**33 passed**，含真实基础训练、配置门禁、import 前拒绝、自定义 policy、未知模型、内置基础 policy 防绕过及核心不误拦。
- D4 当时 Mac/Linux HF5 CPU 回归各 **173 passed、2 skipped**，不是当前最新全套数字。
- 最新 V3 本地 CPU 全套 **206 passed、2 skipped**，见 `../logs/2026-10-14-d5-v3/full-cpu.log`；两个 skip 为需要 CUDA 的 head 专项，不等于 D4 长尾 skip。
- 原分布式入口明确 **9 skipped**：OPT 2、Command 2、BLIP2 1、ChatGLM2 2、旧 Deepseek 2。它们覆盖当前被拒的路径，不记 PASS。
- skip 标记放在旧包装器外层；测试依赖仅在实际执行时加载，避免不支持用例在 skip 前触发无关 model_zoo 导入。未受限制的其他用例不隐藏依赖/运行失败。

证据：

- [D4 初查](../logs/2026-10-12-d4-audit.md) / [原始 JSON](../logs/2026-10-12-d4-audit.json)
- [D4 修复与验证](../logs/2026-10-12-d4-fix/README.md)
- `../logs/2026-10-12-d4-fix/final-focused-base-policy.log`
- `../logs/2026-10-12-d4-fix/linux-skips.log`
- [最新 D5 状态](../logs/2026-10-14-d5-v3/D5-matrix-status.md)

这些是仓库内相对文件链接，尚未变成公开在线日志链接。

## 6. 解除门禁的标准

1. 固定模型族/任务头、HF 版本及 remote-code revision，明确支持配置。
2. 建立同权重、同输入、同精度对照，调用真实 policy/ShardFormer；检查必要替换目标确实命中。
3. 通过输出、loss、有限值、全参数梯度/更新、参数身份测试；Cache/生成、checkpoint 另立专项。
4. 并行必须用相应真实 GPU/拓扑；CPU serial 或描述探针不能替代。
5. 保留修复前 FAIL 与修复后 PASS，不放宽原阈值制造绿灯；独立算术控制与原生 HF 基准分开报告。
6. 仅放行已验证配置，并同步修改 gate、非 skip 验收测试、skip reason、文档与日志；不能先删 gate 再把未运行矩阵标绿。
