# Gemma 2C2 公共数值工具开发需求与验收

日期：2026-11-01。提交方：2C Gemma 模型线。评审方：2G2 公共数值工具负责人。

本提案用于确认公共工具的开发范围、优先级、接口和验收标准；不在本 PR 实现这些功能，也不表示 Gemma L1/L2 已通过。请先复用已有实现，再补齐下列缺口。框架缺陷另行跟踪，2C 继续负责模型接入和真实 E4B 单卡训练。

## 1. 背景与现有能力

2C2 需要完成 Gemma 4 E2B L1–L2，以及 E4B 单卡纯文本全参训练：全部文本参数含 PLE，视觉/音频冻结。tiny 用于完整数值对齐，真实 E4B 用于训练跑通和资源实测。

- [公共方案](https://docs.google.com/document/d/1S_QODKOM5YsChz6TVRQsG9pKcBHby0Tw4EkFRAhHW_A/edit?tab=t.0) 的 2026-10-30 v4 称整模型 `check_training`、`data_parallel`、`resume` 已实现待提 PR。
- [#6487](https://github.com/hpcaitech/ColossalAI/pull/6487) 已合并，交付算子/模块采集、基准误差判据和 Trace；本轮核查的官方 main 为 `d414d565d2002a8b4e79d810801860d8b0636aa7`，尚无上述整模型场景。
- 下文 PR1/PR2 指公共方案的开发阶段名称，不是实际 PR 编号。本文列出的接口能力是需求，不预设未发布代码缺少相同功能；接口名称可沿用负责人现有设计。

## 2. 任务划分

| 编号 | 优先级 | 内容 | 处理方式 |
| --- | --- | --- | --- |
| D0 | P0 | 固定试用提交、入口和支持矩阵 | 优先交付已有 PR1/PR2 实现 |
| D1 | P0 | CPU/offload Trace 独立快照 | 已合并 core 的可复现问题 |
| D2 | P0 | 非空检查和预期参数、梯度契约 | 确认已有接口，补完整性检查 |
| D3 | P0 | BF16、CPU offload、FP32 master 参考配方 | 确认支持范围，补配方及审计 |
| D4 | P1 | DP 有效 token 加权和判据校准 | 复用现有 DP 场景并补边界 |
| D5 | P1 | Gemini/FSDP 采集、严格续训及 HF 导出 | 按插件适配，复用已有 resume |
| F1/F2 | 单独跟踪 | DDP BF16 包装、ZeRO BF16 master 续训 | 工具提供回归，框架维护者修复 |

### D0 固定试用版本

提供 PR1/PR2 的分支、固定 SHA、使用示例和支持矩阵，覆盖插件、精度、offload、master、梯度检查点及 checkpoint 格式。不支持的组合明确拒绝，不猜测精度配方。

验收：2C 可用 E2B tiny 调用整模型、DP 和续训入口；结果区分通过、失败、不支持、未执行。明确拒绝是合理的工具行为，但不算该模型配置的 L1/L2 通过。

### D1 固定 CPU/offload Trace 快照

当前 `Trace.__init__` 使用 `t.detach().cpu()`；CPU 张量可能与源状态共享存储。采集后执行原地更新，会改写已采集的 before 值；在 `save` 时才 clone 无法恢复旧值。

开发要求：采集时保证快照与后续训练更新独立；如果提供零拷贝选项，应明确其非快照语义。

验收：

- 捕获 CPU 参数、FP32 master、optimizer 状态后执行原地修改或 optimizer step，Trace 保持捕获时值。
- 此后保存并重新读回 Trace，也保持原始值。
- 覆盖非连续张量和共享存储输入；不得因别名导致重复计数或值被后续更新污染。

### D2 检查预期参数与梯度完整性

当前 core 只记录非 None 梯度，再比较已采集的键。如果三路都缺同一项，仅靠比较无法发现；空比较也能返回通过。合法 unused 参数本身不是错误，需要模型声明预期。

开发要求：

- 默认拒绝空比较，或要求显式 `allow_empty` 并在报告中标明零项比较。
- 接受模型准备回调，统一配置冻结规则、attention 实现和非重入梯度检查点；在独立构建的 truth/ref/test 上一致应用。
- 接受预期 trainable、expected gradient、合法 unused/frozen、tied aliases 声明。
- 检查 optimizer 对预期训练参数的身份覆盖；tied 参数按唯一 Parameter 计数，在各自模型内检查别名身份，不比较不同模型的指针是否相同。
- 包装、分片后通过插件适配映射回逻辑参数，不能只比较双方恰好采集到的交集。

验收：声明必须有梯度的 PLE 被遗漏、optimizer 漏参数或 tied 身份破坏时失败；合法冻结/unused 时通过；报告定位逻辑参数名和缺失类型。

### D3 明确 BF16 CPU offload 参考配方

确认或补齐 ZeRO-1/2 的 `precision="bf16"`、`cpu_offload=True`、`master_weights=True` 组合。方案已有 BF16 配方，但 CPU offload 覆盖尚未明确；这不等于其未发布实现一定不支持。

独立参考须匹配相同初始值、BF16 工作权重、FP32 master 和更新语义，审计梯度、master、optimizer moments 的 dtype/设备。不能用 autocast 配方代替整体 BF16 工作权重加 FP32 master；未知组合明确报不支持。

验收：tiny 的 logits、loss、逐参数梯度、一步权重和短程曲线按公共判据对齐，附配方和审计报告。dtype/设备不匹配应在数值判定前清楚报告。

### D4 按有效 token 检查 DP 等价

允许模型提供 loss 的归一化质量或有效 token 计数。causal LM 的分母是 shifted labels 中非 `-100` 的目标 token 总数：需覆盖全部 rank、全部累积 micro-batch，不能简单平均不同长度 batch/rank 的 mean loss。

验收：

- 保留不等长度与 padding，在相同全局 token 数据上比较 1/2 卡 DP 与单卡梯度累积。
- 明确 DDP 的 rank 平均与 token 加权之间的缩放关系。
- 单 rank/micro-batch 零有效 token 的处理不除零、不破坏 collective；全局零有效 token 明确报错。
- 输出逐参数实际误差、阈值和归一化规则；增加错误缩放、漏同步的故障注入测试。
- 文档对 FP32 也采用约 8.3% 相对误差上界；请校准适用性，给出 dtype/层深/归约方案的依据，不直接指定未经验证的新容差。

DP 场景主要比较梯度；一步权重和短程曲线由 `check_training` 检查。不要求所有低精度 Adam DP 测试的更新权重逐位相等。

### D5 补齐分片插件及续训适配

优先提供 Gemini/FSDP 的梯度采集或重建、模型与优化器保存恢复、HF 导出适配，复用已有 `resume` 设计，不要求重写场景。

验收：在确定性配置下比较连续训练 4 步与训练 2 步后保存、全新重建模型/插件/优化器、再训练 2 步；比较后两步 loss 与最终参数逐位一致，dtype 不变。

- 检查 master、Adam moments、scheduler 状态；声明 RNG 和数据进度的保存边界，随机场景须保证恢复一致。
- HF 导出后使用 `from_pretrained` 读回，单列结果，不替代完整续训验收。
- tiny 做完整 L2；真实大模型支持顺序或分片采集，避免默认三份整模同时驻留 GPU。选择性检查必须标注覆盖范围，不能当全量对齐。

## 3. 单独分派的框架问题

| 编号 | 已有线索与证据边界 | 分工与验收 |
| --- | --- | --- |
| F1 | 方案报告 TorchDDP BF16 构造失败；`colossalai/booster/mixed_precision/bf16.py` 只有 pass，基类 configure 为抽象方法，源码与报告一致 | 公共工具 owner 提供回归，mixed-precision/plugin 维护者修复，2C 复测 BF16 覆盖 |
| F2 | 方案报告 ZeRO-1/2 BF16 续训丢失 FP32 master 低位；`colossalai/zero/low_level/low_level_optim.py` 的 update_master_params 从工作权重复制 master。本轮未执行 GPU 续训复现 | 工具 owner 提供固定复现和检测，checkpoint/ZeRO 维护者修复，按 D5 严格续训验收 |

不通过 skip、xfail 或放宽容差将 F1/F2 记为模型验收通过；如果暂不支持，明确报告未完成的验收项。

## 4. CPU 最小复现

下面是观察当前行为的诊断脚本，不是“问题已修复”的回归测试。环境为 Python 3.10.12、Torch 2.13.0+cu130、Transformers 5.17.0、safetensors 0.8.0、0 可见 GPU。实际探针在 `0a373d83` 上退出 0；其 `tests/kit/numerics/` 与上述官方 main 无差异，无需该未发布 Gemma 分支即可复现。

在装好依赖的仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES= python - <<'PY'
import json
import torch
from tests.kit.numerics import Trace, judge, run_fwd_bwd

assert torch.cuda.device_count() == 0
p = torch.tensor([1.])
trace = Trace.capture({"w": p})
p.add_(1.)

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.used = torch.nn.Parameter(torch.tensor([1.]))
        self.unused = torch.nn.Parameter(torch.tensor([2.]))

    def forward(self, x):
        return self.used * x

model = Toy()
inputs = {"x": torch.tensor([3.])}
truth = run_fwd_bwd(model, inputs, torch.float64)
ref = run_fwd_bwd(model, inputs, torch.float32)
test = run_fwd_bwd(model, inputs, torch.float32)
print(json.dumps({
    "trace_after": trace["w"].tolist(),
    "shares_storage": p.data_ptr() == trace["w"].data_ptr(),
    "empty_passed": judge({}, {}, {}).passed,
    "unused_missing_all": all("grad/param/unused" not in t for t in (truth, ref, test)),
    "missing_unused_passed": judge(test, ref, truth).passed,
}, indent=2))
PY
```

当前行为：

```json
{
  "trace_after": [2.0],
  "shares_storage": true,
  "empty_passed": true,
  "unused_missing_all": true,
  "missing_unused_passed": true
}
```

修复目标：Trace 保持 `[1.0]`；默认拒绝空比较；只有模型声明 unused 预期有梯度时才将缺失判为失败，合法 unused 应允许。这些证据针对已合并 core，不证明未发布 PR1/PR2 存在同样缺陷，也不证明 Gemma 实际漏梯度。

## 5. 分工和后续交付

- 2G2 负责公共场景、配方、采集器、判据、支持矩阵和报告。
- 对应框架维护者处理 F1/F2；复现、实现修复分别跟踪。
- 2C 负责 Gemma tiny、policy、labels、冻结和 PLE/KV/tied 具体规则，提供已有 CPU DP/续训样例，执行真实 E4B 单卡训练与 GPU/CPU 峰值实测。

建议分批交付 D0–D3、D4–D5；实现 PR 附最终 SHA、环境、完整命令、GPU 数、FAST_TEST、退出码、通过/失败/跳过、日志/JUnit 和未覆盖项。重跑或容差调整须说明依据。

本需求 PR 的 review 请确认优先级、接口能力和分工；功能完成与模型 L1/L2 验收留给后续实现及实测 PR，不以文档合并代替。
