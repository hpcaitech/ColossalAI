# LLaDA / LLaDA2 / Ouro 正式接入状态

日期：2026-10-28。
分支：feat/support_llada_and_ouro，从 main d414d565d2002a8b4e79d810801860d8b0636aa7 创建。

## 已实现的第一步

| 路径 | 职责 |
| --- | --- |
| colossalai/shardformer/modeling/llada.py | 独立子类工厂；补齐 HF5 加载元数据和 tie_weights 协议，保留官方初始化与计算 |
| colossalai/shardformer/modeling/llada2_moe.py | 两个 LLaDA2 快照的独立子类工厂；适配 HF5 共享权重映射 |
| colossalai/shardformer/modeling/ouro.py | 独立 LM/Model/RoPE 子类；兼容构造和 HF5 掩码入口，继承官方循环/门控/loss |
| colossalai/shardformer/policies/{llada,llada2_moe,ouro}.py | 最小 native policy，验证支持边界，保持官方计算和参数 |
| colossalai/shardformer/policies/_remote.py | 三个模型族共用的版本和非并行配置检查 |
| colossalai/shardformer/policies/auto_policy.py | 注册适配模型类；原始 LLaDA 类提供明确的构造入口提示 |
| tests/kit/remote_models.py | 五个固定快照的离线 tiny 配置、输入和工厂 |
| tests/test_shardformer/test_model/test_shard_remote_models.py | 实际 policy 接入、计算比较、权重往返、隔离及负例回归 |
| scripts/phase2/2d1/check_adapters.py、run.sh adapters | 可重复运行的容器内测试入口，保存日志和文件 SHA256 |
| docs/HF5-COMPATIBILITY.md | 正式使用方式、版本、固定快照与支持边界 |

生产代码不依赖 scripts 中的探测补丁，不修改 HF 缓存、安装文件或全局 RoPE 注册表。
Ouro 的官方 forward 原本允许传入掩码字典；适配层提前构造正确掩码后复用该入口，
因此无需复制循环、退出门控或 labels 分支。构造入口必须在 ShardFormer.optimize 前调用。

与最初建议相比，新增 LLaDA/LLaDA2 modeling 的原因是保存加载测试发现了真实的 HF5
协议问题；仅做前向、反向无法覆盖这些问题。tiny 工厂暂放 tests/kit，
避免把尚未支持并行的模型直接纳入完整 model_zoo 分布式矩阵。

## 当前验证范围

Transformers 5.17.0、PyTorch 2.13.0+cu130、CPU / FP32 / eager、无 KV cache。
五个 snapshot 各自独立验证；LLaDA2.1/2.2 不因类名相同而合并案例。
native policy 不做 kernel 或并行替换，保留所有原参数对象。

同权重、同输入下比较输出、loss、参数梯度和 AdamW 更新。
Ouro 参考使用官方计算与测试专用初始化桥接，并独立构造三角掩码；
尚未在官方旧版依赖环境中生成完整训练数值基准。
LLaDA/LLaDA2 使用固定掩码位置的 CE，不代表完整扩散目标。

## 后续

1. 2D2：定义并实现正式扩散训练目标（掩码采样、时间步权重等）。
2. 2D3–2D5：GPU/BF16 数值、官方旧版环境参考、真实权重加载和 Booster/ZeRO 接入。
3. 在有独立数值证据后增加 kernel 替换，再扩展 TP/PP/SP/EP；每项增加专门回归。
4. 扩大支持后再把这些 tiny 工厂注册到完整 model_zoo 测试矩阵。

这些实现不改变原 2D1 技术探测报告的证据范围，也不代表 L1/L2 或大模型训练验收完成。

验证记录：2026-10-28，105 passed / 0 failed / 0 skipped。
主日志：/results/2d1/adapters-20261028-121109-MVBu0Z.log。
JUnit：/results/2d1/adapters-validation-20261028-1212.xml。
