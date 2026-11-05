# 2D1 一页侦察报告

日期：2026-10-28。分支：`feat/support_llada_and_ouro`，基点 `d414d565`。
环境：Python 3.10 / Torch 2.13.0+cu130 / Transformers 5.17.0。
状态：**技术侦察完成，可进入正式兼容实现；按任务文档，周会范围确认仍由负责人完成。**

## 范围与结果

全部结果限于 CPU / FP32 / tiny 随机初始化 / 无 KV cache。原始 LLaDA 系列无需适配；
Ouro 的实验补丁仅在探测进程内生效，尚未实现正式支持。

| 官方模型 | 固定快照前缀 | 规模约数¹ | 模型卡许可 | tiny 创建 / 前向 / 反向 / 一步更新 |
|---|---|---:|---|---|
| [GSAI-ML/LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) | 0f2787f2 | 8.016B | MIT | 原始代码通过，21 个参数张量更新 |
| [inclusionAI/LLaDA2.1-mini](https://huggingface.co/inclusionAI/LLaDA2.1-mini) | 20e64e2a | 16.256B | Apache-2.0 | 原始代码通过，28 个参数张量更新 |
| [inclusionAI/LLaDA2.2-mini](https://huggingface.co/inclusionAI/LLaDA2.2-mini) | e4f77d25 | 16.256B | Apache-2.0 | 原始代码通过，31 个参数张量更新 |
| [ByteDance/Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B) | 574fa66c | 1.435B | Apache-2.0 | 诊断适配后通过，27 个参数张量更新；原始创建失败 |
| [ByteDance/Ouro-2.6B](https://huggingface.co/ByteDance/Ouro-2.6B) | 1ed04250 | 2.668B | Apache-2.0 | 诊断适配后通过，27 个参数张量更新；原始创建失败 |

¹按权重 header 的存储元素数近似统计，包含少量 buffer；完整快照见 FINDINGS.md。
缓存共 79 个清单文件、23 个权重分片：文件大小、分片 header/index、离线 tokenizer 均通过。
56 个非权重文件重新计算 SHA256；大权重引用维护者已有的 Hub 校验记录，本次未重新读取全部权重做哈希。

## 版本与官方参考

**本项目当前只验证 5.17.0；上游最低兼容版本未能从公开声明中统一确定，不把推荐版本或 config 保存版本当成下界。**
LLaDA 官方 README 推荐 4.38.2；Ouro 模型卡建议 <4.56.0、推荐 4.54.1。
五个固定 config 中记录的版本依次为 4.46.3 / 4.57.1 / 5.2.0 / 4.55.0 / 4.55.0；这些只是保存时版本。

- LLaDA：[官方 GUIDELINES.md](https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md) 给出加噪、按 mask 概率加权及 SFT 核心损失；[README](https://github.com/ML-GSAI/LLaDA#pre-training-and-supervised-fine-tuning) 明确未发布完整训练框架。
- LLaDA2：[官方 dFactory](https://github.com/inclusionAI/dFactory) 的 `tasks/train_llada2_bd.py`、`configs/sft/llada2_mini_bd_sft.yaml` 是 block-diffusion 参考。其支持表仍把 2.1 标为 Coming Soon，不能视为 2.1/2.2 已完成官方训练支持。
- Ouro：固定 HF 快照中的 `modeling_ouro.py` 提供循环与 labels 训练分支。尚未定位到可确认的原始完整训练框架；项目页链接的 [LoopLM README](https://github.com/rkstgr/LoopLM) 自称重实现，不当作官方预训练参考。

## 单卡配置与开发落点

开发档统一用已验证的 tiny（2 层、隐藏维度 64、词表 128、batch=2、seq=16）；Ouro 为 2 轮。
真实模型候选均从 batch=1、seq=128、BF16 起步，再逐步增加长度；以下是容量估算，不是实测最大配置：
Ouro 1.4B/2.6B 按 16 字节/参数约需 23/43 GB 模型训练状态，可优先安排 H20 单卡；
LLaDA-8B、LLaDA2-mini 采用优化器 CPU offload，按项目估算显存约 32/65 GB、CPU 状态约 96/195 GB，另留激活和运行开销。
当前宿主机有 8 张 H20，**容器 NVML 失败且 RAM 限额仅 16 GiB**；真实 GPU/offload 测试前须修复访问并提高相应内存额度。

正式适配放 `colossalai/shardformer/modeling/{ouro,llada,llada2_moe}.py`，替换规则放对应 `policies/`；
Ouro 需创建前入口，不能仅靠创建后的 policy。方案见 [CODE_PLACEMENT.md](CODE_PLACEMENT.md)，本次未实现这些文件。

## 证据与下一步

一键复跑：`bash /workspace/ColossalAI/scripts/phase2/2d1/run.sh suite`。
最终汇总：`/results/2d1/suite-20261028-084707-3727/summary.json`，10 项预期均满足；
原始 Ouro 的失败是被明确断言的已知阻塞，不计作原始代码可用。
下一步由用户安排正式 modeling 兼容实现；随后推进 2D2 的正式扩散目标和 2D3–2D5 的 L1/L2。
尚未验收：真实权重执行、GPU/BF16、官方数值对齐、KV cache、训练 checkpoint、DP/ZeRO/Gemini/FSDP、收敛。
依据：[阶段任务与 L0 验收标准](https://docs.google.com/document/d/19itx5USHwb-OLy4dEOwSdwPfVhQpGGOmAx0xMbnLBuk/edit?tab=t.0)。
